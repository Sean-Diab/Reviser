"""
Inference utilities for Cursor Transformer.

Used by train.py for periodic qualitative checks during training.
Compatible with the embedding-based canvas model in this workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from reviser.data import (
    SPECIAL_TOKENS,
    get_move_amount,
    get_move_token_id,
    is_insert_token,
    is_move_token,
)
from reviser.model import CursorTransformer


@dataclass
class CursorState:
    prompt_tokens: List[int]
    canvas: List[int]
    cursor_pos: int
    action_history: List[int]

    @classmethod
    def from_prompt(cls, prompt_tokens: List[int]) -> "CursorState":
        return cls(prompt_tokens=prompt_tokens, canvas=[], cursor_pos=0, action_history=[])

    def can_move(self, amount: int) -> bool:
        new_pos = self.cursor_pos + amount
        return 0 <= new_pos <= len(self.canvas)

    def can_delete(self) -> bool:
        return self.cursor_pos > 0

    def execute_action(self, token_id: int) -> bool:
        if is_move_token(token_id):
            amt = get_move_amount(token_id)
            if not self.can_move(amt):
                return False
            self.cursor_pos += amt
            self.action_history.append(token_id)
            return True

        if token_id == SPECIAL_TOKENS.delete:
            if not self.can_delete():
                return False
            self.canvas.pop(self.cursor_pos - 1)
            self.cursor_pos -= 1
            self.action_history.append(token_id)
            return True

        if is_insert_token(token_id):
            self.canvas.insert(self.cursor_pos, token_id)
            self.cursor_pos += 1
            self.action_history.append(token_id)
            return True

        self.action_history.append(token_id)
        return True


def _mask_invalid_actions(logits: torch.Tensor, state: CursorState) -> torch.Tensor:
    logits = logits.clone()

    for amount in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]:
        try:
            tok = get_move_token_id(amount)
        except ValueError:
            continue
        if not state.can_move(amount):
            logits[tok] = float("-inf")

    if not state.can_delete():
        logits[SPECIAL_TOKENS.delete] = float("-inf")

    return logits


def _action_distribution(actions: List[int]) -> dict[str, int]:
    counts = {
        "insert": 0,
        "move": 0,
        "delete": 0,
        "end_of_response": 0,
        "other": 0,
        "total": 0,
    }
    for token_id in actions:
        tid = int(token_id)
        counts["total"] += 1
        if tid == int(SPECIAL_TOKENS.end_of_response):
            counts["end_of_response"] += 1
        elif tid == int(SPECIAL_TOKENS.delete):
            counts["delete"] += 1
        elif is_move_token(tid):
            counts["move"] += 1
        elif is_insert_token(tid):
            counts["insert"] += 1
        else:
            counts["other"] += 1
    return counts


def _fmt_action_distribution(dist: dict[str, int]) -> str:
    total = max(1, int(dist.get("total", 0)))
    pieces = []
    for key in ("insert", "move", "delete", "end_of_response", "other"):
        count = int(dist.get(key, 0))
        pieces.append(f"{key}={count} ({100.0 * count / total:.2f}%)")
    pieces.append(f"total={int(dist.get('total', 0))}")
    return " | ".join(pieces)


def _build_canvas_tensors(
    prompt_tokens: List[int],
    action_history: List[int],
    device: torch.device,
    max_canvas_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    IMPORTANT: Use shifted canvas semantics to match supervised training:
    - At END_OF_INPUT position, canvas must be [CURSOR]
    - In no-EOI mode (DAgger), shift semantics still apply (handled by build_canvas_tensors_shifted)
    """
    from rl.env import build_canvas_tensors_shifted  # local import to avoid circulars

    return build_canvas_tensors_shifted(
        prompt_tokens=list(prompt_tokens),
        action_history=list(action_history),
        device=device,
        max_canvas_len=int(max_canvas_len),
        special_tokens=SPECIAL_TOKENS,
    )


@torch.no_grad()
def generate_response(
    model: CursorTransformer,
    tokenizer,
    prompt_text: str,
    device: torch.device,
    max_steps: int,
    max_canvas_len: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> Tuple[str, CursorState]:
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_tokens.append(SPECIAL_TOKENS.end_of_input)
    state = CursorState.from_prompt(prompt_tokens)

    for _ in range(max_steps):
        full_ids = state.prompt_tokens + state.action_history
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        canvas_ids, canvas_mask = _build_canvas_tensors(
            prompt_tokens=state.prompt_tokens,
            action_history=state.action_history,
            device=device,
            max_canvas_len=max_canvas_len,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            canvas_ids=canvas_ids,
            canvas_mask=canvas_mask,
        )

        logits = out["logits"][:, -1, :].squeeze(0)
        logits = _mask_invalid_actions(logits, state)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            cutoff = topk_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())

        if token_id == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(token_id)
            break

        state.execute_action(token_id)

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


@torch.no_grad()
def generate_from_action_prefix(
    *,
    model: CursorTransformer,
    tokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    max_canvas_len: int,
    temperature: float = 0.8,
    top_k: int = 50,
    allow_eor: bool = True,
    min_new_actions_before_eor: int = 0,
) -> Tuple[str, CursorState]:
    """
    Completion-style inference in the same format as C4 bucketed training:
    - prompt is empty (we start with END_OF_INPUT only)
    - we feed a prefix of *action tokens* after END_OF_INPUT
    - model continues generating action tokens, updating the canvas each step.
    """
    prompt_tokens = [SPECIAL_TOKENS.end_of_input]
    state = CursorState.from_prompt(prompt_tokens)

    # Replay prefix actions to build the starting canvas/history.
    for a in prefix_actions:
        a = int(a)
        if a == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(a)
            break
        state.execute_action(a)

    min_new_actions_before_eor = max(0, int(min_new_actions_before_eor))
    for gen_step in range(int(max_new_actions)):
        full_ids = state.prompt_tokens + state.action_history
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        canvas_ids, canvas_mask = _build_canvas_tensors(
            prompt_tokens=state.prompt_tokens,
            action_history=state.action_history,
            device=device,
            max_canvas_len=max_canvas_len,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            canvas_ids=canvas_ids,
            canvas_mask=canvas_mask,
        )

        logits = out["logits"][:, -1, :].squeeze(0)
        logits = _mask_invalid_actions(logits, state)

        # Match training-time rollouts: do not allow the model to terminate early unless
        # the canvas already matches the target. (For logging we don't always have target,
        # so this is a no-op here; DAgger rollouts enforce it during training.)

        if temperature != 1.0:
            logits = logits / temperature

        if (not bool(allow_eor)) or (gen_step < min_new_actions_before_eor):
            logits[int(SPECIAL_TOKENS.end_of_response)] = float("-inf")

        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            cutoff = topk_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())

        if token_id == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(token_id)
            break
        state.execute_action(token_id)

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


@torch.no_grad()
def generate_from_action_prefix_argmax(
    *,
    model: CursorTransformer,
    tokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    max_canvas_len: int,
    allow_eor: bool = True,
    min_new_actions_before_eor: int = 0,
) -> Tuple[str, CursorState]:
    """
    Same as generate_from_action_prefix, but uses greedy decoding (argmax) instead of sampling.
    """
    prompt_tokens = [SPECIAL_TOKENS.end_of_input]
    state = CursorState.from_prompt(prompt_tokens)

    # Replay prefix actions to build the starting canvas/history.
    for a in prefix_actions:
        a = int(a)
        if a == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(a)
            break
        state.execute_action(a)

    min_new_actions_before_eor = max(0, int(min_new_actions_before_eor))
    for gen_step in range(int(max_new_actions)):
        full_ids = state.prompt_tokens + state.action_history
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        canvas_ids, canvas_mask = _build_canvas_tensors(
            prompt_tokens=state.prompt_tokens,
            action_history=state.action_history,
            device=device,
            max_canvas_len=max_canvas_len,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            canvas_ids=canvas_ids,
            canvas_mask=canvas_mask,
        )

        logits = out["logits"][:, -1, :].squeeze(0)
        logits = _mask_invalid_actions(logits, state)

        if (not bool(allow_eor)) or (gen_step < min_new_actions_before_eor):
            logits[int(SPECIAL_TOKENS.end_of_response)] = float("-inf")

        token_id = int(torch.argmax(logits).item())

        if token_id == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(token_id)
            break
        state.execute_action(token_id)

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


@torch.no_grad()
def generate_from_action_prefix_cached(
    *,
    model: CursorTransformer,
    tokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    max_canvas_len: int,
    temperature: float = 0.8,
    top_k: int = 50,
    allow_eor: bool = True,
    min_new_actions_before_eor: int = 0,
) -> Tuple[str, CursorState]:
    """
    Same interface/semantics as generate_from_action_prefix, but uses model.forward_step_cached_with_hidden
    when available for faster incremental decoding.
    """
    # Fallback if model doesn't support cached stepping.
    if not hasattr(model, "forward_step_cached_with_hidden"):
        return generate_from_action_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix_actions,
            device=device,
            max_new_actions=max_new_actions,
            max_canvas_len=max_canvas_len,
            temperature=temperature,
            top_k=top_k,
            allow_eor=allow_eor,
            min_new_actions_before_eor=min_new_actions_before_eor,
        )

    prompt_tokens = [SPECIAL_TOKENS.end_of_input]
    state = CursorState.from_prompt(prompt_tokens)

    # Prefill cache with END_OF_INPUT (position 0), with canvas=[CURSOR].
    from rl.env import canvas_row_with_cursor_tensors  # local import

    B = 1
    token0 = torch.tensor([int(SPECIAL_TOKENS.end_of_input)], dtype=torch.long, device=device)
    c0, m0 = canvas_row_with_cursor_tensors(
        canvas=[],
        cursor_pos=0,
        device=device,
        max_canvas_len=int(max_canvas_len),
        special_tokens=SPECIAL_TOKENS,
    )
    past_key_values = None
    logits_next, hidden_last, past_key_values = model.forward_step_cached_with_hidden(
        token_id=token0,
        canvas_ids_last=c0.expand(B, -1).contiguous(),
        canvas_mask_last=m0.expand(B, -1).contiguous(),
        past_key_values=past_key_values,
        position_idx=0,
    )
    pos_idx = 0

    # Replay prefix actions, updating canvas and stepping cache.
    for a in prefix_actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response):
            state.action_history.append(a)
            break
        state.execute_action(a)
        row, mask = canvas_row_with_cursor_tensors(
            canvas=list(state.canvas),
            cursor_pos=int(state.cursor_pos),
            device=device,
            max_canvas_len=int(max_canvas_len),
            special_tokens=SPECIAL_TOKENS,
        )
        pos_idx += 1
        logits_next, hidden_last, past_key_values = model.forward_step_cached_with_hidden(
            token_id=torch.tensor([a], dtype=torch.long, device=device),
            canvas_ids_last=row,
            canvas_mask_last=mask,
            past_key_values=past_key_values,
            position_idx=int(pos_idx),
        )

    # Sample max_new_actions more
    min_new_actions_before_eor = max(0, int(min_new_actions_before_eor))
    for gen_step in range(int(max_new_actions)):
        logits = logits_next.squeeze(0)  # (V,)
        logits = _mask_invalid_actions(logits, state)

        if float(temperature) != 1.0:
            logits = logits / float(temperature)

        if (not bool(allow_eor)) or (gen_step < min_new_actions_before_eor):
            logits[int(SPECIAL_TOKENS.end_of_response)] = float("-inf")

        if top_k is not None and int(top_k) > 0:
            k = int(top_k)
            topk_vals, _ = torch.topk(logits, k)
            cutoff = topk_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(torch.nan_to_num(logits, neginf=-1e9, posinf=1e9), dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())

        if token_id == int(SPECIAL_TOKENS.end_of_response):
            state.action_history.append(token_id)
            break

        state.execute_action(token_id)

        row, mask = canvas_row_with_cursor_tensors(
            canvas=list(state.canvas),
            cursor_pos=int(state.cursor_pos),
            device=device,
            max_canvas_len=int(max_canvas_len),
            special_tokens=SPECIAL_TOKENS,
        )
        pos_idx += 1
        logits_next, hidden_last, past_key_values = model.forward_step_cached_with_hidden(
            token_id=torch.tensor([token_id], dtype=torch.long, device=device),
            canvas_ids_last=row,
            canvas_mask_last=mask,
            past_key_values=past_key_values,
            position_idx=int(pos_idx),
        )

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


@torch.no_grad()
def run_inference_test_rollout_data_eoi_prefix_n(
    model: CursorTransformer,
    tokenizer,
    *,
    bucketed_data_dir: str,
    device: str,
    max_canvas_len: int,
    n_examples: int = 3,
    prefix_tokens: int = 5,
    total_max_actions: int = 80,
    temperature: float = 0.9,
    top_k: int = 0,
    seed: int | None = None,
) -> None:
    """
    Training-aligned inference for rollout_data (token_ids schema):
    - prompt_tokens = [END_OF_INPUT]
    - forced prefix actions = first N target tokens (insert actions)
    - generate until total action-history length reaches total_max_actions OR model emits EOR.
    Prints the decoded prefix tokens, model canvas, and ground-truth canvas.
    """
    import random
    import pathlib
    import numpy as np

    rng = random.Random(seed)
    root = pathlib.Path(bucketed_data_dir) / "buckets"
    shards = sorted(root.glob("len_*/*.npz"))
    if not shards:
        print(f"[inference_rollout_data_prefix_n] No shards found under {root}", flush=True)
        return

    dev = torch.device(device)
    pref_n = max(0, int(prefix_tokens))
    total_max = max(1, int(total_max_actions))
    max_new = max(0, total_max - pref_n)
    top_k_i = int(top_k)

    print(
        f"[inference_rollout_data_prefix_n] seed={seed} n_examples={int(n_examples)} "
        f"prefix_tokens={pref_n} total_max_actions={total_max} max_new={max_new} top_k={top_k_i}",
        flush=True,
    )

    shown = 0
    attempts = 0
    while shown < int(n_examples) and attempts < int(n_examples) * 50:
        attempts += 1
        sp = rng.choice(shards)
        z = np.load(sp, allow_pickle=True)
        if "token_ids" not in z:
            continue
        n = int(len(z["token_ids"]))
        if n <= 0:
            continue
        idx = int(rng.randrange(n))
        toks = z["token_ids"][idx]
        if hasattr(toks, "tolist"):
            toks = toks.tolist()
        toks = [int(t) for t in list(toks)]
        if not toks:
            continue

        prefix = [int(t) for t in toks[:pref_n]]
        prefix_text = tokenizer.decode(prefix).replace("\r", "")
        gt_text = tokenizer.decode(toks).replace("\r", "")

        out_text, st = generate_from_action_prefix_cached(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix,
            device=dev,
            max_new_actions=int(max_new),
            max_canvas_len=int(max_canvas_len),
            temperature=float(temperature),
            top_k=top_k_i,
        )

        eor = bool(st.action_history and int(st.action_history[-1]) == int(SPECIAL_TOKENS.end_of_response))
        rel = str(sp.relative_to(root)).replace("\\", "/")

        # Action-type distribution (insert / move / delete) over generated steps.
        from data import is_move_token, is_insert_token  # local import

        def _type_dist(xs: List[int]) -> tuple[int, int, int, int]:
            ins = mov = dele = 0
            total = 0
            for a in xs:
                a = int(a)
                if a == int(SPECIAL_TOKENS.end_of_response):
                    continue
                total += 1
                if a == int(SPECIAL_TOKENS.delete):
                    dele += 1
                elif is_move_token(a):
                    mov += 1
                elif is_insert_token(a):
                    ins += 1
            return ins, mov, dele, total

        # st.action_history includes the forced prefix actions first.
        hist = [int(a) for a in list(st.action_history)]
        gen_hist = hist[int(len(prefix)) :]
        ins_g, mov_g, del_g, tot_g = _type_dist(gen_hist)
        ins_all, mov_all, del_all, tot_all = _type_dist(hist)
        def _pct(n: int, d: int) -> float:
            return 100.0 * float(n) / float(max(1, d))

        print(f"[inference_rollout_data_prefix_n] source={rel} idx={idx}", flush=True)
        print(f"[inference_rollout_data_prefix_n] prefix_tokens={len(prefix)} eor={eor} canvas_len={len(st.canvas)}", flush=True)
        print(
            "[inference_rollout_data_prefix_n] action_mix_generated(excl_EOR): "
            f"insert={_pct(ins_g, tot_g):.2f}% move={_pct(mov_g, tot_g):.2f}% delete={_pct(del_g, tot_g):.2f}% (n={tot_g})",
            flush=True,
        )
        print(
            "[inference_rollout_data_prefix_n] action_mix_all(excl_EOR): "
            f"insert={_pct(ins_all, tot_all):.2f}% move={_pct(mov_all, tot_all):.2f}% delete={_pct(del_all, tot_all):.2f}% (n={tot_all})",
            flush=True,
        )
        print("[inference_rollout_data_prefix_n] ------ forced_prefix_tokens (decoded) ------", flush=True)
        print(prefix_text, flush=True)
        print("[inference_rollout_data_prefix_n] ------ model_canvas ------", flush=True)
        print(out_text.replace("\r", ""), flush=True)
        print("[inference_rollout_data_prefix_n] ------ true_canvas ------", flush=True)
        print(gt_text, flush=True)
        print("", flush=True)

        shown += 1

@torch.no_grad()
def generate_from_action_prefix_no_eoi(
    *,
    model: CursorTransformer,
    tokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    max_canvas_len: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> Tuple[str, CursorState]:
    """
    DAgger-style completion where we DO NOT prepend END_OF_INPUT.
    The sequence starts directly with action tokens (typically initial insert tokens),
    and the model continues generating action tokens.
    """
    prompt_tokens: List[int] = []
    state = CursorState.from_prompt(prompt_tokens)

    # Replay prefix actions to build the starting canvas/history.
    for a in prefix_actions:
        a = int(a)
        if a == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(a)
            break
        state.execute_action(a)

    for _ in range(max_new_actions):
        full_ids = state.prompt_tokens + state.action_history
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        canvas_ids, canvas_mask = _build_canvas_tensors(
            prompt_tokens=state.prompt_tokens,
            action_history=state.action_history,
            device=device,
            max_canvas_len=max_canvas_len,
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            canvas_ids=canvas_ids,
            canvas_mask=canvas_mask,
        )

        logits = out["logits"][:, -1, :].squeeze(0)
        logits = _mask_invalid_actions(logits, state)

        if temperature != 1.0:
            logits = logits / temperature

        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            cutoff = topk_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())

        if token_id == SPECIAL_TOKENS.end_of_response:
            state.action_history.append(token_id)
            break
        state.execute_action(token_id)

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


@torch.no_grad()
def generate_from_action_prefix_no_eoi_cached(
    *,
    model: CursorTransformer,
    tokenizer,
    prefix_actions: List[int],
    device: torch.device,
    max_new_actions: int,
    max_canvas_len: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> Tuple[str, CursorState]:
    """
    Same as generate_from_action_prefix_no_eoi, but uses model.forward_step_cached (KV caching)
    for much faster incremental decoding.
    """
    # Fallback if model doesn't support cached stepping.
    if not hasattr(model, "forward_step_cached"):
        return generate_from_action_prefix_no_eoi(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix_actions,
            device=device,
            max_new_actions=max_new_actions,
            max_canvas_len=max_canvas_len,
            temperature=temperature,
            top_k=top_k,
        )

    from rl.env import canvas_row_with_cursor_tensors  # local import

    state = CursorState.from_prompt([])
    past_key_values = None
    next_logits = None
    pos_idx = 0

    def _step_cached(token_id: int) -> torch.Tensor:
        nonlocal past_key_values, next_logits, pos_idx
        c_last, m_last = canvas_row_with_cursor_tensors(
            canvas=list(state.canvas),
            cursor_pos=int(state.cursor_pos),
            device=device,
            max_canvas_len=int(max_canvas_len),
            special_tokens=SPECIAL_TOKENS,
        )
        # Use bf16 autocast on CUDA for speed (matches training).
        use_amp = (device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            next_logits, past_key_values = model.forward_step_cached(
                token_id=torch.tensor([int(token_id)], dtype=torch.long, device=device),
                canvas_ids_last=c_last,
                canvas_mask_last=m_last,
                past_key_values=past_key_values,
                position_idx=pos_idx,
            )
        pos_idx += 1
        return next_logits[0]

    # Prefill prefix tokens into cache, updating env state AFTER feeding each token.
    for a in prefix_actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response):
            state.action_history.append(a)
            break
        _step_cached(a)
        state.execute_action(a)

    for _ in range(int(max_new_actions)):
        if next_logits is None:
            # No prefix; seed logits by feeding a dummy insert token is not valid.
            # Instead, we require at least 1 prefix action for no-EOI mode.
            break
        logits = next_logits[0]
        logits = _mask_invalid_actions(logits, state)

        if temperature != 1.0:
            logits = logits / temperature
        if top_k is not None and top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            cutoff = topk_vals[-1]
            logits = torch.where(logits < cutoff, torch.full_like(logits, float("-inf")), logits)

        probs = F.softmax(logits, dim=-1)
        token_id = int(torch.multinomial(probs, num_samples=1).item())
        if token_id == int(SPECIAL_TOKENS.end_of_response):
            state.action_history.append(token_id)
            break

        # Advance cache with this token (state BEFORE applying), then apply it.
        _step_cached(token_id)
        state.execute_action(token_id)

    response_text = tokenizer.decode(state.canvas)
    return response_text, state


def _load_random_bucketed_prefix(
    *,
    bucketed_data_dir: str,
    min_prefix: int = 32,
    max_prefix: int = 96,
) -> Tuple[List[int], str]:
    """
    Pick a random bucketed shard sample and return a prefix of its actions (excluding END_OF_INPUT).
    Returns (prefix_actions, source_tag).
    """
    import random as pyrandom
    from pathlib import Path

    import numpy as np

    root = Path(bucketed_data_dir) / "buckets"
    shards = list(root.glob("len_*/*.npz"))
    if not shards:
        raise FileNotFoundError(f"No bucketed shards found under: {root}")
    sp = pyrandom.choice(shards)
    z = np.load(sp, allow_pickle=True)

    n = int(len(z["input_ids"]))
    i = pyrandom.randrange(n)
    inp = z["input_ids"][i]
    if hasattr(inp, "tolist"):
        inp = inp.tolist()
    prompt_len = int(z["prompt_lengths"][i])
    actions = inp[prompt_len:]

    # Remove final EOR if present so we can ask the model to continue.
    if actions and int(actions[-1]) == int(SPECIAL_TOKENS.end_of_response):
        actions = actions[:-1]

    if len(actions) <= 0:
        return [], f"{sp.parent.name}/{sp.name} idx={i}"

    if len(actions) < min_prefix:
        prefix_n = len(actions)
    else:
        prefix_n = pyrandom.randint(min_prefix, min(max_prefix, len(actions)))

    return [int(a) for a in actions[:prefix_n]], f"{sp.parent.name}/{sp.name} idx={i}"


def _load_random_bucketed_actions(*, bucketed_data_dir: str, rng) -> Tuple[List[int], str]:
    """
    Pick a random bucketed shard sample and return its full action sequence (excluding END_OF_INPUT),
    with any trailing END_OF_RESPONSE stripped so the model can continue.
    Returns (actions, source_tag).
    """
    from pathlib import Path

    import numpy as np

    root = Path(bucketed_data_dir) / "buckets"
    shards = list(root.glob("len_*/*.npz"))
    if not shards:
        raise FileNotFoundError(f"No bucketed shards found under: {root}")
    sp = rng.choice(shards)
    z = np.load(sp, allow_pickle=True)

    n = int(len(z["input_ids"]))
    i = rng.randrange(n)
    inp = z["input_ids"][i]
    if hasattr(inp, "tolist"):
        inp = inp.tolist()
    prompt_len = int(z["prompt_lengths"][i])
    actions = inp[prompt_len:]

    # Remove final EOR if present so we can ask the model to continue.
    if actions and int(actions[-1]) == int(SPECIAL_TOKENS.end_of_response):
        actions = actions[:-1]

    return [int(a) for a in actions], f"{sp.parent.name}/{sp.name} idx={i}"


def _load_random_rollout_data_tokens(*, bucketed_data_dir: str, rng) -> Tuple[List[int], str]:
    """
    Pick a random rollout_data shard sample and return its token_ids (raw final text tokens).
    Shard schema:
      token_ids: object[N][List[int]]
      lengths: int16[N]
    """
    from pathlib import Path

    import numpy as np

    root = Path(bucketed_data_dir) / "buckets"
    shards = list(root.glob("len_*/*.npz"))
    if not shards:
        raise FileNotFoundError(f"No bucketed shards found under: {root}")
    sp = rng.choice(shards)
    z = np.load(sp, allow_pickle=True)
    if "token_ids" not in z:
        raise KeyError(f"Expected rollout_data schema with 'token_ids' in {sp} (keys={list(z.keys())})")
    n = int(len(z["token_ids"]))
    i = rng.randrange(n)
    toks = z["token_ids"][i]
    if hasattr(toks, "tolist"):
        toks = toks.tolist()
    toks = [int(t) for t in list(toks)]
    return toks, f"{sp.parent.name}/{sp.name} idx={i}"


def _replay_actions_to_canvas_text(tokenizer, actions: List[int]) -> str:
    st = CursorState.from_prompt([SPECIAL_TOKENS.end_of_input])
    for a in actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response):
            break
        st.execute_action(a)
    return tokenizer.decode(st.canvas)


def _replay_actions_to_canvas_tokens_no_eoi(actions: List[int]) -> List[int]:
    st = CursorState.from_prompt([])
    for a in actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response):
            break
        st.execute_action(a)
    return [int(t) for t in st.canvas]


def run_inference_test_dagger_userprefix(
    model,
    tokenizer,
    *,
    bucketed_data_dir: str,
    device: str = "cuda",
    max_canvas_len: int = 263,
    n_examples: int = 3,
    prefix_min: int = 1,
    prefix_max: int = 7,
    max_new_actions: int = 120,
    seed: int | None = None,
) -> None:
    """
    DAgger-aligned inference (EOI + user-typed prefix, match SL delimiter):
    - Sample a random bucketed shard sample (restoration-action trajectory).
    - Replay actions to get the final canvas tokens (this is our "text"/target token sequence).
    - Choose K~Uniform[prefix_min, prefix_max], take the first K target tokens as initial insert actions (not graded).
    - Run model to generate more actions starting from that prefix (after END_OF_INPUT).

    Prints initial canvas (after K inserts), model canvas, and true final canvas.
    """
    import os
    import random as pyrandom

    model.eval()
    dev = torch.device(device)

    seed_used = int(seed) if seed is not None else (int.from_bytes(os.urandom(8), "little", signed=False) % (2**31 - 1))
    rng = pyrandom.Random(seed_used)
    print(
        f"[inference_dagger_userprefix] seed={seed_used} gen_max={int(max_new_actions)} n_examples={int(n_examples)}",
        flush=True,
    )

    for _ in range(int(n_examples)):
        # Prefer rollout_data schema (token_ids). Fall back to bucketed action-trajectory schema.
        try:
            target_tokens, tag = _load_random_rollout_data_tokens(bucketed_data_dir=bucketed_data_dir, rng=rng)
        except Exception:
            actions, tag = _load_random_bucketed_actions(bucketed_data_dir=bucketed_data_dir, rng=rng)
            target_tokens = _replay_actions_to_canvas_tokens_no_eoi(actions)
        if not target_tokens:
            print(f"[inference_dagger_userprefix] source={tag} (empty target) - skipping", flush=True)
            continue

        # Match DAgger burn-in rules:
        # - <5 tokens: no prefix
        # - 5-9 tokens: K~Uniform[1,3]
        # - >=10 tokens: K~Uniform[1,prefix_max]
        n_tgt = int(len(target_tokens))
        if n_tgt < 5:
            k = 0
        elif n_tgt < 10:
            k = rng.randint(1, min(3, n_tgt))
        else:
            hi = min(max(1, int(prefix_max)), n_tgt)
            k = rng.randint(1, hi)
        prefix = [int(t) for t in target_tokens[:k]]
        init_text = tokenizer.decode(prefix)
        true_text = tokenizer.decode(target_tokens)

        # EOI-based completion (match SL): prompt starts with END_OF_INPUT and canvas at EOI is [CURSOR].
        resp_text, st = generate_from_action_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix,
            device=dev,
            max_new_actions=int(max_new_actions),
            max_canvas_len=int(max_canvas_len),
            temperature=0.9,
            top_k=5,
        )

        def _fmt(s: str) -> str:
            out = s.replace("\n", "\\n")
            if len(out) > 600:
                out = out[:600] + "..."
            return out

        print(f"[inference_dagger_userprefix] source={tag}", flush=True)
        print(
            f"[inference_dagger_userprefix] prefix_tokens={len(prefix)} gen_max={int(max_new_actions)} "
            f"generated_total_actions={len(st.action_history)}",
            flush=True,
        )
        print(
            f"[inference_dagger_userprefix] eor={bool(st.action_history and st.action_history[-1] == SPECIAL_TOKENS.end_of_response)} "
            f"canvas_len={len(st.canvas)}",
            flush=True,
        )
        print("[inference_dagger_userprefix] ------ initial_canvas ------", flush=True)
        print(f"[inference_dagger_userprefix] { _fmt(init_text) }", flush=True)
        print("[inference_dagger_userprefix] ------ model_canvas ------", flush=True)
        print(f"[inference_dagger_userprefix] { _fmt(resp_text) }", flush=True)
        print("[inference_dagger_userprefix] ------ true_canvas ------", flush=True)
        print(f"[inference_dagger_userprefix] { _fmt(true_text) }", flush=True)
        print("", flush=True)


def run_inference_test_bucketed_prefix_k(
    model,
    tokenizer,
    *,
    bucketed_data_dir: str,
    device: str = "cuda",
    max_canvas_len: int = 263,
    n_examples: int = 3,
    prefix_actions: int = 20,
    max_new_actions: int = 120,
    seed: int | None = None,
) -> None:
    """
    Training-aligned inference (no distribution shift):
      [END_OF_INPUT] + first K action tokens from a real shard sample => model continues generating actions.

    For each example we print:
    - initial canvas after replaying the first K actions (ground-truth prefix)
    - model final canvas after continuing
    - true final canvas from the shard sample (replay full action list)
    """
    model.eval()
    dev = torch.device(device)
    k = max(0, int(prefix_actions))
    # Use a per-inference RNG so sampling is reproducible if desired and loggable either way.
    import os
    import random as pyrandom

    seed_used = int(seed) if seed is not None else (int.from_bytes(os.urandom(8), "little", signed=False) % (2**31 - 1))
    rng = pyrandom.Random(seed_used)
    print(
        f"[inference_c4_kprefix] seed={seed_used} prefix_actions={k} gen_max={int(max_new_actions)} n_examples={int(n_examples)}",
        flush=True,
    )

    for _ in range(int(n_examples)):
        actions, tag = _load_random_bucketed_actions(bucketed_data_dir=bucketed_data_dir, rng=rng)
        if not actions:
            print(f"[inference_c4_kprefix] source={tag} (empty actions) - skipping", flush=True)
            continue

        prefix = actions[: min(k, len(actions))]
        init_text = _replay_actions_to_canvas_text(tokenizer, prefix)
        true_text = _replay_actions_to_canvas_text(tokenizer, actions)

        resp_text, st = generate_from_action_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix,
            device=dev,
            max_new_actions=int(max_new_actions),
            max_canvas_len=int(max_canvas_len),
            temperature=0.9,
            top_k=5,
        )

        def _fmt(s: str) -> str:
            out = s.replace("\n", "\\n")
            if len(out) > 600:
                out = out[:600] + "..."
            return out

        print(f"[inference_c4_kprefix] source={tag}", flush=True)
        print(
            f"[inference_c4_kprefix] prefix_actions={len(prefix)} gen_max={int(max_new_actions)} "
            f"generated_total_actions={len(st.action_history)}",
            flush=True,
        )
        print(
            f"[inference_c4_kprefix] eor={bool(st.action_history and st.action_history[-1] == SPECIAL_TOKENS.end_of_response)} "
            f"canvas_len={len(st.canvas)}",
            flush=True,
        )
        print("[inference_c4_kprefix] ------ initial_canvas ------", flush=True)
        print(f"[inference_c4_kprefix] { _fmt(init_text) }", flush=True)
        print("[inference_c4_kprefix] ------ model_canvas ------", flush=True)
        print(f"[inference_c4_kprefix] { _fmt(resp_text) }", flush=True)
        print("[inference_c4_kprefix] ------ true_canvas ------", flush=True)
        print(f"[inference_c4_kprefix] { _fmt(true_text) }", flush=True)
        print("", flush=True)


def run_inference_test_bucketed(
    model,
    tokenizer,
    *,
    bucketed_data_dir: str,
    device: str = "cuda",
    max_canvas_len: int = 263,
    n_examples: int = 3,
    max_new_actions: int = 512,
) -> None:
    """
    Inference test in the same format training uses for bucketed/C4 runs:
    END_OF_INPUT + action-token prefix => model continues generating actions.
    """
    model.eval()
    dev = torch.device(device)

    for _ in range(int(n_examples)):
        prefix, tag = _load_random_bucketed_prefix(
            bucketed_data_dir=bucketed_data_dir,
            min_prefix=32,
            max_prefix=96,
        )
        resp, st = generate_from_action_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix_actions=prefix,
            device=dev,
            max_new_actions=int(max_new_actions),
            max_canvas_len=int(max_canvas_len),
            temperature=0.9,
            top_k=5,
        )

        out = resp.replace("\n", "\\n")
        if len(out) > 600:
            out = out[:600] + "..."

        print(f"[inference_c4] source={tag}", flush=True)
        print(f"[inference_c4] prefix_actions={len(prefix)} total_actions={len(st.action_history)}", flush=True)
        print(
            f"[inference_c4] eor={bool(st.action_history and st.action_history[-1] == SPECIAL_TOKENS.end_of_response)} "
            f"canvas_len={len(st.canvas)}",
            flush=True,
        )
        print(f"[inference_c4] canvas={out}", flush=True)
        print("", flush=True)


def run_inference_test(model, tokenizer, device: str = "cuda", max_canvas_len: int = 263) -> None:
    model.eval()
    dev = torch.device(device)

    prompts = [
        "Hello!",
        "What is 2+2?",
        "Write a 2-line haiku about winter.",
    ]

    for p in prompts:
        resp, st = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt_text=p,
            device=dev,
            max_steps=80,
            max_canvas_len=int(max_canvas_len),
            temperature=0.8,
            top_k=50,
        )
        print(f"[inference] prompt={p!r}", flush=True)
        print(f"[inference] steps={len(st.action_history)} eor={bool(st.action_history and st.action_history[-1] == SPECIAL_TOKENS.end_of_response)}", flush=True)
        print(f"[inference] response={resp!r}", flush=True)
        print("", flush=True)
