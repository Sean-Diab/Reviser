"""Prompt lock (notes/promptlock-2026-10-08.md): the first P canvas tokens (the 35-token prompt) can never be edited.

Rule everywhere: a MOVE is legal only if P <= cursor + amt <= len (landing exactly on P is legal); DELETE (removes the
token left of the cursor) is legal only if cursor > P. Inserts and EOR are never masked. Lock off = the old rule
(P = 0 bound), unchanged.

Covers
- training mask (scripts/humanorder/promptlock_patch_v1/): collator cursor/len == an independent executor replay;
  mask boundaries; an illegal gold label raises; masked CE finite, <= unmasked, == CE over the locked legal set;
  legality_state off == the lossmask_patch_v1 collator; check_promptlock.py scan flags illegal supervised labels.
- decoder (scripts/humanorder/decode_eval_slice.py --lock_prompt): a stub model that wants to DELETE / move left
  damages the prompt without the lock and never with it (greedy and sampled); repeated installs do not stack.
- GRPO (scripts/grpo/train_grpo.py --lock_prompt): legal_mask(plen) == hunk_oracle.State(lock).valid; generate() with
  the lock keeps every prompt and stores only legal actions in the allowed sets policy_terms trains on.
- DAgger (scripts/dagger/hunk_oracle.State lock, reviser_lm.valid_mask): lock 0 == the original rule; lock P rule.

The training-mask and decoder tests import a July-control trainer tree, so they run in a subprocess with a temporary
tree on PYTHONPATH (bundle code/src + the patch files), which keeps them independent of other tests' imports.

Run (WSL venv, CPU only): CUDA_VISIBLE_DEVICES= /opt/venv/bin/python tests/test_promptlock.py   (or -m pytest)
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "dagger"))
sys.path.insert(0, str(ROOT / "scripts" / "grpo"))
sys.path.insert(0, str(ROOT / "scripts" / "humanorder" / "promptlock_patch_v1"))

import torch  # noqa: E402

import hunk_oracle as ho  # noqa: E402

DELETE, EOR, MOVE_START, N_MOVES = 50259, 50260, 50261, 20
CODE_SRC = ROOT / "experiments" / "july_control" / "code" / "src"
PATCHED_TRAINER = ROOT / "experiments" / "july_control" / "patches" / "pretrain_resttraj.patched.py"
PL = ROOT / "scripts" / "humanorder" / "promptlock_patch_v1"
_TREES: dict = {}


def legal_trajectory(rng: random.Random, P: int, n_after: int, vocab: int = 300, p_move: float = 0.3,
                     p_del: float = 0.15):
    """P prompt inserts, then n_after random actions that are legal under the prompt lock, then EOR."""
    acts = [rng.randrange(vocab) for _ in range(P)]
    st = ho.State(lock=0)
    for a in acts:
        st.step(a)
    st.lock = P
    for _ in range(n_after):
        u = rng.random()
        if u < p_move:
            opts = [t for t in range(MOVE_START, MOVE_START + N_MOVES) if st.valid(t)]
            if opts:
                # sometimes land exactly on the prompt end, when that is one move away
                exact = [t for t in opts if st.cursor + ho.move_amount(t) == P]
                a = rng.choice(exact) if exact and rng.random() < 0.3 else rng.choice(opts)
                acts.append(a)
                st.step(a)
                continue
        if u < p_move + p_del and st.valid(DELETE):
            acts.append(DELETE)
            st.step(DELETE)
            continue
        a = rng.randrange(vocab)
        acts.append(a)
        st.step(a)
    acts.append(EOR)
    return acts


def _tree(kind: str) -> Path:
    """kind 'pl': bundle code/src + promptlock trainer/collator/dataset; 'work': bundle code/src + patched trainer
    (the decoder tree hq_decode.sh uses)."""
    if kind in _TREES:
        return _TREES[kind]
    d = Path(tempfile.mkdtemp(prefix=f"promptlock_{kind}_"))
    src = d / "src"
    shutil.copytree(CODE_SRC, src, ignore=shutil.ignore_patterns("__pycache__"))
    if kind == "pl":
        shutil.copy(PL / "pretrain_resttraj.py", src / "reviser" / "training" / "pretrain_resttraj.py")
        shutil.copy(PL / "resttraj_pretrain_collator.py", src / "reviser" / "data" / "resttraj_pretrain_collator.py")
        shutil.copy(PL / "raining_resttraj_bucketed_dataset.py",
                    src / "reviser" / "data" / "raining_resttraj_bucketed_dataset.py")
    else:
        shutil.copy(PATCHED_TRAINER, src / "reviser" / "training" / "pretrain_resttraj.py")
    _TREES[kind] = src
    return src


def _run(kind: str, code: str) -> dict:
    """Run `code` in a fresh CPU python with the tree on PYTHONPATH; it must print one JSON object last."""
    env = dict(os.environ, PYTHONPATH=str(_tree(kind)), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    pre = (f"import sys, json, random\nsys.path.insert(0, {str(ROOT / 'tests')!r})\n"
           f"sys.path.insert(0, {str(ROOT / 'scripts' / 'humanorder')!r})\n"
           "import torch\nfrom test_promptlock import legal_trajectory, ho, ROOT, DELETE, EOR, MOVE_START, N_MOVES\n")
    r = subprocess.run([sys.executable, "-c", pre + textwrap.dedent(code)], env=env, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, f"rc={r.returncode}\nSTDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-5000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------------------------------------------
# training mask
# ---------------------------------------------------------------------------------------------------------------

def test_trainer_mask_boundaries_and_fatal_label():
    res = _run("pl", """
        import reviser.training.pretrain_resttraj as T
        V = 50291
        # P = 5; positions: (cursor, len) = (7, 8), (5, 8), (6, 6), (5, 5)
        cur = torch.tensor([[7, 5, 6, 5]]); ln = torch.tensor([[8, 8, 6, 5]])
        batch = {"legal_cursor": cur, "legal_canvas_len": ln, "prefix_lens": torch.tensor([5])}
        lab = torch.full((1, 4), -100)
        z = torch.zeros(1, 4, V)
        m, n_chk, n_bad = T._promptlock_mask_logits(z, lab, batch, where="test")
        mv = lambda a: ho.move_id(a)
        inf = lambda t, tid: bool(torch.isinf(m[0, t, tid]))
        out = {
            "t0_move-2_to_P_legal": not inf(0, mv(-2)), "t0_move-4_illegal": inf(0, mv(-4)),
            "t0_move+1_legal": not inf(0, mv(1)), "t0_move+2_past_end_illegal": inf(0, mv(2)),
            "t0_delete_legal": not inf(0, DELETE),
            "t1_delete_at_P_illegal": inf(1, DELETE), "t1_move-1_illegal": inf(1, mv(-1)), "t1_move+1_legal": not inf(1, mv(1)),
            "t2_delete_at_P+1_legal": not inf(2, DELETE), "t2_move-1_legal": not inf(2, mv(-1)),
            "t2_move-2_illegal": inf(2, mv(-2)), "t2_move+1_illegal": inf(2, mv(1)),
            "t3_only_inserts_eor": int(torch.isinf(m[0, 3]).sum()) == 21 and not inf(3, EOR),
            "inserts_eor_never": not bool(torch.isinf(m[..., :DELETE]).any()) and not bool(torch.isinf(m[..., EOR]).any())
                                 and not bool(torch.isinf(m[..., MOVE_START + N_MOVES:]).any()),
            "input_not_modified": not bool(torch.isinf(z).any()),
        }
        def raises(t, label):
            l2 = torch.full((1, 4), -100); l2[0, t] = label
            try:
                T._promptlock_mask_logits(torch.zeros(1, 4, V), l2, batch, where="test")
                return False
            except RuntimeError as e:
                return "[promptlock]" in str(e)
        out["delete_at_P_raises"] = raises(1, DELETE)
        out["move_left_of_P_raises"] = raises(2, mv(-2))
        out["move_past_end_raises"] = raises(0, mv(2))
        out["move_to_P_ok"] = not raises(0, mv(-2))
        out["delete_at_P+1_ok"] = not raises(2, DELETE)
        out["insert_ok"] = not raises(1, 123)
        print(json.dumps(out))
    """)
    bad = [k for k, v in res.items() if not v]
    assert not bad, bad


def test_collator_state_mask_and_ce():
    res = _run("pl", """
        import importlib.util
        import torch.nn.functional as F
        from pathlib import Path
        import reviser.training.pretrain_resttraj as T
        import reviser.data.resttraj_pretrain_collator as C
        root = ROOT
        spec = importlib.util.spec_from_file_location(
            "reviser.data._lm_collator", root / "scripts/humanorder/lossmask_patch_v1/resttraj_pretrain_collator.py")
        LM = importlib.util.module_from_spec(spec); sys.modules[spec.name] = LM; spec.loader.exec_module(LM)
        rng = random.Random(5); g = torch.Generator().manual_seed(5)
        kw = dict(pad_token_id=50256, max_seq_length=512, max_canvas_length=512, use_canvas=False)
        c_on, c_off, c_lm = C.RestTrajPretrainCollator(**kw, legality_state=True), C.RestTrajPretrainCollator(**kw), LM.RestTrajPretrainCollator(**kw)
        out = {"state_ok": True, "mask_ok": True, "off_eq_lossmask": True, "on_eq_off_shared": True, "ce_ok": True,
               "finite": True, "le_unmasked": True, "labels": 0, "exact_P_moves": 0, "deletes_P+1": 0}
        for rep in range(6):
            samples = []
            for i in range(4):
                P = 35 if rep % 3 else rng.choice([0, 5, 35])
                acts = legal_trajectory(rng, P, rng.randint(20, 300))
                lm = [0] * len(acts)
                for j in range(P, len(acts)):
                    lm[j] = 1 if rng.random() < 0.05 else 0
                samples.append({"action_ids": acts, "prefix_len": P, "loss_mask": lm})
            on, off, lmb = c_on(samples), c_off(samples), c_lm(samples)
            for k in set(off) | set(lmb):
                if k in ("collate_ms", "canvas_ms"):
                    continue
                out["off_eq_lossmask"] &= k in off and k in lmb and bool(torch.equal(off[k], lmb[k]))
                out["on_eq_off_shared"] &= bool(torch.equal(on[k], off[k]))
            assert "legal_cursor" not in off and "legal_cursor" in on
            B, S = on["labels"].shape
            logits = torch.randn(B, S, 50291, generator=g) * 3
            masked, n_chk, n_bad = T._promptlock_mask_logits(logits, on["labels"], on, where="test")
            assert n_bad == 0
            out["labels"] += n_chk
            blk_ids = [DELETE, EOR] + list(range(MOVE_START, MOVE_START + N_MOVES))
            for r, s in enumerate(samples):
                P = s["prefix_len"]; st = ho.State(); n = len(s["action_ids"])
                ref_ill = torch.zeros(n, len(blk_ids), dtype=torch.bool)
                for t, a in enumerate(s["action_ids"]):
                    if int(on["legal_cursor"][r, t]) != st.cursor or int(on["legal_canvas_len"][r, t]) != len(st.canvas):
                        out["state_ok"] = False
                    lk = ho.State(st.canvas, st.cursor, P)          # independent reference: the locked DAgger state
                    for j, tid in enumerate(blk_ids):
                        ref_ill[t, j] = (tid != EOR) and not lk.valid(tid)
                    if t >= P and ho.is_move(a) and st.cursor + ho.move_amount(a) == P:
                        out["exact_P_moves"] += 1
                    if t >= P and a == DELETE and st.cursor == P + 1:
                        out["deletes_P+1"] += 1
                    st.step(a)
                got_ill = torch.isinf(masked[r, :n, DELETE:MOVE_START + N_MOVES])
                out["mask_ok"] &= bool(torch.equal(got_ill, ref_ill))
                out["mask_ok"] &= not bool(torch.isinf(masked[r, :, :DELETE]).any()) and not bool(torch.isinf(masked[r, :, MOVE_START + N_MOVES:]).any())
                lab = on["labels"][r, :n]
                v = lab != -100
                ref_logits = logits[r, :n].clone()
                ref_logits[:, DELETE:MOVE_START + N_MOVES] = ref_logits[:, DELETE:MOVE_START + N_MOVES].masked_fill(ref_ill, float("-inf"))
                ref = F.cross_entropy(ref_logits[v], lab[v], reduction="none")
                got_ce = F.cross_entropy(masked[r, :n][v], lab[v], reduction="none")
                un = F.cross_entropy(logits[r, :n][v], lab[v], reduction="none")
                out["ce_ok"] &= bool(torch.allclose(ref, got_ce, atol=1e-4))
                out["finite"] &= bool(torch.isfinite(got_ce).all())
                out["le_unmasked"] &= bool((got_ce <= un + 1e-5).all())
            ce = F.cross_entropy(masked.view(-1, 50291), on["labels"].view(-1), ignore_index=-100)
            out["finite"] &= bool(torch.isfinite(ce))
        print(json.dumps(out))
    """)
    flags = {k: v for k, v in res.items() if isinstance(v, bool)}
    assert all(flags.values()), res
    assert res["labels"] > 1000 and res["exact_P_moves"] > 0 and res["deletes_P+1"] > 0, res


def test_scan_flags_illegal_supervised_labels():
    import check_promptlock as CP
    rng = random.Random(3)
    P = 35
    good = [legal_trajectory(rng, P, 120) for _ in range(5)]
    bad_del = good[0][:P] + [7, DELETE, DELETE, 8, EOR]                    # 2nd DELETE at cursor P
    bad_mv = good[1][:P] + [7, 8, ho.move_id(-4), 9, EOR]                   # lands at P - 2
    exact = good[2][:P] + [7, 8, ho.move_id(-2), 9, ho.move_id(1), DELETE, EOR]   # move to exactly P; DELETE at P+1
    rows = good + [bad_del, bad_mv, exact]
    with tempfile.TemporaryDirectory() as td:
        for with_mask in (False, True):
            sd = Path(td) / f"m{int(with_mask)}" / "shards"
            sd.mkdir(parents=True)
            toks = np.concatenate([np.asarray(r, np.int32) for r in rows])
            offs, pos = [], 0
            for r in rows:
                offs.append((pos, len(r)))
                pos += len(r)
            np.save(sd / "rest_tokens_shard000000.npy", toks)
            np.save(sd / "rest_offsets_shard000000.npy", np.asarray(offs, np.int64))
            np.save(sd / "prefix_len_shard000000.npy", np.full(len(rows), P, np.uint16))
            if with_mask:   # mask exactly the two illegal labels: they are then unsupervised
                m = np.zeros(len(toks), np.uint8)
                m[offs[5][0] + P + 2] = 1
                m[offs[6][0] + P + 2] = 1
                np.save(sd / "rest_lossmask_shard000000.npy", m)
            res = CP.scan(str(sd.parent))
            assert res["illegal_after_prompt"] == 2, res
            assert res["illegal_supervised"] == (0 if with_mask else 2), res
            assert res["ok"] == with_mask, res
            assert res["moves_to_exactly_P"] >= 1 and res["deletes_at_P_plus_1"] >= 1, res
            assert {(e["row"], e["action"]) for e in res["examples"]} == {(5, DELETE), (6, ho.move_id(-4))}, res


# ---------------------------------------------------------------------------------------------------------------
# decoder
# ---------------------------------------------------------------------------------------------------------------

def test_decoder_lock_keeps_prompt():
    res = _run("work", """
        import torch.nn as nn
        import decode_eval_slice as DES
        import reviser.training.pretrain_resttraj as PR
        from reviser.data.vocabulary import get_move_token_id
        V = 50291

        class Tok:
            def get_move_token_id(self, amt):
                return get_move_token_id(amt)
            def decode(self, ids):
                return ""

        class Stub(nn.Module):
            # wants DELETE most, then moves to the left (largest first), then token 11; EOR never
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))
                b = torch.full((V,), -5.0)
                b[11] = 1.0
                b[DELETE] = 6.0
                for k, a in enumerate([1, 2, 4, 8, 16, 32, 64, 128, 256, 512]):
                    b[ho.move_id(-a)] = 3.0 + 0.1 * k
                b[EOR] = -30.0
                self.register_buffer("b", b)
            def forward(self, input_ids, attention_mask=None, canvas_ids=None, canvas_mask=None, labels=None):
                B, L = input_ids.shape
                return {"logits": self.b.expand(B, L, V).clone()}

        prompt = list(range(100, 135))
        def run(greedy, seed):
            torch.manual_seed(seed)
            _, st = PR._generate_from_action_prefix_nocanvas(model=Stub(), tokenizer=Tok(), prefix_actions=prompt,
                device=torch.device("cpu"), max_new_actions=60, max_seq_len=256, temperature=0.9, top_k=50, greedy=greedy)
            return [int(x) for x in st.canvas], [int(x) for x in st.action_history]
        out = {}
        orig = PR.CursorState
        c0, _ = run(True, 0)
        out["unlocked_greedy_damages"] = c0[:35] != prompt
        out["unlocked_sampled_damaged"] = sum(run(False, s)[0][:35] != prompt for s in range(10))
        prev = DES.install_prompt_lock(35)
        out["install_returns_original"] = prev is orig
        out["installed_subclass"] = issubclass(PR.CursorState, orig) and PR.CursorState is not orig
        c1, a1 = run(True, 0)
        out["locked_greedy_intact"] = c1[:35] == prompt
        out["locked_greedy_new_actions"] = len(a1) - 35
        out["locked_sampled_intact"] = sum(run(False, s)[0][:35] == prompt for s in range(10))
        # every generated action was legal under the lock (replayed with the locked oracle state)
        st = ho.State(prompt, 35, 35); ok = True
        for a in a1[35:]:
            ok &= st.valid(a); st.step(a)
        out["locked_actions_all_legal"] = ok and st.canvas == c1
        # a second install (other prompt length) replaces the lock instead of stacking subclasses
        DES.install_prompt_lock(10)
        out["reinstall_base_is_original"] = PR.CursorState.__mro__[1] is orig and PR.CursorState.PROMPT_LEN == 10
        PR.CursorState = orig
        print(json.dumps(out))
    """)
    assert res["unlocked_greedy_damages"] and res["unlocked_sampled_damaged"] >= 5, res
    assert res["install_returns_original"] and res["installed_subclass"], res
    assert res["locked_greedy_intact"] and res["locked_sampled_intact"] == 10, res
    assert res["locked_actions_all_legal"] and res["locked_greedy_new_actions"] == 60, res
    assert res["reinstall_base_is_original"], res


# ---------------------------------------------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------------------------------------------

def test_grpo_legal_mask_plen():
    import train_grpo as tg
    rng = random.Random(1)
    B = 400
    lens = [rng.randint(0, 600) for _ in range(B)]
    plen = [rng.choice([0, 35, min(35, n)]) if n >= 35 else rng.randint(0, n) for n in lens]
    cur = [rng.randint(p, n) for p, n in zip(plen, lens)]
    cur_t, ln_t, pl_t = torch.tensor(cur), torch.tensor(lens), torch.tensor(plen)
    V = ho.VOCAB_SIZE
    m_lock = tg.legal_mask(cur_t, ln_t, V, "cpu", pl_t)
    m_old = tg.legal_mask(cur_t, ln_t, V, "cpu")
    m_zero = tg.legal_mask(cur_t, ln_t, V, "cpu", torch.zeros(B, dtype=torch.long))
    assert torch.equal(m_old, m_zero)
    ids = [DELETE] + list(range(MOVE_START, MOVE_START + N_MOVES))
    for b in range(B):
        st = ho.State([1] * lens[b], cur[b], plen[b])
        st0 = ho.State([1] * lens[b], cur[b])
        for t in ids:
            assert bool(m_lock[b, t]) == st.valid(t), (b, t, cur[b], lens[b], plen[b])
            assert bool(m_old[b, t]) == st0.valid(t)
    assert m_lock[:, :DELETE].all() and m_lock[:, EOR].all()


class _GrpoStub(torch.nn.Module):
    """forward_step_cached like the GPT-2 Reviser: logits favour DELETE and left moves (the prompt-damaging actions)."""

    def __init__(self):
        super().__init__()
        b = torch.full((ho.VOCAB_SIZE,), -4.0)
        b[11], b[12] = 0.5, 0.5
        b[DELETE] = 3.0
        for a in ho.MOVE_AMOUNTS:
            b[ho.move_id(-a)] = 2.0
        b[EOR] = -2.0
        self.b = b

    def forward_step_cached(self, token_id, canvas_ids_last, canvas_mask_last, past_key_values, position_idx):
        return self.b.expand(token_id.shape[0], -1).clone(), past_key_values


def test_grpo_generate_lock():
    import train_grpo as tg
    rng = random.Random(2)
    prefixes = [[rng.randrange(50000) for _ in range(35)] for _ in range(3)]
    res = {}
    for lock in (False, True):
        gen = torch.Generator().manual_seed(0)
        samples, acts, idx, slot, lp, _ = tg.generate(_GrpoStub(), prefixes, 4, 40, 1.0, 50, "cpu", gen,
                                                      explore_p=0.3, explore_del_p=0.2, lock_prompt=lock)
        intact = sum(s["canvas"][:35] == prefixes[s["prompt"]] for s in samples)
        illegal_in_sets = 0
        for b, s in enumerate(samples):
            st = ho.State(list(prefixes[s["prompt"]]), 35, 35)
            for t, a in enumerate(s["actions"]):
                allowed = [int(x) for x in idx[b, t] if x >= 0]
                assert a in allowed and int(idx[b, t, int(slot[b, t])]) == a
                # move / DELETE entries of the stored allowed set that the prompt lock forbids at this state
                illegal_in_sets += sum((x == DELETE or ho.is_move(x)) and not st.valid(x) for x in allowed)
                if a == EOR:
                    break
                if lock:
                    assert not (a == DELETE or ho.is_move(a)) or st.valid(a)
                st.step(a)
        res[lock] = (intact, illegal_in_sets, len(samples))
    assert res[True][0] == res[True][2] and res[True][1] == 0, res
    assert res[False][0] < res[False][2] and res[False][1] > 0, res     # the stub does damage prompts when unlocked


# ---------------------------------------------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------------------------------------------

def _orig_valid(canvas_len, cursor, tid):
    """hunk_oracle.State.valid before the lock option."""
    if tid == DELETE:
        return cursor > 0
    if ho.is_move(tid):
        return 0 <= cursor + ho.move_amount(tid) <= canvas_len
    return ho.is_insert(tid) or tid == EOR


def test_dagger_state_lock():
    rng = random.Random(4)
    ids = [DELETE, EOR, 5, ho.GPT2_EOT] + list(range(MOVE_START, MOVE_START + N_MOVES))
    for _ in range(300):
        n = rng.randint(0, 80)
        c = rng.randint(0, n)
        st = ho.State([1] * n, c)
        assert st.lock == 0 and st.copy().lock == 0
        for t in ids:
            assert st.valid(t) == _orig_valid(n, c, t)
        P = rng.randint(0, c)
        lk = ho.State([1] * n, c, P)
        assert lk.copy().lock == P
        for t in ids:
            want = (c > P) if t == DELETE else ((P <= c + ho.move_amount(t) <= n) if ho.is_move(t) else st.valid(t))
            assert lk.valid(t) == want
    # step: lock 0 behaves as before; lock P never changes the prompt even for unchecked actions
    prompt = list(range(10, 20))
    a = ho.State(prompt + [1, 2], 12)
    b = ho.State(prompt + [1, 2], 12, 10)
    for t in [DELETE, DELETE, DELETE, ho.move_id(-8), DELETE, 7, ho.move_id(-512), DELETE]:
        a.step(t)
        b.step(t)
    assert b.canvas[:10] == prompt and b.cursor >= 10
    assert a.canvas[:10] != prompt


def test_dagger_valid_mask_lock():
    import reviser_lm as rl
    rng = random.Random(6)
    states, plens = [], []
    for _ in range(200):
        n = rng.randint(0, 600)
        P = rng.randint(0, min(n, 35))
        c = rng.randint(P, n)
        states.append(ho.State([1] * n, c))
        plens.append(P)
    m0 = rl.valid_mask(states, "cpu")
    m_param = rl.valid_mask(states, "cpu", plen=plens)
    m_state = rl.valid_mask([ho.State(s.canvas, s.cursor, p) for s, p in zip(states, plens)], "cpu")
    assert torch.equal(m_param, m_state)
    assert torch.equal(rl.valid_mask(states, "cpu", plen=0), m0)
    ids = [DELETE] + list(range(MOVE_START, MOVE_START + N_MOVES))
    for b, (s, P) in enumerate(zip(states, plens)):
        lk = ho.State(s.canvas, s.cursor, P)
        for t in ids:
            assert bool(m0[b, t]) == _orig_valid(len(s.canvas), s.cursor, t)
            assert bool(m_param[b, t]) == lk.valid(t)
    assert m_param[:, :DELETE].all() and m_param[:, EOR].all()


if __name__ == "__main__":
    # plain runner (the WSL venv has no pytest): python tests/test_promptlock.py
    import time
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)", flush=True)
            except Exception:
                import traceback
                fails += 1
                print(f"FAIL {name}", flush=True)
                traceback.print_exc()
    for p in _TREES.values():
        shutil.rmtree(p.parent, ignore_errors=True)
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
