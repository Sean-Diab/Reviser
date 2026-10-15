"""
Full Cursor Transformer model.

Combines all components into the complete model for training and inference.
Includes an incremental step API with KV caching for rollout/inference speed.
"""

from typing import Optional, Dict, Tuple, Any, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CursorConfig
from .embeddings import TokenEmbedding, RotaryPositionalEmbedding, CanvasEmbedding
from .layers import TransformerBlockStack


class CursorTransformer(nn.Module):
    """
    Cursor Transformer for edit-based text generation.

    Architecture:
    - Token embedding (tied with output projection)
    - Canvas embedding (attention pooling, added to token embeddings)
    - RoPE positional embeddings (applied in attention)
    - 24 transformer layers (all self-attention only, no cross-attention)
    - Final layer norm
    - Output projection (tied with embedding)
    """

    def __init__(self, config: CursorConfig):
        """
        Initialize the Cursor Transformer.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            initializer_range=config.initializer_range,
        )

        # Positional embeddings
        pe = str(getattr(config, "positional_encoding", "rope")).lower()
        if pe == "rope":
            # RoPE (shared across all layers)
            self.rope: Optional[RotaryPositionalEmbedding] = RotaryPositionalEmbedding(
                dim=config.head_dim,
                max_seq_length=config.max_seq_length,
                theta=config.rope_theta,
            )
            self.position_embedding = None
            self.canvas_position_embedding = None
            # Learned absolute positions for canvas tokens BEFORE attention pooling.
            # This is independent of edit-history positional encoding (RoPE).
            self.canvas_pool_position_embedding = nn.Embedding(config.max_canvas_length, config.d_model)
            nn.init.normal_(self.canvas_pool_position_embedding.weight, mean=0.0, std=config.initializer_range)
        elif pe == "absolute":
            # GPT-2 style learned absolute positional embeddings
            self.rope = None
            self.position_embedding = nn.Embedding(config.max_seq_length, config.d_model)
            self.canvas_position_embedding = nn.Embedding(config.max_canvas_length, config.d_model)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=config.initializer_range)
            nn.init.normal_(self.canvas_position_embedding.weight, mean=0.0, std=config.initializer_range)
            # Alias for consistent naming with rope-mode checkpoints.
            self.canvas_pool_position_embedding = self.canvas_position_embedding
        else:
            raise ValueError(f"Unknown positional_encoding={config.positional_encoding!r}. Expected 'rope' or 'absolute'.")

        # Canvas embedding (attention pooling only, no projection)
        self.canvas_embedding = CanvasEmbedding(
            d_model=config.d_model,
            initializer_range=config.initializer_range,
        )

        # Transformer layers (all self-attention only)
        self.layers = TransformerBlockStack(
            n_layers=config.n_layers,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            head_dim=config.head_dim,
            dropout=config.dropout,
            attention_dropout=config.attention_dropout,
            layer_norm_eps=config.layer_norm_eps,
            activation=config.activation,
            rope=self.rope,
        )

        # Final layer norm
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # Output projection (optionally tied with embedding)
        if config.tie_embeddings:
            self.output_proj = None  # Will use embedding weight
        else:
            self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(config.dropout)

        # Warn at most once if we clamp bad token ids from data (prevents CUDA embedding gather asserts).
        self._warned_token_oob: bool = False

        # Initialize weights
        self.apply(self._init_weights)

    def _sanitize_token_ids(self, ids: torch.Tensor, *, name: str) -> torch.Tensor:
        """
        Clamp ids to [0, vocab_size) before nn.Embedding.

        Out-of-range values (corrupt shards, tokenizer mismatch) otherwise trigger CUDA device-side
        asserts in vectorized_gather inside embedding lookup.
        """
        vs = int(self.config.vocab_size)
        ids = ids.long()
        bad = (ids < 0) | (ids >= vs)
        if bad.any():
            if not self._warned_token_oob:
                warnings.warn(
                    f"CursorTransformer: clamping {int(bad.sum().item())} out-of-range token id(s) "
                    f"in {name} (valid [0, {vs - 1}], saw min={int(ids.min())}, max={int(ids.max())}). "
                    "Verify trajectory shards match the GPT-2 + special-token vocabulary.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_token_oob = True
            ids = ids.clamp(0, vs - 1)
        return ids

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        canvas_ids: Optional[torch.Tensor] = None,
        canvas_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len), 1 = attend
            canvas_ids: Canvas token IDs (batch, seq_len, max_canvas_len)
            canvas_mask: Canvas mask (batch, seq_len, max_canvas_len)
            labels: Target labels for loss computation (batch, seq_len).
                    For standard LM training, labels should already be shifted
                    (predict-next-token) and use -100 to ignore positions.

        Returns:
            Dictionary with:
            - logits: Output logits (batch, seq_len, vocab_size)
            - loss: Cross-entropy loss (if labels provided)
            - hidden_states: Final hidden states (batch, seq_len, d_model) if return_hidden_states=True
        """
        batch_size, seq_len = input_ids.shape

        input_ids = self._sanitize_token_ids(input_ids, name="input_ids")
        if canvas_ids is not None:
            canvas_ids = self._sanitize_token_ids(canvas_ids, name="canvas_ids")

        # Embed input tokens
        token_emb = self.token_embedding(input_ids)  # (batch, seq_len, d_model)
        if self.position_embedding is not None:
            if seq_len > self.config.max_seq_length:
                raise ValueError(f"seq_len={seq_len} exceeds max_seq_length={self.config.max_seq_length} for absolute positions")
            pos = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)  # (1, S)
            token_emb = token_emb + self.position_embedding(pos)  # (B,S,d) + (1,S,d)

        # Compute canvas embeddings if provided
        if canvas_ids is not None and canvas_mask is not None:
            # Embed canvas tokens using the same token embedding matrix
            canvas_token_emb = self.token_embedding(canvas_ids)  # (batch, seq_len, max_canvas_len, d_model)
            if getattr(self, "canvas_pool_position_embedding", None) is not None:
                canvas_len = int(canvas_token_emb.size(2))
                if canvas_len > self.config.max_canvas_length:
                    raise ValueError(
                        f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for canvas positions"
                    )
                cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)  # (C,)
                cpos_emb = self.canvas_pool_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)  # (1,1,C,d)
                canvas_token_emb = canvas_token_emb + cpos_emb
            elif self.canvas_position_embedding is not None:
                canvas_len = int(canvas_token_emb.size(2))
                if canvas_len > self.config.max_canvas_length:
                    raise ValueError(
                        f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for absolute canvas positions"
                    )
                cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)  # (C,)
                cpos_emb = self.canvas_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)  # (1,1,C,d)
                canvas_token_emb = canvas_token_emb + cpos_emb

            # Encode canvas into summary vectors using attention pooling
            canvas_emb = self.canvas_embedding(canvas_token_emb, canvas_mask)  # (batch, seq_len, d_model)

            # Add canvas embedding to (possibly-masked) token embedding
            # Stage 3 spec: x_t = e_t + m_t (no extra scaling term).
            x = token_emb + canvas_emb  # (batch, seq_len, d_model)
        else:
            # No canvas provided, use token embeddings only
            x = token_emb

        # Apply dropout
        x = self.embed_dropout(x)

        # Pass through transformer layers (no cross-attention)
        x = self.layers(
            x,
            attention_mask=attention_mask,
        )

        # Final layer norm
        x = self.final_ln(x)

        # Output projection
        if self.config.tie_embeddings:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.output_proj(x)

        # Compute loss if labels provided
        output = {"logits": logits}
        if return_hidden_states:
            output["hidden_states"] = x

        if labels is not None:
            # Mask out invalid / structurally-illegal targets so they don't contribute to loss.
            # This is intentionally done inside the model so it applies regardless of which
            # training script constructs the shifted labels.
            labels = self._mask_invalid_labels(
                labels=labels,
                attention_mask=attention_mask,
                canvas_ids=canvas_ids,
                canvas_mask=canvas_mask,
            )

            # NOTE: For typical LM training, labels should already be shifted outside the model.
            # Guard against the edge case where *all* labels are ignored (-100), which would
            # otherwise yield NaN loss.
            flat_labels = labels.view(-1)
            if (flat_labels != -100).any():
                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    flat_labels,
                    ignore_index=-100,
                )
            else:
                loss = logits.new_tensor(0.0)
            output["loss"] = loss

        return output

    @torch.no_grad()
    def forward_step_cached(
        self,
        *,
        token_id: torch.Tensor,  # (B,) or (B,1)
        canvas_ids_last: Optional[torch.Tensor],  # (B, C)
        canvas_mask_last: Optional[torch.Tensor],  # (B, C)
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Incremental decoding step with KV caching.

        Given the last observed token_id at position `position_idx`, returns logits for the NEXT token
        plus updated KV cache.

        This mirrors the usual LM semantics: logits at position t predict token t+1.
        """
        self.eval()
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)  # (B,1)
        B, T = token_id.shape
        assert T == 1, "forward_step_cached expects a single token (B,1)"

        device = token_id.device
        attention_mask = torch.ones((B, 1), dtype=torch.long, device=device)

        # Build per-step canvas tensors (B,1,C) from last-row inputs.
        canvas_ids = None
        canvas_mask = None
        if canvas_ids_last is not None and canvas_mask_last is not None:
            canvas_ids = canvas_ids_last.unsqueeze(1)
            canvas_mask = canvas_mask_last.unsqueeze(1)

        # Absolute position for RoPE.
        if position_idx is None:
            # Fallback: assume position 0 if caller doesn't specify.
            pos = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            pos = torch.full((B, 1), int(position_idx), dtype=torch.long, device=device)

        # Use bf16 autocast on CUDA for speed (matches training defaults).
        use_amp = (device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            token_id = self._sanitize_token_ids(token_id, name="token_id")
            if canvas_ids is not None:
                canvas_ids = self._sanitize_token_ids(canvas_ids, name="canvas_ids")
            # Embed input token
            token_emb = self.token_embedding(token_id)  # (B,1,d)
            if self.position_embedding is not None:
                if int(pos.max().item()) >= self.config.max_seq_length:
                    raise ValueError(f"position_idx={int(pos.max().item())} exceeds max_seq_length={self.config.max_seq_length}")
                token_emb = token_emb + self.position_embedding(pos)  # (B,1,d)

            if canvas_ids is not None and canvas_mask is not None:
                canvas_token_emb = self.token_embedding(canvas_ids)  # (B,1,C,d)
                if getattr(self, "canvas_pool_position_embedding", None) is not None:
                    canvas_len = int(canvas_token_emb.size(2))
                    if canvas_len > self.config.max_canvas_length:
                        raise ValueError(
                            f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for canvas positions"
                        )
                    cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)
                    cpos_emb = self.canvas_pool_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)
                    canvas_token_emb = canvas_token_emb + cpos_emb
                elif self.canvas_position_embedding is not None:
                    canvas_len = int(canvas_token_emb.size(2))
                    if canvas_len > self.config.max_canvas_length:
                        raise ValueError(
                            f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for absolute canvas positions"
                        )
                    cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)
                    cpos_emb = self.canvas_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)
                    canvas_token_emb = canvas_token_emb + cpos_emb
                canvas_emb = self.canvas_embedding(canvas_token_emb, canvas_mask)  # (B,1,d)
                x = token_emb + canvas_emb
            else:
                x = token_emb

            # No dropout in eval; keep deterministic.
            x, present = self.layers(
                x,
                attention_mask=attention_mask,
                positions=pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            x = self.final_ln(x)
            if self.config.tie_embeddings:
                logits = F.linear(x, self.token_embedding.weight)  # (B,1,V)
            else:
                logits = self.output_proj(x)
        return logits[:, 0, :], present

    @torch.no_grad()
    def forward_step_cached_with_hidden(
        self,
        *,
        token_id: torch.Tensor,  # (B,) or (B,1)
        canvas_ids_last: Optional[torch.Tensor],  # (B, C)
        canvas_mask_last: Optional[torch.Tensor],  # (B, C)
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Same as forward_step_cached, but also returns the final hidden state for the current token.

        Returns:
          - logits_next: (B, V) logits that predict the NEXT token
          - hidden_last: (B, d_model) hidden state at the current position (after final LN)
          - present_kv: updated KV cache
        """
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)  # (B,1)
        B, T = token_id.shape
        assert T == 1, "forward_step_cached_with_hidden expects a single token (B,1)"

        device = token_id.device
        attention_mask = torch.ones((B, 1), dtype=torch.long, device=device)

        canvas_ids = None
        canvas_mask = None
        if canvas_ids_last is not None and canvas_mask_last is not None:
            canvas_ids = canvas_ids_last.unsqueeze(1)
            canvas_mask = canvas_mask_last.unsqueeze(1)

        if position_idx is None:
            pos = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            pos = torch.full((B, 1), int(position_idx), dtype=torch.long, device=device)

        # Use bf16 autocast on CUDA for speed.
        use_amp = (device.type == "cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            token_id = self._sanitize_token_ids(token_id, name="token_id")
            if canvas_ids is not None:
                canvas_ids = self._sanitize_token_ids(canvas_ids, name="canvas_ids")
            token_emb = self.token_embedding(token_id)  # (B,1,d)
            if self.position_embedding is not None:
                if int(pos.max().item()) >= self.config.max_seq_length:
                    raise ValueError(f"position_idx={int(pos.max().item())} exceeds max_seq_length={self.config.max_seq_length}")
                token_emb = token_emb + self.position_embedding(pos)  # (B,1,d)

            if canvas_ids is not None and canvas_mask is not None:
                canvas_token_emb = self.token_embedding(canvas_ids)  # (B,1,C,d)
                if getattr(self, "canvas_pool_position_embedding", None) is not None:
                    canvas_len = int(canvas_token_emb.size(2))
                    if canvas_len > self.config.max_canvas_length:
                        raise ValueError(
                            f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for canvas positions"
                        )
                    cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)
                    cpos_emb = self.canvas_pool_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)
                    canvas_token_emb = canvas_token_emb + cpos_emb
                elif self.canvas_position_embedding is not None:
                    canvas_len = int(canvas_token_emb.size(2))
                    if canvas_len > self.config.max_canvas_length:
                        raise ValueError(
                            f"canvas_len={canvas_len} exceeds max_canvas_length={self.config.max_canvas_length} for absolute canvas positions"
                        )
                    cpos = torch.arange(canvas_len, device=canvas_ids.device, dtype=torch.long)
                    cpos_emb = self.canvas_position_embedding(cpos).view(1, 1, canvas_len, self.config.d_model)
                    canvas_token_emb = canvas_token_emb + cpos_emb
                canvas_emb = self.canvas_embedding(canvas_token_emb, canvas_mask)  # (B,1,d)
                x = token_emb + canvas_emb
            else:
                x = token_emb

            x, present = self.layers(
                x,
                attention_mask=attention_mask,
                positions=pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            x = self.final_ln(x)  # (B,1,d)
            hidden_last = x[:, 0, :]  # (B,d)
            if self.config.tie_embeddings:
                logits = F.linear(x, self.token_embedding.weight)  # (B,1,V)
            else:
                logits = self.output_proj(x)
            logits_next = logits[:, 0, :]  # (B,V)
        return logits_next, hidden_last, present

    # ---------------------------------------------------------------------
    # Invalid-token handling (training + inference)
    # ---------------------------------------------------------------------
    @staticmethod
    def _special_token_ids() -> Dict[str, int]:
        """
        Token ID mapping for Cursor special tokens.

        These IDs are determined by `data/tokenizer.py` which adds 34 tokens to GPT-2 (50257)
        in a fixed order: CURSOR, END_OF_INPUT, DELETE, END_OF_RESPONSE, then move tokens,
        then 10 reserved tokens.
        """
        GPT2_VOCAB_SIZE = 50257
        return {
            "gpt2_eot": GPT2_VOCAB_SIZE - 1,      # 50256
            "cursor": GPT2_VOCAB_SIZE + 0,       # 50257
            "end_of_input": GPT2_VOCAB_SIZE + 1, # 50258
            "delete": GPT2_VOCAB_SIZE + 2,       # 50259
            "end_of_response": GPT2_VOCAB_SIZE + 3,  # 50260
            "move_start": GPT2_VOCAB_SIZE + 4,   # 50261
            "reserved_start": GPT2_VOCAB_SIZE + 24,  # 50281 (after 20 move tokens)
            "reserved_end_excl": GPT2_VOCAB_SIZE + 34,  # 50291
        }

    @staticmethod
    def _move_amounts() -> torch.Tensor:
        # Powers of 2 up to 512, in the same order as `data/tokenizer.py`.
        return torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 256, 512], dtype=torch.long)

    def _mask_invalid_labels(
        self,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        canvas_ids: Optional[torch.Tensor],
        canvas_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Return a copy of `labels` with invalid targets set to -100 so they are ignored.

        We treat the following as invalid targets:
        - GPT-2 <|endoftext|> (50256): used as PAD in this repo; also not a valid "insert"
        - [CURSOR], [END_OF_INPUT], [RESERVED_*]: structural / untrained
        - DELETE when cursor is at position 0
        - MOVE +/-k when that move would take cursor out of bounds given the canvas state
        """
        ids = self._special_token_ids()
        out = labels.clone()

        # Ignore padding positions (if attention_mask is provided)
        if attention_mask is not None:
            out = out.masked_fill(attention_mask == 0, -100)

        # Always ignore GPT-2 EOT as a target.
        out = out.masked_fill(out == ids["gpt2_eot"], -100)

        # Always ignore structural / reserved tokens as targets.
        out = out.masked_fill(out == ids["cursor"], -100)
        out = out.masked_fill(out == ids["end_of_input"], -100)
        out = out.masked_fill((out >= ids["reserved_start"]) & (out < ids["reserved_end_excl"]), -100)

        # If we have canvas state, also ignore invalid DELETE/MOVE given cursor bounds.
        if canvas_ids is None or canvas_mask is None:
            return out

        # canvas_ids: (B, T, C)
        # canvas_mask: (B, T, C) with 1 for valid tokens
        cursor_tok = ids["cursor"]
        delete_tok = ids["delete"]

        # canvas_len_with_cursor = number of masked-in tokens in each row
        canvas_len_with_cursor = canvas_mask.long().sum(dim=-1)  # (B, T)
        content_len = torch.clamp(canvas_len_with_cursor - 1, min=0)  # (B, T)

        # cursor_idx = index of cursor token (0..C-1). If missing, this will be 0.
        cursor_matches = (canvas_ids == cursor_tok) & (canvas_mask != 0)
        has_cursor = cursor_matches.any(dim=-1)  # (B, T)
        cursor_idx = cursor_matches.long().argmax(dim=-1)  # (B, T)
        cursor_idx = torch.where(has_cursor, cursor_idx, torch.zeros_like(cursor_idx))

        # DELETE invalid at cursor_idx == 0
        can_delete = cursor_idx > 0
        out = torch.where((out == delete_tok) & (~can_delete), torch.full_like(out, -100), out)

        # MOVE validity checks
        move_amounts = self._move_amounts().to(device=out.device)
        move_start = ids["move_start"]
        # token IDs for +k and -k for each amount (10 each)
        idxs = torch.arange(move_amounts.numel(), device=out.device, dtype=torch.long)
        move_pos_ids = move_start + idxs * 2
        move_neg_ids = move_pos_ids + 1

        # For each move amount, mask targets that would go OOB.
        for amt, pos_id, neg_id in zip(move_amounts.tolist(), move_pos_ids.tolist(), move_neg_ids.tolist()):
            # +amt invalid if cursor_idx + amt > content_len
            invalid_pos = (out == pos_id) & (cursor_idx + amt > content_len)
            # -amt invalid if cursor_idx < amt
            invalid_neg = (out == neg_id) & (cursor_idx < amt)
            out = out.masked_fill(invalid_pos | invalid_neg, -100)

        return out

    def _mask_invalid_action_logits_last(
        self,
        logits: torch.Tensor,
        canvas_ids_last: Optional[torch.Tensor],
        canvas_mask_last: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Mask invalid tokens in logits for inference at the current (last) timestep.
        `logits`: (B, V)
        `canvas_ids_last`: (B, C) canvas tokens *with cursor inserted*
        `canvas_mask_last`: (B, C)
        """
        ids = self._special_token_ids()
        out = logits.clone()

        # Ban EOT + structural + reserved tokens from being sampled as actions.
        out[:, ids["gpt2_eot"]] = float("-inf")
        out[:, ids["cursor"]] = float("-inf")
        out[:, ids["end_of_input"]] = float("-inf")
        out[:, ids["reserved_start"]:ids["reserved_end_excl"]] = float("-inf")

        if canvas_ids_last is None or canvas_mask_last is None:
            return out

        cursor_tok = ids["cursor"]
        delete_tok = ids["delete"]

        canvas_len_with_cursor = canvas_mask_last.long().sum(dim=-1)  # (B,)
        content_len = torch.clamp(canvas_len_with_cursor - 1, min=0)  # (B,)
        cursor_matches = (canvas_ids_last == cursor_tok) & (canvas_mask_last != 0)
        has_cursor = cursor_matches.any(dim=-1)  # (B,)
        cursor_idx = cursor_matches.long().argmax(dim=-1)  # (B,)
        cursor_idx = torch.where(has_cursor, cursor_idx, torch.zeros_like(cursor_idx))

        # Mask DELETE if invalid
        can_delete = cursor_idx > 0
        out[:, delete_tok] = torch.where(can_delete, out[:, delete_tok], torch.full_like(out[:, delete_tok], float("-inf")))

        # Mask invalid moves for each batch element
        move_amounts = self._move_amounts().to(device=out.device)
        move_start = ids["move_start"]
        idxs = torch.arange(move_amounts.numel(), device=out.device, dtype=torch.long)
        move_pos_ids = move_start + idxs * 2
        move_neg_ids = move_pos_ids + 1

        for amt, pos_id, neg_id in zip(move_amounts.tolist(), move_pos_ids.tolist(), move_neg_ids.tolist()):
            invalid_pos = cursor_idx + amt > content_len
            invalid_neg = cursor_idx < amt
            out[:, pos_id] = torch.where(invalid_pos, torch.full_like(out[:, pos_id], float("-inf")), out[:, pos_id])
            out[:, neg_id] = torch.where(invalid_neg, torch.full_like(out[:, neg_id], float("-inf")), out[:, neg_id])

        return out

    def generate_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        canvas_ids: Optional[torch.Tensor] = None,
        canvas_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a single step (for inference).

        Args:
            input_ids: Current sequence (batch, seq_len)
            attention_mask: Attention mask
            canvas_ids: Current canvas states
            canvas_mask: Canvas mask
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering

        Returns:
            Tuple of (next_token, logits)
        """
        # Forward pass
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            canvas_ids=canvas_ids,
            canvas_mask=canvas_mask,
        )

        # Get logits for last position
        logits = outputs["logits"][:, -1, :]  # (batch, vocab_size)

        # Mask invalid tokens using the current canvas state (if provided).
        # This makes inference safer even if the caller forgets to apply masking.
        if canvas_ids is not None and canvas_mask is not None:
            logits = self._mask_invalid_action_logits_last(
                logits=logits,
                canvas_ids_last=canvas_ids[:, -1, :],
                canvas_mask_last=canvas_mask[:, -1, :],
            )

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_top_k,
                torch.full_like(logits, float("-inf")),
                logits,
            )

        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            # Scatter back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        return next_token.squeeze(-1), logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        end_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a complete response.

        Note: This is a simplified version. Full inference requires
        tracking the canvas state as we execute actions.

        Args:
            prompt_ids: Prompt token IDs (batch, prompt_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            end_token_id: Token ID that signals end of generation

        Returns:
            Generated sequence (batch, prompt_len + generated_len)
        """
        self.eval()
        device = prompt_ids.device
        batch_size = prompt_ids.size(0)

        # Start with prompt
        generated = prompt_ids

        for _ in range(max_new_tokens):
            # Get next token
            next_token, _ = self.generate_step(
                input_ids=generated,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            # Check for end token
            if end_token_id is not None:
                if (next_token == end_token_id).all():
                    break

        return generated

    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(config: Optional[CursorConfig] = None) -> CursorTransformer:
    """
    Factory function to create a Cursor Transformer.

    Args:
        config: Optional configuration. Uses default if not provided.

    Returns:
        CursorTransformer instance
    """
    if config is None:
        config = CursorConfig()
    return CursorTransformer(config)
