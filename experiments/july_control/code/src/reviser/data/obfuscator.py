"""
Obfuscation and restoration trajectory generation for Cursor transformer.

This module implements the obfuscation process that corrupts a target sequence
and generates restoration trajectories for training.
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from .vocabulary import (
    SPECIAL_TOKENS,
    MOVE_AMOUNTS,
    get_move_token_id,
    decompose_move,
    GPT2_VOCAB_SIZE,
    GPT2_EOT_TOKEN_ID,
)


class ActionType(Enum):
    """Types of edit actions."""
    INSERT = "insert"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class ObfuscationStep:
    """A single step in the obfuscation trajectory."""
    action: ActionType
    # For INSERT: the token that was inserted
    # For DELETE: the token that was deleted
    # For MOVE: the move amount
    value: int
    # Cursor position before this action
    cursor_pos: int


@dataclass
class TrainingSample:
    """A complete training sample with input, labels, and canvas states."""
    # Full input sequence: prompt + END_OF_INPUT + restoration trajectory
    input_ids: List[int]
    # Labels: same as input_ids but prompt masked with -100
    labels: List[int]
    # True canvas state at each timestep (for cross-attention)
    # canvas_states[t] is the canvas after executing actions 0..t-1
    canvas_states: List[List[int]]
    # Length of prompt (including END_OF_INPUT)
    prompt_length: int


class Obfuscator:
    """
    Generates obfuscation and restoration trajectories for training.

    The obfuscation process corrupts a sequence using:
    - 70% delete: removes a token
    - 18% move: moves cursor to a random valid position
    - 12% insert: inserts a random token

    When reversed, the restoration trajectory becomes:
    - 70% insert: restores deleted tokens (from obfuscation deletes)
    - 18% move: reverses cursor movements
    - 12% delete: removes junk tokens (from obfuscation inserts)
    """

    def __init__(
        self,
        delete_prob: float = 0.70,
        move_prob: float = 0.18,
        insert_prob: float = 0.12,
        after_insert_delete_prob: Optional[float] = None,
        after_insert_move_prob: Optional[float] = None,
        after_insert_insert_prob: Optional[float] = None,
        max_edit_steps: int = 1000,
        vocab_size: int = GPT2_VOCAB_SIZE,
        seed: Optional[int] = None,
    ):
        """
        Initialize the obfuscator.

        Args:
            delete_prob: Probability of delete action during obfuscation
            move_prob: Probability of move action during obfuscation
            insert_prob: Probability of insert action during obfuscation
            after_insert_delete_prob: Optional override delete probability for the *next* action
                immediately following an INSERT. If provided, must be provided with the other
                after_insert_* probs and they must sum to 1.
            after_insert_move_prob: See after_insert_delete_prob.
            after_insert_insert_prob: See after_insert_delete_prob.
            max_edit_steps: Maximum number of edit steps allowed
            vocab_size: Size of vocabulary for random token insertion
            seed: Random seed for reproducibility
        """
        assert abs(delete_prob + move_prob + insert_prob - 1.0) < 1e-6, "Probabilities must sum to 1"

        self.delete_prob = delete_prob
        self.move_prob = move_prob
        self.insert_prob = insert_prob
        self.after_insert_delete_prob = after_insert_delete_prob
        self.after_insert_move_prob = after_insert_move_prob
        self.after_insert_insert_prob = after_insert_insert_prob
        self.max_edit_steps = max_edit_steps
        self.vocab_size = vocab_size

        if seed is not None:
            random.seed(seed)

        # Validate after-insert distribution if any part is provided.
        any_after = any(x is not None for x in (after_insert_delete_prob, after_insert_move_prob, after_insert_insert_prob))
        if any_after:
            if after_insert_delete_prob is None or after_insert_move_prob is None or after_insert_insert_prob is None:
                raise ValueError("If using after_insert_* probabilities, you must provide all three.")
            s = float(after_insert_delete_prob) + float(after_insert_move_prob) + float(after_insert_insert_prob)
            if abs(s - 1.0) > 1e-6:
                raise ValueError(f"after_insert_* probabilities must sum to 1 (got {s}).")

    def obfuscate(
        self,
        tokens: List[int],
        initial_cursor_pos: int = 0,
        num_obfuscation_steps: Optional[int] = None,
    ) -> Tuple[List[int], int, List[ObfuscationStep]]:
        """
        Obfuscate a sequence by applying random edits.

        Args:
            tokens: Original token sequence to obfuscate
            initial_cursor_pos: Starting cursor position (default 0)
            num_obfuscation_steps: Number of obfuscation steps to apply.
                If None, obfuscates until canvas is empty.
                If specified, applies exactly this many steps (cursor ends at random position).

        Returns:
            Tuple of:
            - Final obfuscated sequence
            - Final cursor position
            - List of obfuscation steps (to be reversed for restoration)

        Raises:
            ValueError: If max_edit_steps is exceeded before completion
        """
        canvas = list(tokens)
        cursor_pos = initial_cursor_pos
        steps: List[ObfuscationStep] = []
        prev_action: Optional[ActionType] = None

        # Determine target: either fixed number of steps or until empty
        if num_obfuscation_steps is not None:
            target_steps = min(num_obfuscation_steps, self.max_edit_steps)
            obfuscate_until_empty = False
        else:
            target_steps = self.max_edit_steps
            obfuscate_until_empty = True

        for step_idx in range(target_steps):
            # Check termination condition
            if obfuscate_until_empty and len(canvas) == 0:
                break
            if not obfuscate_until_empty and step_idx >= num_obfuscation_steps:
                break

            # If canvas is empty and we're doing fixed steps, we can't continue
            if len(canvas) == 0:
                break

            # Sample action, respecting validity constraints
            action = self._sample_valid_action(canvas, cursor_pos, prev_action=prev_action)

            if action == ActionType.DELETE:
                deleted_token = canvas[cursor_pos - 1]
                canvas.pop(cursor_pos - 1)
                steps.append(ObfuscationStep(
                    action=ActionType.DELETE,
                    value=deleted_token,
                    cursor_pos=cursor_pos,
                ))
                cursor_pos -= 1
                prev_action = action

            elif action == ActionType.MOVE:
                # Sample a valid move amount
                move_amount = self._sample_move(canvas, cursor_pos)
                if move_amount != 0:
                    steps.append(ObfuscationStep(
                        action=ActionType.MOVE,
                        value=move_amount,
                        cursor_pos=cursor_pos,
                    ))
                    cursor_pos += move_amount
                prev_action = action

            elif action == ActionType.INSERT:
                # Insert a random token at cursor position
                # Avoid GPT-2 "<|endoftext|>" (50256). If it appears in training targets,
                # the model learns to insert it frequently, which looks like "junk" and
                # harms edit rollouts.
                if self.vocab_size == GPT2_VOCAB_SIZE:
                    random_token = random.randint(0, GPT2_VOCAB_SIZE - 2)  # [0..50255]
                else:
                    random_token = random.randint(0, self.vocab_size - 1)
                    if random_token == GPT2_EOT_TOKEN_ID and self.vocab_size > 1:
                        random_token = max(0, random_token - 1)
                canvas.insert(cursor_pos, random_token)
                steps.append(ObfuscationStep(
                    action=ActionType.INSERT,
                    value=random_token,
                    cursor_pos=cursor_pos,
                ))
                cursor_pos += 1  # Cursor moves right after insertion
                prev_action = action

        # Only raise error if we're trying to empty the canvas and failed
        if obfuscate_until_empty and len(canvas) > 0:
            raise ValueError(
                f"Failed to fully obfuscate sequence in {self.max_edit_steps} steps. "
                f"Remaining length: {len(canvas)}"
            )

        return canvas, cursor_pos, steps

    def _sample_action(self, canvas: List[int], cursor_pos: int, *, prev_action: Optional[ActionType] = None) -> ActionType:
        """Sample an action type based on probabilities (optionally conditioned on prev_action)."""
        delete_prob = float(self.delete_prob)
        move_prob = float(self.move_prob)
        insert_prob = float(self.insert_prob)

        if (
            prev_action == ActionType.INSERT
            and self.after_insert_delete_prob is not None
            and self.after_insert_move_prob is not None
            and self.after_insert_insert_prob is not None
        ):
            delete_prob = float(self.after_insert_delete_prob)
            move_prob = float(self.after_insert_move_prob)
            insert_prob = float(self.after_insert_insert_prob)

        r = random.random()
        if r < delete_prob:
            return ActionType.DELETE
        elif r < delete_prob + move_prob:
            return ActionType.MOVE
        else:
            return ActionType.INSERT

    def _sample_valid_action(self, canvas: List[int], cursor_pos: int, *, prev_action: Optional[ActionType] = None) -> ActionType:
        """
        Sample an action type, masking out invalid operations.

        If DELETE cannot be performed (cursor_pos == 0), mask it out
        and resample from the remaining valid actions with renormalized probabilities.
        """
        # Check which actions are valid
        can_delete = cursor_pos > 0

        if can_delete:
            # All actions valid, use normal probabilities
            return self._sample_action(canvas, cursor_pos, prev_action=prev_action)
        else:
            # DELETE is masked out, renormalize probabilities for MOVE and INSERT
            # Use conditioned distribution if applicable, then renormalize.
            delete_prob = float(self.delete_prob)
            move_prob = float(self.move_prob)
            insert_prob = float(self.insert_prob)
            if (
                prev_action == ActionType.INSERT
                and self.after_insert_delete_prob is not None
                and self.after_insert_move_prob is not None
                and self.after_insert_insert_prob is not None
            ):
                delete_prob = float(self.after_insert_delete_prob)
                move_prob = float(self.after_insert_move_prob)
                insert_prob = float(self.after_insert_insert_prob)

            total_prob = move_prob + insert_prob
            r = random.random()
            if r < move_prob / total_prob:
                return ActionType.MOVE
            else:
                return ActionType.INSERT

    def _sample_move(self, canvas: List[int], cursor_pos: int) -> int:
        """
        Sample a valid move amount.

        The cursor can be at positions 0 to len(canvas) inclusive.
        """
        max_right = len(canvas) - cursor_pos
        max_left = cursor_pos

        # Collect valid move amounts
        valid_moves = []
        for amount in MOVE_AMOUNTS:
            if amount <= max_right:
                valid_moves.append(amount)
            if amount <= max_left:
                valid_moves.append(-amount)

        if not valid_moves:
            return 0

        return random.choice(valid_moves)

    def create_restoration_trajectory(
        self,
        obfuscation_steps: List[ObfuscationStep],
        final_sequence_length: int,
        add_final_random_move: bool = False,
    ) -> List[int]:
        """
        Create restoration trajectory by reversing obfuscation steps.

        For each obfuscation step:
        - INSERT -> DELETE
        - DELETE -> INSERT (the deleted token)
        - MOVE +n -> MOVE -n

        Args:
            obfuscation_steps: Steps from obfuscation process
            final_sequence_length: Length of the final restored sequence (only used if add_final_random_move=True)
            add_final_random_move: If True, append an extra MOVE at the end to place the cursor at a random
                position in the final sequence. Default False because the reverse+invert construction already
                restores the original cursor position (especially when random_cursor_start=True).

        Returns:
            List of token IDs representing the restoration trajectory
        """
        restoration_tokens = []

        # Reverse the steps
        for step in reversed(obfuscation_steps):
            if step.action == ActionType.INSERT:
                # Obfuscation inserted -> restoration deletes
                restoration_tokens.append(SPECIAL_TOKENS.delete)

            elif step.action == ActionType.DELETE:
                # Obfuscation deleted -> restoration inserts the token
                restoration_tokens.append(step.value)

            elif step.action == ActionType.MOVE:
                # Obfuscation moved +n -> restoration moves -n
                # The move amount is already valid (from MOVE_AMOUNTS), so just reverse it
                restoration_tokens.append(get_move_token_id(-step.value))

        # Optional: add a random MOVE at the end to randomize cursor position in the final sequence.
        # NOTE: This is OFF by default because it changes the action-mix distribution and is not part of the
        # core reverse+invert construction described in the notes/spec.
        if bool(add_final_random_move) and int(final_sequence_length) > 0:
            target_cursor_pos = random.randint(0, int(final_sequence_length))
            if target_cursor_pos > 0:
                move_amount = self._find_closest_move_amount(int(target_cursor_pos))
                if move_amount != 0:
                    restoration_tokens.append(get_move_token_id(int(move_amount)))

        # Append end of response token
        restoration_tokens.append(SPECIAL_TOKENS.end_of_response)

        return restoration_tokens

    def _find_closest_move_amount(self, target: int) -> int:
        """Find the closest valid move amount to target."""
        if target == 0:
            return 0
        # MOVE_AMOUNTS is sorted, find closest
        from .vocabulary import MOVE_AMOUNTS
        closest = min(MOVE_AMOUNTS, key=lambda x: abs(x - target))
        return closest

    def generate_training_sample(
        self,
        prompt_tokens: List[int],
        response_tokens: List[int],
        random_cursor_start: bool = False,
        add_final_random_move: bool = False,
    ) -> TrainingSample:
        """
        Generate a complete training sample from prompt and response.

        Args:
            prompt_tokens: Tokenized prompt (input to the model)
            response_tokens: Tokenized response (what we train to reconstruct)
            random_cursor_start: If True, start cursor at random position (default False)
                                Note: Spec says to always start at position 0

        Returns:
            TrainingSample with input_ids, labels, and canvas states
        """
        # Choose initial cursor position
        # Per spec: "make the initial sequence always start with the cursor position zero"
        if random_cursor_start and len(response_tokens) > 0:
            # Cursor can be at any position from 0 to len(response_tokens) inclusive
            initial_cursor_pos = random.randint(0, len(response_tokens))
        else:
            initial_cursor_pos = 0

        # Obfuscate the response (obfuscates until empty by default)
        obfuscated_canvas, final_cursor_pos, obfuscation_steps = self.obfuscate(
            tokens=response_tokens,
            initial_cursor_pos=initial_cursor_pos,
        )

        # Create restoration trajectory
        restoration_trajectory = self.create_restoration_trajectory(
            obfuscation_steps,
            final_sequence_length=len(response_tokens),
            add_final_random_move=bool(add_final_random_move),
        )

        # Build input sequence: prompt + END_OF_INPUT + restoration trajectory
        input_ids = (
            list(prompt_tokens) +
            [SPECIAL_TOKENS.end_of_input] +
            restoration_trajectory
        )

        # Create labels (mask prompt with -100)
        prompt_length = len(prompt_tokens) + 1  # +1 for END_OF_INPUT
        labels = [-100] * prompt_length + restoration_trajectory

        # Generate canvas states for cross-attention
        canvas_states = self._generate_canvas_states(
            response_tokens,
            obfuscation_steps,
            restoration_trajectory,
            prompt_length,
        )

        return TrainingSample(
            input_ids=input_ids,
            labels=labels,
            canvas_states=canvas_states,
            prompt_length=prompt_length,
        )

    def _generate_canvas_states(
        self,
        original_tokens: List[int],
        obfuscation_steps: List[ObfuscationStep],
        restoration_trajectory: List[int],
        prompt_length: int,
    ) -> List[List[int]]:
        """
        Generate the true canvas state at each timestep.

        canvas_states[t] represents C_t, the canvas state used to predict action t.
        For prompt tokens (t < prompt_length), canvas is empty.
        For restoration tokens, we simulate the restoration process.

        IMPORTANT: Canvas states include the CURSOR token at the current cursor position,
        so the model knows where the cursor is at each timestep.

        Args:
            original_tokens: The original response tokens
            obfuscation_steps: Steps from obfuscation
            restoration_trajectory: The restoration token sequence
            prompt_length: Length of prompt including END_OF_INPUT

        Returns:
            List of canvas states (with CURSOR token inserted), one per input token
        """
        # First, compute the final obfuscated state (which is empty)
        # Then replay restoration to get intermediate states

        # For prompt tokens, canvas is empty (they don't need canvas info)
        canvas_states = [[] for _ in range(prompt_length)]

        # Simulate restoration from the obfuscated (empty) state
        canvas: List[int] = []
        cursor_pos = 0

        # We need to replay based on restoration trajectory
        # But we need to track which restoration token we're at
        restoration_idx = 0

        while restoration_idx < len(restoration_trajectory):
            token = restoration_trajectory[restoration_idx]

            # Record canvas state BEFORE this action (with CURSOR token inserted)
            canvas_with_cursor = list(canvas)
            canvas_with_cursor.insert(cursor_pos, SPECIAL_TOKENS.cursor)
            canvas_states.append(canvas_with_cursor)

            if token == SPECIAL_TOKENS.delete:
                # Delete token to the left of cursor
                if cursor_pos > 0:
                    canvas.pop(cursor_pos - 1)
                    cursor_pos -= 1
                restoration_idx += 1

            elif token == SPECIAL_TOKENS.end_of_response:
                # End of restoration
                restoration_idx += 1

            elif self._is_move_token(token):
                # Move cursor
                move_amount = self._get_move_amount(token)
                cursor_pos += move_amount
                cursor_pos = max(0, min(cursor_pos, len(canvas)))
                restoration_idx += 1

            else:
                # Insert token at cursor position
                canvas.insert(cursor_pos, token)
                cursor_pos += 1
                restoration_idx += 1

        return canvas_states

    def _is_move_token(self, token_id: int) -> bool:
        """Check if token is a move token."""
        from .vocabulary import is_move_token
        return is_move_token(token_id)

    def _get_move_amount(self, token_id: int) -> int:
        """Get move amount from move token."""
        from .vocabulary import get_move_amount
        return get_move_amount(token_id)


def try_generate_sample(
    obfuscator: Obfuscator,
    prompt_tokens: List[int],
    response_tokens: List[int],
    max_attempts: int = 3,
    random_cursor_start: bool = False,
    add_final_random_move: bool = False,
) -> Optional[TrainingSample]:
    """
    Try to generate a training sample, with retries on failure.

    Some samples may fail if they can't be fully obfuscated within max_edit_steps.
    This function retries a few times before giving up.

    Args:
        obfuscator: The Obfuscator instance
        prompt_tokens: Tokenized prompt
        response_tokens: Tokenized response
        max_attempts: Number of attempts before giving up
        random_cursor_start: If True, start cursor at a random index in the response
            before obfuscating until empty. Default False for backward compatibility.

    Returns:
        TrainingSample if successful, None if all attempts fail
    """
    for _ in range(max_attempts):
        try:
            return obfuscator.generate_training_sample(
                prompt_tokens,
                response_tokens,
                random_cursor_start=random_cursor_start,
                add_final_random_move=bool(add_final_random_move),
            )
        except ValueError:
            continue
    return None
