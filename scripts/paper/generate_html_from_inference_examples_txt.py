#!/usr/bin/env python3
"""
Generate interactive HTML visualizations from `inference_examples.txt`.

This reuses the same results-schema as the existing notebook visualization HTML:
- results: List[{prompt_index, original_text, original_tokens, obfuscated_text, states: [...] }]
- each state: {tokens, token_strings, cursor, action, action_detail, text}

We do NOT require the old notebook dependencies (model, tokenizer, etc).
We only parse decoded edit-history actions and replay them on an initially-empty canvas.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MOVE_RE = re.compile(r"\[MOVE\s*([+-]?\d+)\]")
MOVE_UNDERSCORE_RE = re.compile(r"\[MOVE_([+-]?\d+)\]")


@dataclass
class ParsedExample:
    prompt: str
    response_text: str
    actions: list[str]  # decoded action tokens like "' hello'" or "[MOVE -4]" or "[DELETE]"


def _unquote_token_string(s: str) -> str:
    """
    Convert lines like "' hello'" or "'\\n'" into the real token string.
    Falls back to raw text if it isn't a valid python literal.
    """
    s = s.strip()
    if not s:
        return ""
    # Most files store inserts as repr(...) (single-quoted python string)
    if (len(s) >= 2) and ((s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')):
        try:
            return ast.literal_eval(s)
        except Exception:
            return s[1:-1]
    return s


def _parse_move_amount(action: str) -> int | None:
    m = MOVE_RE.search(action)
    if m:
        return int(m.group(1))
    m = MOVE_UNDERSCORE_RE.search(action)
    if m:
        return int(m.group(1))
    return None


def parse_inference_examples_txt(text: str) -> list[ParsedExample]:
    """
    Parse `inference_examples.txt` which may contain multiple formats.
    We extract only examples that include an edit-history action list.
    """
    lines = text.splitlines()
    out: list[ParsedExample] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Format C (PROMPT: / RESPONSE_TEXT: / ACTIONS ...)
        if line == "PROMPT:":
            i += 1
            prompt_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "RESPONSE_TEXT:":
                prompt_lines.append(lines[i].rstrip("\n"))
                i += 1
            prompt = "\n".join([p for p in prompt_lines]).strip()
            if i < len(lines) and lines[i].strip() == "RESPONSE_TEXT:":
                i += 1
            resp_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("STEPS:"):
                resp_lines.append(lines[i].rstrip("\n"))
                i += 1
            response_text = "\n".join(resp_lines).rstrip()

            # find actions table
            while i < len(lines) and "ACTIONS (token_id -> token_string)" not in lines[i]:
                i += 1
            actions: list[str] = []
            if i < len(lines) and "ACTIONS (token_id -> token_string)" in lines[i]:
                i += 1
                while i < len(lines):
                    row = lines[i].strip()
                    if not row:
                        i += 1
                        break
                    # rows look like: "60 -> ]" or "50262 -> [MOVE_-1]"
                    if "->" in row:
                        _, tok = row.split("->", 1)
                        tok = tok.strip()
                        # normalize move token formatting
                        tok = tok.replace("[MOVE_-", "[MOVE -").replace("[MOVE_+", "[MOVE +")
                        actions.append(tok)
                    i += 1

            if prompt and actions:
                out.append(ParsedExample(prompt=prompt, response_text=response_text, actions=actions))
            continue

        # Format A/B ("prompt:" blocks)
        if line.startswith("prompt:"):
            # prompt line may be: prompt: '...'
            prompt = line[len("prompt:") :].strip()
            prompt = _unquote_token_string(prompt)

            # scan forward for response_text
            response_text = ""
            actions: list[str] = []

            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("response_text:") and lines[j].strip() != "RESPONSE_TEXT:":
                # Some prompts are written over multiple lines:
                # prompt:
                # Write a short apology...
                if lines[i].strip() == "prompt:":
                    # collect until blank line
                    prompt_lines = []
                    k = i + 1
                    while k < len(lines) and lines[k].strip() and not lines[k].strip().startswith("RESPONSE_TEXT:"):
                        prompt_lines.append(lines[k].rstrip("\n"))
                        k += 1
                    prompt = "\n".join(prompt_lines).strip()
                    j = k
                    break
                j += 1

            # response_text in either "response_text:" or "RESPONSE_TEXT:"
            if j < len(lines) and lines[j].strip().startswith("response_text:"):
                response_text = _unquote_token_string(lines[j].strip()[len("response_text:") :].strip())
                j += 1
            elif j < len(lines) and lines[j].strip() == "RESPONSE_TEXT:":
                j += 1
                resp_lines = []
                while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith("STEPS:") and not lines[j].strip().startswith("steps:"):
                    resp_lines.append(lines[j].rstrip("\n"))
                    j += 1
                response_text = "\n".join(resp_lines).rstrip()

            # After response_text, many reports include blank lines and other sections before actions.
            # Skip blank lines first.
            while j < len(lines) and not lines[j].strip():
                j += 1

            # Find actions list header: prefer explicit "edit_history_tokens_decoded:".
            # Some reports include "edit_history_token_ids:" and other sections before it, so scan forward.
            scan_limit = min(len(lines), j + 500)
            found_header = False
            while j < scan_limit:
                s = lines[j].strip()
                if s.startswith("edit_history_tokens_decoded:"):
                    found_header = True
                    break
                # Stop scanning if we hit the start of another example.
                if s.startswith("prompt:") or s == "PROMPT:" or s.startswith("RUN "):
                    break
                j += 1

            # If we landed on "steps:" then continue scanning for actions header or start of actions
            if j < len(lines) and lines[j].strip().startswith("steps:"):
                j += 1

            if j < len(lines) and lines[j].strip().startswith("edit_history_tokens_decoded:"):
                j += 1

            # collect action lines until blank line or next prompt-like marker
            k = j
            while k < len(lines):
                s = lines[k].strip()
                if not s:
                    break
                if s.startswith("prompt:") or s == "PROMPT:" or s.startswith("Prompts & Responses"):
                    break
                if s.startswith("RUN "):
                    break
                if s.startswith("steps:") or s.startswith("response_text:") or s == "RESPONSE_TEXT:":
                    # unexpected marker; stop to avoid merging blocks
                    break
                actions.append(s)
                k += 1

            # Heuristic: only accept blocks that actually look like action lists
            if prompt and response_text and any(("MOVE" in a or "DELETE" in a or "END_OF_RESPONSE" in a or a.startswith("'") or a.startswith('"')) for a in actions):
                out.append(ParsedExample(prompt=prompt, response_text=response_text, actions=actions))

            i = max(i + 1, k)
            continue

        i += 1

    return out


def replay_actions_to_states(example: ParsedExample) -> list[dict[str, Any]]:
    """
    Replays decoded actions on a token-string canvas.
    """
    tokens: list[str] = []
    cursor = 0

    def snapshot(action: str, action_detail: str) -> dict[str, Any]:
        return {
            "tokens": tokens.copy(),
            "token_strings": tokens.copy(),
            "cursor": cursor,
            "action": action,
            "action_detail": action_detail,
            "text": "".join(tokens),
        }

    states: list[dict[str, Any]] = [snapshot("START", "Initial empty canvas")]

    for raw in example.actions:
        a = raw.strip()
        if a in ("[END_OF_RESPONSE]", "[STOP]"):
            states.append(snapshot("STOP", "STOP - Generation complete"))
            break

        if a == "[DELETE]":
            if cursor > 0 and tokens:
                deleted = tokens[cursor - 1]
                tokens.pop(cursor - 1)
                cursor -= 1
                states.append(snapshot("DELETE", f"DELETE {deleted!r}"))
            else:
                states.append(snapshot("DELETE", "DELETE (invalid - cursor at start)"))
            continue

        mv = _parse_move_amount(a)
        if mv is not None:
            old = cursor
            cursor = max(0, min(len(tokens), cursor + mv))
            states.append(snapshot(f"[MOVE{mv:+d}]", f"MOVE {mv:+d} (position {old} -> {cursor})"))
            continue

        # Otherwise treat as INSERT with token string
        tok_str = _unquote_token_string(a)
        tokens.insert(cursor, tok_str)
        cursor += 1
        # escape newlines for readability in action_detail
        disp = tok_str.replace("\n", "\\n")
        states.append(snapshot("INSERT", f"INSERT {disp!r}"))

    return states


def generate_html_with_data(results: list[dict[str, Any]]) -> str:
    """
    Interactive standalone trajectory viewer with autoplay and speed control.
    """
    results_json = json.dumps(results, indent=2, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit History Visualization</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        * {{
            scrollbar-color: #454a4d #202324;
        }}
        *::-webkit-scrollbar-thumb {{
            background: #454a4d;
            border-radius: 8px;
            border: 2px solid #202324;
        }}
        *::-webkit-scrollbar-track {{
            background: #202324;
        }}
        :root {{
            --page-bg: #132889;
            --card-bg: #181a1b;
            --card-bg-2: #1b1e1f;
            --text: #f0e6d2;
            --muted: #c6b79e;
            --line: #334155;
            --accent: #1d4ed8;
        }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #132889 0%, #5e3c82 100%); padding: 20px; min-height: 100vh; color: var(--text); }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: var(--card-bg); padding: 30px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.28); margin-bottom: 30px; text-align: center; border: 1px solid var(--line); }}
        .header h1 {{ color: var(--text); font-size: 2.5em; margin-bottom: 10px; }}
        .example-card {{ background: var(--card-bg); padding: 25px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.28); margin-bottom: 25px; border: 1px solid var(--line); }}
        .example-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #273244; }}
        .example-title {{ font-size: 1.3em; color: var(--text); font-weight: 600; }}
        .step-info {{ background: var(--accent); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; font-weight: 500; }}
        .original-text {{ background: var(--card-bg-2); padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid var(--accent); }}
        .original-text-label {{ font-size: 0.85em; color: var(--muted); font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .original-text-content {{ color: var(--text); font-size: 1.1em; line-height: 1.6; font-family: 'Courier New', monospace; white-space: pre-wrap; }}
        .text-display {{ background: var(--card-bg-2); padding: 20px; border-radius: 8px; min-height: 100px; margin-bottom: 20px; border: 2px solid var(--line); }}
        .text-display-content {{ font-size: 1.2em; line-height: 1.8; color: var(--text); font-family: 'Courier New', monospace; word-wrap: break-word; white-space: pre-wrap; }}
        .cursor {{ display: inline-block; width: 3px; height: 1.4em; background: #ff4757; animation: blink 1s infinite; margin: 0 2px; vertical-align: middle; }}
        @keyframes blink {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0; }} }}
        .action-display {{ background: linear-gradient(135deg, #1d4ed8 0%, #5b3f94 100%); color: white; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; font-size: 1em; font-weight: 500; min-height: 52px; display: flex; align-items: center; justify-content: center; }}
        .controls {{ display: flex; gap: 12px; justify-content: center; align-items: center; flex-wrap: wrap; }}
        button {{ background: linear-gradient(135deg, #1d4ed8 0%, #5b3f94 100%); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1em; cursor: pointer; transition: all 0.3s ease; font-weight: 500; }}
        button:hover:not(:disabled) {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4); }}
        button:disabled {{ opacity: 0.5; cursor: not-allowed; }}
        .btn-prev {{ background: linear-gradient(135deg, #5b0d5b 0%, #8b1a52 100%); }}
        .btn-next {{ background: linear-gradient(135deg, #0b75c9 0%, #13b8c9 100%); }}
        .btn-reset {{ background: linear-gradient(135deg, #0ca678 0%, #14b8a6 100%); }}
        .btn-prefix {{ background: linear-gradient(135deg, #7c2d12 0%, #a16207 100%); color: #f8fafc; }}
        .btn-play {{ background: linear-gradient(135deg, #374151 0%, #111827 100%); }}
        .speed-wrap {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.9em; }}
        .speed-slider {{ width: 140px; }}
        .progress-bar {{ width: 100%; height: 8px; background: #273244; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #1d4ed8 0%, #5b3f94 100%); transition: width 0.3s ease; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
        .stat-box {{ background: var(--card-bg-2); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid var(--line); }}
        .stat-label {{ font-size: 0.85em; color: var(--muted); margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 1.5em; color: #60a5fa; font-weight: 700; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Edit History Visualization</h1>
        </div>
        <div id="content"></div>
    </div>
    <script>
        const results = {results_json};
        const PREFIX_LEN = 35;
        let currentSteps = new Array(results.length).fill(0);
        let playTimers = new Array(results.length).fill(null);
        let playSpeeds = new Array(results.length).fill(0.3);
        window.onload = function() {{ renderExamples(); }};

        function getPrefixTarget(result) {{
            const pl = (typeof result.prefix_len === 'number') ? result.prefix_len : PREFIX_LEN;
            const totalSteps = result.states.length;
            return Math.min(Math.max(0, pl), Math.max(0, totalSteps - 1));
        }}

        function renderExamples() {{
            const content = document.getElementById('content');
            content.innerHTML = '';
            results.forEach((result, index) => {{
                content.appendChild(createExampleCard(result, index));
            }});
        }}

        function createExampleCard(result, index) {{
            const card = document.createElement('div');
            card.className = 'example-card';
            const currentStep = currentSteps[index];
            const state = result.states[currentStep];
            const totalSteps = result.states.length;
            const progress = (currentStep / Math.max(1, (totalSteps - 1))) * 100;
            const prefixTarget = getPrefixTarget(result);
            card.innerHTML = `
                <div class="example-header">
                    <div class="example-title">Example ${{index + 1}}</div>
                    <div class="step-info" id="step-info-${{index}}">Step ${{currentStep + 1}} / ${{totalSteps}}</div>
                </div>
                <div class="original-text">
                    <div class="original-text-label">Prompt</div>
                    <div class="original-text-content">${{escapeHtml(result.prompt_text)}}</div>
                </div>
                <div class="original-text">
                    <div class="original-text-label">Final Response</div>
                    <div class="original-text-content">${{escapeHtml(result.original_text)}}</div>
                </div>
                <div class="progress-bar"><div class="progress-fill" id="progress-${{index}}" style="width: ${{progress}}%"></div></div>
                <div class="action-display" id="action-${{index}}">${{escapeHtml(state.action_detail)}}</div>
                <div class="text-display"><div class="text-display-content" id="text-${{index}}">${{renderTextWithCursor(state.cursor, state)}}</div></div>
                <div class="controls">
                    <button id="btn-prev-${{index}}" class="btn-prev" onclick="stepBackward(${{index}})" ${{currentStep === 0 ? 'disabled' : ''}}>Previous</button>
                    <button class="btn-reset" onclick="resetStep(${{index}})">Reset</button>
                    <button class="btn-prefix" onclick="jumpToPrefixEnd(${{index}})" ${{(prefixTarget === 0) ? 'disabled' : ''}}>Prefix End</button>
                    <button id="btn-next-${{index}}" class="btn-next" onclick="stepForward(${{index}})" ${{currentStep === totalSteps - 1 ? 'disabled' : ''}}>Next</button>
                    <button id="btn-play-${{index}}" class="btn-play" onclick="togglePlay(${{index}})">${{playTimers[index] ? 'Pause' : 'Play'}}</button>
                    <div class="speed-wrap">
                        <span id="speed-label-${{index}}">Speed: ${{playSpeeds[index].toFixed(1)}}s</span>
                        <input class="speed-slider" type="range" min="0.1" max="1" step="0.1" value="${{playSpeeds[index].toFixed(1)}}" oninput="setPlaySpeed(${{index}}, this.value)"/>
                    </div>
                </div>
                <div class="stats">
                    <div class="stat-box"><div class="stat-label">Total Steps</div><div class="stat-value" id="stat-total-${{index}}">${{totalSteps}}</div></div>
                    <div class="stat-box"><div class="stat-label">Current Tokens</div><div class="stat-value" id="stat-tokens-${{index}}">${{(state.token_strings || []).length}}</div></div>
                    <div class="stat-box"><div class="stat-label">Cursor Position</div><div class="stat-value" id="stat-cursor-${{index}}">${{state.cursor}}</div></div>
                </div>
            `;
            return card;
        }}

        function renderTextWithCursor(cursorPos, state) {{
            let html = '';
            const tokenStrings = state.token_strings || [];
            if (tokenStrings.length === 0) return '<span class="cursor"></span>';
            for (let i = 0; i <= tokenStrings.length; i++) {{
                if (i === cursorPos) html += '<span class="cursor"></span>';
                if (i < tokenStrings.length) html += escapeHtml(tokenStrings[i]);
            }}
            return html;
        }}

        function stepForward(index) {{ if (currentSteps[index] < results[index].states.length - 1) {{ currentSteps[index]++; updateExample(index); }} }}
        function stepBackward(index) {{ if (currentSteps[index] > 0) {{ currentSteps[index]--; updateExample(index); }} }}
        function resetStep(index) {{ currentSteps[index] = 0; updateExample(index); }}
        function jumpToPrefixEnd(index) {{
            const result = results[index];
            currentSteps[index] = getPrefixTarget(result);
            updateExample(index);
        }}

        function setPlaySpeed(index, value) {{
            const v = Math.max(0.1, Math.min(1.0, Number(value) || 0.3));
            playSpeeds[index] = v;
            const lbl = document.getElementById(`speed-label-${{index}}`);
            if (lbl) lbl.textContent = `Speed: ${{v.toFixed(1)}}s`;
            if (playTimers[index]) startAuto(index);
        }}

        function startAuto(index) {{
            stopAuto(index, false);
            playTimers[index] = setInterval(() => {{
                const maxStep = results[index].states.length - 1;
                if (currentSteps[index] < maxStep) {{ currentSteps[index]++; updateExample(index); }}
                else {{ stopAuto(index, true); }}
            }}, Math.max(100, Math.min(1000, playSpeeds[index] * 1000)));
            const btn = document.getElementById(`btn-play-${{index}}`);
            if (btn) btn.textContent = 'Pause';
        }}

        function stopAuto(index, refresh) {{
            if (playTimers[index]) {{ clearInterval(playTimers[index]); playTimers[index] = null; }}
            const btn = document.getElementById(`btn-play-${{index}}`);
            if (btn) btn.textContent = 'Play';
            if (refresh) updateExample(index);
        }}

        function togglePlay(index) {{
            if (playTimers[index]) stopAuto(index, true);
            else startAuto(index);
        }}
        function fixMojibake(text) {{
            if (text === null || text === undefined) return '';
            let s = String(text);
            // Safe fallback: normalize replacement-character artifacts.
            s = s.replace(/��/g, "'")
                 .replace(/�/g, "'");
            return s;
        }}

        function updateExample(index) {{
            const result = results[index];
            const state = result.states[currentSteps[index]];
            const totalSteps = result.states.length;
            const progress = (currentSteps[index] / Math.max(1, (totalSteps - 1))) * 100;

            const textNode = document.getElementById(`text-${{index}}`);
            if (textNode) textNode.innerHTML = renderTextWithCursor(state.cursor, state);
            const actionNode = document.getElementById(`action-${{index}}`);
            if (actionNode) actionNode.textContent = fixMojibake(state.action_detail || '');
            const stepNode = document.getElementById(`step-info-${{index}}`);
            if (stepNode) stepNode.textContent = `Step ${{currentSteps[index] + 1}} / ${{totalSteps}}`;
            const progNode = document.getElementById(`progress-${{index}}`);
            if (progNode) progNode.style.width = `${{progress}}%`;
            const tokNode = document.getElementById(`stat-tokens-${{index}}`);
            if (tokNode) tokNode.textContent = String((state.token_strings || []).length);
            const curNode = document.getElementById(`stat-cursor-${{index}}`);
            if (curNode) curNode.textContent = String(state.cursor);

            const prevBtn = document.getElementById(`btn-prev-${{index}}`);
            if (prevBtn) prevBtn.disabled = currentSteps[index] === 0;
            const nextBtn = document.getElementById(`btn-next-${{index}}`);
            if (nextBtn) nextBtn.disabled = currentSteps[index] >= (totalSteps - 1);
            const playBtn = document.getElementById(`btn-play-${{index}}`);
            if (playBtn) playBtn.textContent = playTimers[index] ? 'Pause' : 'Play';
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = fixMojibake((text === null || text === undefined) ? '' : String(text));
            return div.innerHTML;
        }}

        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') results.forEach((_, index) => stepForward(index));
            else if (e.key === 'ArrowLeft') results.forEach((_, index) => stepBackward(index));
            else if (e.key === 'r' || e.key === 'R') results.forEach((_, index) => resetStep(index));
        }});
    </script>
</body>
</html>
"""


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", type=str, default=str(Path.cwd() / "inference_examples.txt"))
    ap.add_argument("--out_file", type=str, default=str(Path.cwd() / "visualization" / "inference_examples_visualization.html"))
    args = ap.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file)

    txt = in_path.read_text(encoding="utf-8", errors="replace")
    examples = parse_inference_examples_txt(txt)

    results: list[dict[str, Any]] = []
    for idx, ex in enumerate(examples, start=1):
        states = replay_actions_to_states(ex)
        results.append(
            {
                "prompt_index": idx,
                "prompt_text": ex.prompt,
                "original_text": ex.response_text,
                "original_tokens": states[-1]["tokens"],
                "obfuscated_text": "",
                "states": states,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_html_with_data(results), encoding="utf-8")
    print(f"Wrote HTML visualization to: {out_path}")
    print(f"Parsed examples with edit history: {len(examples)}")


if __name__ == "__main__":
    main()

