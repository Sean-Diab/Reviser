#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts._common import (
    base_parser,
    ensure_src_on_path,
    iter_records,
    load_json_or_jsonl,
    load_yaml,
    pick_prompt_text,
    pick_response_text,
)

ensure_src_on_path()

from reviser.data import SPECIAL_TOKENS, get_move_amount, is_move_token  # noqa: E402
from reviser.data.tokenizer import CursorTokenizer  # noqa: E402


def _clean_prompt_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"^\s*Rank\s*#.*?\|\s*prompt\+response\s*\n\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Prompt:\s*\n?", "", text, flags=re.IGNORECASE)
    return text.strip("\n")


def _clean_visible_text(text: str) -> str:
    if not text:
        return ""

    m1 = chr(0x00E2) + chr(0x20AC) + chr(0x2122)
    m2 = chr(0x00E2) + chr(0x20AC) + chr(0x02DC)
    m3 = chr(0x00E2) + chr(0x20AC) + chr(0x0153)
    m4 = chr(0x00E2) + chr(0x20AC) + chr(0xFFFD)
    m5 = chr(0x00E2) + chr(0x20AC) + chr(0x201C)
    m6 = chr(0x20AC) + chr(0x201C)

    d1 = chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x201A) + chr(0x00AC) + chr(0x00E2) + chr(0x201E) + chr(0x00A2)
    d2 = chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x201A) + chr(0x00AC) + chr(0x02DC)
    d3 = chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x201A) + chr(0x00AC) + chr(0x0153)
    d4 = chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x201A) + chr(0x00AC) + chr(0xFFFD)
    d5 = chr(0x00C3) + chr(0x00A2) + chr(0x00E2) + chr(0x201A) + chr(0x00AC) + chr(0x201C)
    d6 = chr(0x00C3) + chr(0x00A2) + chr(0x201A) + chr(0x00AC) + chr(0x0153)

    return (
        text.replace(m1, "'")
        .replace(m2, "'")
        .replace(m3, '"')
        .replace(m4, '"')
        .replace(m5, "--")
        .replace(m6, "--")
        .replace(d1, "'")
        .replace(d2, "'")
        .replace(d3, '"')
        .replace(d4, '"')
        .replace(d5, "--")
        .replace(d6, "--")
    )


def _move_in_bounds(cursor: int, amount: int, canvas_len: int) -> int:
    nxt = cursor + int(amount)
    if nxt < 0:
        return 0
    if nxt > canvas_len:
        return canvas_len
    return nxt


def _decode_token(tok: CursorTokenizer, tid: int) -> str:
    try:
        return _clean_visible_text(tok.decode([int(tid)]))
    except Exception:
        return f"<{int(tid)}>"


def _as_int_list(x: Any) -> list[int]:
    if not isinstance(x, list):
        return []
    out: list[int] = []
    for v in x:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _state_from_canvas(tok: CursorTokenizer, canvas: list[int], cursor: int, action: str, detail: str) -> dict[str, Any]:
    token_strings = [_decode_token(tok, t) for t in canvas]
    return {
        "tokens": list(canvas),
        "token_strings": token_strings,
        "cursor": int(cursor),
        "action": action,
        "action_detail": _clean_visible_text(detail),
        "text": "".join(token_strings),
    }


def _reconstruct_states_from_actions(tok: CursorTokenizer, actions: list[int]) -> list[dict[str, Any]]:
    canvas: list[int] = []
    cursor = 0
    states = [_state_from_canvas(tok, canvas, cursor, "START", "Initial empty canvas")]
    for a in actions:
        a = int(a)
        if a == int(SPECIAL_TOKENS.end_of_response):
            states.append(_state_from_canvas(tok, canvas, cursor, "END_OF_RESPONSE", "[END_OF_RESPONSE]"))
            break
        if a == int(SPECIAL_TOKENS.end_of_input):
            states.append(_state_from_canvas(tok, canvas, cursor, "END_OF_INPUT", "[END_OF_INPUT]"))
            continue
        if a == int(SPECIAL_TOKENS.delete):
            if cursor > 0:
                canvas.pop(cursor - 1)
                cursor -= 1
            states.append(_state_from_canvas(tok, canvas, cursor, "DELETE", "DELETE"))
            continue
        if is_move_token(a):
            amt = int(get_move_amount(a))
            cursor = _move_in_bounds(cursor, amt, len(canvas))
            states.append(_state_from_canvas(tok, canvas, cursor, "MOVE", f"[MOVE {amt:+d}]"))
            continue
        canvas.insert(cursor, a)
        ins = _decode_token(tok, a)
        cursor += 1
        states.append(_state_from_canvas(tok, canvas, cursor, "INSERT", f"INSERT '{ins}'"))
    return states


def _normalize_states(tok: CursorTokenizer, row: dict[str, Any]) -> list[dict[str, Any]]:
    states = row.get("states")
    if isinstance(states, list) and states:
        out: list[dict[str, Any]] = []
        for i, st in enumerate(states):
            if not isinstance(st, dict):
                continue
            token_strings = st.get("token_strings")
            tokens = _as_int_list(st.get("tokens"))
            if not token_strings and tokens:
                token_strings = [_decode_token(tok, t) for t in tokens]
            out.append(
                {
                    "tokens": tokens,
                    "token_strings": token_strings or [],
                    "cursor": int(st.get("cursor", 0)),
                    "action": st.get("action", "STEP"),
                    "action_detail": _clean_visible_text(str(st.get("action_detail", f"Step {i + 1}"))),
                    "text": _clean_visible_text(st.get("text") or "".join(token_strings or [])),
                }
            )
        if out:
            return out
    traj = row.get("trajectory")
    if isinstance(traj, list) and traj:
        out = []
        for i, st in enumerate(traj):
            if isinstance(st, dict):
                out.append(
                    {
                        "tokens": _as_int_list(st.get("tokens")),
                        "token_strings": st.get("token_strings", []),
                        "cursor": int(st.get("cursor", 0)),
                        "action": st.get("action", "STEP"),
                        "action_detail": _clean_visible_text(str(st.get("action_detail", f"Step {i + 1}"))),
                        "text": _clean_visible_text(st.get("text", "")),
                    }
                )
            else:
                text = _clean_visible_text(str(st))
                out.append(
                    {
                        "tokens": [],
                        "token_strings": [],
                        "cursor": 0,
                        "action": "STEP",
                        "action_detail": f"Step {i + 1}",
                        "text": text,
                    }
                )
        if out:
            return out
    actions = _as_int_list(row.get("action_history") or row.get("actions") or [])
    if actions:
        return _reconstruct_states_from_actions(tok, actions)
    final_text = _clean_visible_text(row.get("final_canvas") or pick_response_text(row) or "")
    return [
        {
            "tokens": [],
            "token_strings": [],
            "cursor": 0,
            "action": "START",
            "action_detail": "Initial empty canvas",
            "text": "",
        },
        {
            "tokens": [],
            "token_strings": [],
            "cursor": 0,
            "action": "FINAL",
            "action_detail": "Final response",
            "text": final_text,
        },
    ]


def _classic_html(results: list[dict[str, Any]], title: str, subtitle: str, prefix_len: int) -> str:
    data_json = json.dumps(results, ensure_ascii=False)
    title_esc = title.replace("<", "&lt;").replace(">", "&gt;")
    subtitle_esc = subtitle.replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>{title_esc}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        * {{ scrollbar-color: #454a4d #202324; }}
        *::-webkit-scrollbar-thumb {{ background: #454a4d; border-radius: 8px; border: 2px solid #202324; }}
        *::-webkit-scrollbar-track {{ background: #202324; }}
        :root {{
            --card-bg: #181a1b;
            --card-bg-2: #1b1e1f;
            --text: #f0e6d2;
            --muted: #c6b79e;
            --line: #334155;
            --accent: #1d4ed8;
        }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #132889 0%, #5e3c82 100%); padding: 20px; min-height: 100vh; color: var(--text); }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #181a1b; padding: 30px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.28); margin-bottom: 30px; text-align: center; border: 1px solid var(--line); }}
        .header h1 {{ color: var(--text); font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ color: var(--muted); font-size: 1.1em; }}
        .example-card {{ background: #181a1b; padding: 25px; border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.28); margin-bottom: 25px; border: 1px solid var(--line); }}
        .example-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #273244; }}
        .example-title {{ font-size: 1.3em; color: var(--text); font-weight: 600; }}
        .step-info {{ background: var(--accent); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9em; font-weight: 500; }}
        .original-text {{ background: #1b1e1f; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid var(--accent); }}
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
        .progress-bar {{ width: 100%; height: 8px; background: #273244; border-radius: 4px; overflow: hidden; margin-bottom: 20px; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #1d4ed8 0%, #5b3f94 100%); transition: width 0.3s ease; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
        .stat-box {{ background: var(--card-bg-2); padding: 15px; border-radius: 8px; text-align: center; border: 1px solid var(--line); }}
        .stat-label {{ font-size: 0.85em; color: var(--muted); margin-bottom: 5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
        .stat-value {{ font-size: 1.5em; color: #60a5fa; font-weight: 700; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>{title_esc}</h1>
            <p>{subtitle_esc}</p>
        </div>
        <div id=\"content\"></div>
    </div>
    <script>
        const results = {data_json};
        const PREFIX_LEN = {int(prefix_len)};
        let currentSteps = new Array(results.length).fill(0);
        window.onload = function() {{ renderExamples(); }};
        function getPrefixTarget(result) {{
            const pl = (typeof result.prefix_len === 'number') ? result.prefix_len : PREFIX_LEN;
            const totalSteps = (result.states || []).length;
            return Math.min(Math.max(0, pl), Math.max(0, totalSteps - 1));
        }}
        function renderExamples() {{
            const content = document.getElementById('content');
            content.innerHTML = '';
            results.forEach((result, index) => {{
                const card = createExampleCard(result, index);
                content.appendChild(card);
            }});
        }}
        function createExampleCard(result, index) {{
            const card = document.createElement('div');
            card.className = 'example-card';
            const states = result.states || [];
            const currentStep = currentSteps[index];
            const state = states[currentStep] || {{ token_strings: [], cursor: 0, action_detail: 'N/A' }};
            const totalSteps = states.length || 1;
            const progress = (currentStep / Math.max(1, (totalSteps - 1))) * 100;
            const prefixTarget = getPrefixTarget(result);
            card.innerHTML = `
                <div class=\"example-header\">
                    <div class=\"example-title\">Example ${{index + 1}}</div>
                    <div class=\"step-info\">Step ${{currentStep + 1}} / ${{totalSteps}}</div>
                </div>
                <div class=\"original-text\">
                    <div class=\"original-text-label\">Prompt</div>
                    <div class=\"original-text-content\">${{escapeHtml(result.prompt_text || '')}}</div>
                </div>
                <div class=\"original-text\">
                    <div class=\"original-text-label\">Final Response</div>
                    <div class=\"original-text-content\">${{escapeHtml(result.original_text || '')}}</div>
                </div>
                <div class=\"progress-bar\"><div class=\"progress-fill\" style=\"width: ${{progress}}%\"></div></div>
                <div class=\"action-display\" id=\"action-${{index}}\">${{escapeHtml(state.action_detail || 'N/A')}}</div>
                <div class=\"text-display\"><div class=\"text-display-content\" id=\"text-${{index}}\">${{renderTextWithCursor(state.cursor || 0, state)}}</div></div>
                <div class=\"controls\">
                    <button class=\"btn-prev\" onclick=\"stepBackward(${{index}})\" ${{currentStep === 0 ? 'disabled' : ''}}>Previous</button>
                    <button class=\"btn-reset\" onclick=\"resetStep(${{index}})\">Reset</button>
                    <button class=\"btn-prefix\" onclick=\"jumpToPrefixEnd(${{index}})\" ${{(prefixTarget === 0) ? 'disabled' : ''}}>Prefix End</button>
                    <button class=\"btn-next\" onclick=\"stepForward(${{index}})\" ${{currentStep === totalSteps - 1 ? 'disabled' : ''}}>Next</button>
                </div>
                <div class=\"stats\">
                    <div class=\"stat-box\"><div class=\"stat-label\">Total Steps</div><div class=\"stat-value\">${{totalSteps}}</div></div>
                    <div class=\"stat-box\"><div class=\"stat-label\">Current Tokens</div><div class=\"stat-value\">${{(state.token_strings || []).length}}</div></div>
                    <div class=\"stat-box\"><div class=\"stat-label\">Cursor Position</div><div class=\"stat-value\">${{state.cursor || 0}}</div></div>
                </div>
            `;
            return card;
        }}
        function renderTextWithCursor(cursorPos, state) {{
            let out = '';
            const tokenStrings = state.token_strings || [];
            if (tokenStrings.length === 0) {{
                return '<span class=\"cursor\"></span>';
            }}
            for (let i = 0; i <= tokenStrings.length; i++) {{
                if (i === cursorPos) {{
                    out += '<span class=\"cursor\"></span>';
                }}
                if (i < tokenStrings.length) {{
                    out += escapeHtml(tokenStrings[i]);
                }}
            }}
            return out;
        }}
        function stepForward(index) {{ if (currentSteps[index] < (results[index].states || []).length - 1) {{ currentSteps[index]++; updateExample(index); }} }}
        function stepBackward(index) {{ if (currentSteps[index] > 0) {{ currentSteps[index]--; updateExample(index); }} }}
        function resetStep(index) {{ currentSteps[index] = 0; updateExample(index); }}
        function jumpToPrefixEnd(index) {{
            const result = results[index];
            currentSteps[index] = getPrefixTarget(result);
            updateExample(index);
        }}
        function updateExample(index) {{
            const result = results[index];
            const state = (result.states || [])[currentSteps[index]] || {{ token_strings: [], cursor: 0, action_detail: 'N/A' }};
            const textNode = document.getElementById(`text-${{index}}`);
            const actionNode = document.getElementById(`action-${{index}}`);
            if (textNode) textNode.innerHTML = renderTextWithCursor(state.cursor || 0, state);
            if (actionNode) actionNode.textContent = state.action_detail || 'N/A';
            const cards = document.getElementsByClassName('example-card');
            cards[index].replaceWith(createExampleCard(result, index));
        }}
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = (text === null || text === undefined) ? '' : String(text);
            return div.innerHTML;
        }}
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'ArrowRight') {{ results.forEach((_, index) => stepForward(index)); }}
            else if (e.key === 'ArrowLeft') {{ results.forEach((_, index) => stepBackward(index)); }}
            else if (e.key === 'r' || e.key === 'R') {{ results.forEach((_, index) => resetStep(index)); }}
            else if (e.key === 'p' || e.key === 'P') {{ results.forEach((_, index) => jumpToPrefixEnd(index)); }}
        }});
    </script>
</body>
</html>
"""


def main() -> None:
    ap = base_parser("Build a trajectory HTML viewer from rollout artifacts.")
    ap.add_argument("--title", default="Reviser Inference Examples")
    ap.add_argument("--subtitle", default="Selected examples with step-by-step canvas trajectories")
    ap.add_argument("--prefix-len", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    payload = load_json_or_jsonl(in_path)
    records = list(iter_records(payload))
    cfg = load_yaml(args.config) if args.config else {}
    prefix_len = args.prefix_len
    if prefix_len is None:
        prefix_len = int(cfg.get("data", {}).get("prefix_length", cfg.get("data", {}).get("prefix_len", 35)))

    tok = CursorTokenizer()
    rows = []
    for i, r in enumerate(records):
        prompt_text = _clean_visible_text(_clean_prompt_text(r.get("prompt_text") or pick_prompt_text(r)))
        response_text = _clean_visible_text(r.get("original_text") or pick_response_text(r))
        states = _normalize_states(tok, r)
        if response_text.strip() == "" and states:
            response_text = _clean_visible_text(states[-1].get("text", ""))

        rows.append(
            {
                "id": r.get("id", i),
                "prompt_index": i + 1,
                "prompt_text": prompt_text,
                "original_text": response_text,
                "states": states,
                "prefix_len": int(r.get("prefix_len", prefix_len)),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_classic_html(rows, args.title, args.subtitle, int(prefix_len)), encoding="utf-8")


if __name__ == "__main__":
    main()
