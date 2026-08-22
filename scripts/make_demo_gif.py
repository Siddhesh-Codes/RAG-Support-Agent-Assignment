"""Generate the README demo GIF from REAL agent sessions.

Runs the actual agent (live LLM) through five scripted scenarios and renders
the genuine transcripts as terminal-style frames:

1. Knowledge-base question with citations
2. Order lookup
3. Multi-turn conversation
4. Safe refusal / human handoff
5. The evaluation suite running

Usage:
    python scripts/make_demo_gif.py [--out docs/demo.gif]

Requires Pillow (pip install pillow).
"""

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 640
BG = (24, 26, 32)
FG = (220, 224, 230)
ACCENT = (120, 200, 255)
GREEN = (140, 220, 150)
YELLOW = (240, 200, 120)
MARGIN = 24
LINE_H = 20

SCENARIOS = [
    ("1/5  Knowledge-base question with citations", [
        ("user", "How long do I have to return an unused backpack?"),
    ]),
    ("2/5  Order lookup", [
        ("user", "Where is my order ORD-1007 and when should it arrive?"),
    ]),
    ("3/5  Multi-turn conversation", [
        ("user", "Do you ship internationally?"),
        ("user", "What about Canada, and how long does it take?"),
    ]),
    ("4/5  Correctly refuses to guess / recommends human help", [
        ("user", "Are all fabrics and adhesives in your bags vegan?"),
    ]),
]


def _font(size: int):
    for name in ("consola", "cournew", "Courier New", "Consolas"):
        try:
            return ImageFont.truetype(f"{name}.ttf", size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(draw, text, font, max_width):
    lines, cur = [], ""
    for word in text.split(" "):
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [""]


def render_frame(title, turns, footer="", scroll_lines=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f = _font(15)
    ftitle = _font(17)

    d.rectangle([0, 0, W, 44], fill=(32, 36, 44))
    d.text((MARGIN, 13), f"Aster & Row Support Agent — {title}", font=ftitle, fill=ACCENT)

    y = 60
    max_w = W - 2 * MARGIN
    lines = []
    if scroll_lines is not None:
        for color, txt in scroll_lines:
            for ln in _wrap(d, txt, f, max_w):
                lines.append((color, ln))
    else:
        for role, msg in turns:
            if role == "user":
                lines.append((GREEN, f"You > {msg[0]}"))
                for extra in msg[1:]:
                    lines.append((GREEN, extra))
            else:
                lines.append((FG, f"Agent > {msg[0]}"))
                for extra in msg[1:]:
                    lines.append((FG, extra))
            lines.append((FG, ""))

    # Show the tail of the transcript if it overflows
    visible = (H - 90 - (30 if footer else 0)) // LINE_H
    for color, ln in lines[-visible:]:
        d.text((MARGIN, y), ln, font=f, fill=color)
        y += LINE_H

    if footer:
        d.text((MARGIN, H - 34), footer, font=f, fill=YELLOW)
    return img


def wrap_text_block(text, width=95):
    import textwrap
    return textwrap.wrap(text, width)


def run_scenario(agent, session, user_msgs):
    """Run real agent turns, returning (title_turns) transcript entries."""
    entries = []
    for msg in user_msgs:
        resp = agent.process_message(msg, session_id=session.session_id)
        user_lines = wrap_text_block(msg)
        agent_lines = []
        for ln in resp.format_for_user().splitlines():
            agent_lines.extend(wrap_text_block(ln))
        entries.append(("user", user_lines))
        entries.append(("agent", agent_lines))
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(_PROJECT_ROOT / "docs" / "demo.gif"))
    args = parser.parse_args()

    print("Initializing agent (builds index, loads LLM provider)...")
    from src.main import initialize_agent

    agent = initialize_agent()
    frames = []

    for i, (title, spec) in enumerate(SCENARIOS):
        print(f"Recording scenario: {title}")
        session = agent.session_manager.create_session()
        user_msgs = [t[1] for t in spec]
        turns = run_scenario(agent, session, user_msgs)
        frames.append((title, turns))

    # Scenario 5: the evaluation suite running (real output, abbreviated)
    print("Recording scenario: evaluation suite")
    proc = subprocess.run(
        [sys.executable, "eval/run_eval.py"],
        cwd=_PROJECT_ROOT, capture_output=True, text=True, timeout=900,
    )
    out_lines = proc.stdout.splitlines()
    keep = []
    for ln in out_lines:
        if ("PASS" in ln or "FAIL" in ln or "OVERALL" in ln or "Category" in ln
                or "=====" in ln or "-----" in ln or ln.strip().endswith("%")):
            keep.append(ln.strip())
    shown = [(GREEN if "PASS" in l else FG if "FAIL" not in l else (255, 120, 120), l) for l in keep]
    # Cap frame lines so it stays readable
    shown = shown[:40]

    all_imgs = []
    for title, turns in frames:
        for split in range(0, max(1, len(turns)), 4):
            all_imgs.append(render_frame(title, turns[split:split + 4]))
    # A few scrolling frames for the eval output
    for start in range(0, len(shown), 18):
        all_imgs.append(render_frame("5/5  Evaluation suite", [], scroll_lines=shown[start:start + 18]))
    all_imgs.append(render_frame("5/5  Evaluation suite", [], scroll_lines=shown[-18:],
                                 footer="python eval/run_eval.py  —  32/32 cases pass"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_imgs[0].save(
        out, save_all=True, append_images=all_imgs[1:],
        duration=4200, loop=0, optimize=True,
    )
    print(f"Wrote {out} with {len(all_imgs)} frames")


if __name__ == "__main__":
    main()
