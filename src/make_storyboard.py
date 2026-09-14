#!/usr/bin/env python3
"""
make_storyboard.py — turn a PDF into a paper2video storyboard with Claude.

    export ANTHROPIC_API_KEY=sk-ant-...
    python make_storyboard.py paper.pdf -o paper_storyboard.json

What it adds over `paper2video.py --plan-only`:

  * the prompt carries the live figure catalogue with each figure's one-line
    description, so the model picks figures that exist AND suit the content;
  * the result is validated — scene types, figure names, narration, page
    references, crop ranges — and any problems are sent back to the model to
    fix, up to --repair times;
  * `--assets` makes page:N figure scenes available, so the model can reuse the
    paper's own diagrams instead of only redrawing them;
  * the output is a file you are expected to edit. The planner is a first draft.

Needs paper2video.py and illustrations.py importable (it reuses the schema and
the figure registry), plus poppler for text extraction. The `anthropic` SDK is
used when installed; otherwise it falls back to urllib, so there is no hard
dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

try:
    import paper2video as P
except ImportError:
    sys.exit("make_storyboard: run this next to paper2video.py")

try:
    import illustrations as I
except ImportError:
    I = None

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-5"
SCENE_TYPES = {"title", "bullets", "equation", "figure", "illustration",
               "split", "bars", "clip", "banner", "takeaways"}


# --------------------------------------------------------------------------- #
# source material
# --------------------------------------------------------------------------- #

def pdf_text(pdf: Path, max_chars: int) -> str:
    out = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                         capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"pdftotext failed on {pdf}: {out.stderr.strip()[:200]}")
    txt = re.sub(r"[ \t]+", " ", out.stdout)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    if len(txt) <= max_chars:
        return txt
    # keep the front (abstract, intro, method) and the tail (results, limits)
    head = int(max_chars * 0.72)
    return txt[:head] + "\n\n[...omitted...]\n\n" + txt[-(max_chars - head):]


def pdf_pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
    m = re.search(r"^Pages:\s+(\d+)", out.stdout, re.M)
    return int(m.group(1)) if m else 0


def catalogue() -> tuple[str, set[str]]:
    """Figure names with their one-line descriptions, so the model can choose
    on meaning rather than on the name alone."""
    if I is None:
        return "(figure library unavailable — avoid illustration scenes)", set()
    lines, names = [], set()
    for name in sorted(I.PRIMITIVES):
        doc = (I.PRIMITIVES[name].__doc__ or "").strip().split("\n")[0]
        lines.append(f"  {name:<22} {doc[:96]}")
        names.add(name)
    if I.PRESETS:
        lines.append("")
        lines.append("  Presets (a primitive with one paper's values already "
                     "filled in; your opts still override):")
        for name, (prim, _) in sorted(I.PRESETS.items()):
            lines.append(f"  {name:<22} -> {prim}")
            names.add(name)
    return "\n".join(lines), names


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #

HOUSE_STYLE = """\
House style, follow all of it:

1. Lead with the problem the paper solves, not with the paper's structure. Do
   not write a scene per section.
2. Prefer `illustration` scenes. Reach for one whenever the idea is spatial,
   sequential or dynamic — a decomposition, a migration, a routing decision, a
   distribution changing shape. Use `bullets` only for genuine lists.
3. Every number that appears in a figure's opts must come from the paper. Do
   not invent values to make a chart look plausible. If you do not have the
   numbers, choose a scene type that does not need them.
4. Narration is spoken aloud. Write full sentences, no markdown, no citations,
   no "as shown in Figure 3". Expand symbols: say "alpha", "root N", "d sub k".
5. Each narration line triggers one reveal in the figure, so order the lines to
   match the order the figure should build.
6. Aim for 3 to 6 narration lines per scene, 14 to 22 scenes total.
7. End with a `takeaways` scene whose `caveat` field states, in proportion,
   what the paper does NOT establish: the scope of the evaluation, the
   baselines missing, the asterisks on the headline number. Be specific and
   quote the paper's own limitations where it states them.
8. Where the results are uneven, say so in the narration rather than only
   reporting the average.
"""

PROMPT = """You are turning a research paper into a short explainer video.

Return ONE JSON object and nothing else — no prose, no markdown fence.

=== STORYBOARD SCHEMA ===
{schema}

=== FIGURES YOU MAY NAME ===
Use these exact names in the "figure" field. Any other name will not render.
{figures}

{assets}

{style}

=== PAPER ===
{paper}
"""

REPAIR = """The storyboard you returned has problems. Fix every one and return
the corrected JSON object, complete, with nothing else around it.

{errors}
"""


def build_prompt(paper: str, figures: str, assets_note: str,
                 extra: str | None) -> str:
    style = HOUSE_STYLE + (f"\n9. Additional instruction: {extra}\n" if extra else "")
    return PROMPT.format(schema=P.SCHEMA_DOC, figures=figures,
                         assets=assets_note, style=style, paper=paper)


# --------------------------------------------------------------------------- #
# the call
# --------------------------------------------------------------------------- #

def ask(messages: list[dict], model: str, key: str, max_tokens: int) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        r = client.messages.create(model=model, max_tokens=max_tokens,
                                   messages=messages)
        return "".join(b.text for b in r.content if b.type == "text")
    except ImportError:
        pass
    import urllib.error
    import urllib.request
    body = json.dumps({"model": model, "max_tokens": max_tokens,
                       "messages": messages}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json", "x-api-key": key,
        "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf8", "replace")[:400]
        sys.exit(f"Anthropic API returned {e.code}: {detail}")
    return "".join(b.get("text", "") for b in payload.get("content", []))


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(raw[start:end + 1])


# --------------------------------------------------------------------------- #
# validation — the part that makes the loop worth having
# --------------------------------------------------------------------------- #

def validate(board: dict, known: set[str], n_pages: int) -> list[str]:
    errs: list[str] = []
    scenes = board.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return ["The object has no non-empty \"scenes\" list."]
    if not board.get("title"):
        errs.append("Top level is missing a \"title\".")

    for i, sc in enumerate(scenes):
        tag = f"scene {i}"
        if not isinstance(sc, dict):
            errs.append(f"{tag}: not an object.")
            continue
        st = sc.get("type")
        if st not in SCENE_TYPES:
            errs.append(f"{tag}: type {st!r} is not one of "
                        f"{', '.join(sorted(SCENE_TYPES))}.")
        lines = sc.get("narration")
        if not isinstance(lines, list) or not lines:
            errs.append(f"{tag}: needs a non-empty \"narration\" list.")
        else:
            for j, ln in enumerate(lines):
                short = (not isinstance(ln, str)
                         or (len(ln.split()) < 2 and len(str(ln).strip()) < 12))
                if short:
                    errs.append(f"{tag} narration[{j}]: too short to speak.")
                elif re.search(r"[*_`#]|\[\d+\]|\bFig(ure)?\.? ?\d", str(ln)):
                    errs.append(f"{tag} narration[{j}]: remove markdown, "
                                f"citations and figure references — this is "
                                f"read aloud.")

        if st == "illustration":
            fig = sc.get("figure")
            if not fig:
                errs.append(f"{tag}: illustration scenes need a \"figure\".")
            elif known and fig not in known:
                near = ", ".join(_near(fig, known)) or "none"
                errs.append(f"{tag}: figure {fig!r} does not exist. "
                            f"Closest available: {near}.")
        if st == "figure":
            img = str(sc.get("image", ""))
            if img.startswith("page:"):
                try:
                    n = int(img.split(":", 1)[1])
                except ValueError:
                    errs.append(f"{tag}: image {img!r} is malformed.")
                else:
                    if n_pages and not 1 <= n <= n_pages:
                        errs.append(f"{tag}: page:{n} is outside the document "
                                    f"(1..{n_pages}).")
            elif not img:
                errs.append(f"{tag}: figure scenes need an \"image\".")
            crop = sc.get("crop")
            if crop is not None:
                if (not isinstance(crop, list) or len(crop) != 4
                        or not all(isinstance(c, (int, float)) for c in crop)):
                    errs.append(f"{tag}: crop must be [x0, y0, x1, y1].")
                elif not all(0 <= c <= 1 for c in crop):
                    errs.append(f"{tag}: crop values are fractions of the page "
                                f"and must lie in 0..1.")
                elif crop[0] >= crop[2] or crop[1] >= crop[3]:
                    errs.append(f"{tag}: crop must satisfy x0<x1 and y0<y1.")

    if scenes[-1].get("type") != "takeaways":
        errs.append("The last scene must be a \"takeaways\" scene.")
    elif not scenes[-1].get("caveat"):
        errs.append("The takeaways scene needs a \"caveat\" saying what the "
                    "paper does not establish.")
    if len(scenes) < 8:
        errs.append(f"Only {len(scenes)} scenes — aim for 14 to 22.")

    # an equation scene carrying a "figure" animates a diagram beside the
    # formula, so it counts as visual too
    ill = sum(1 for s in scenes if isinstance(s, dict)
              and (s.get("type") in ("illustration", "figure", "split", "bars")
                   or (s.get("type") == "equation" and s.get("figure"))))
    if ill < len(scenes) * 0.45:
        errs.append(f"Only {ill} of {len(scenes)} scenes are visual. Convert "
                    f"text-heavy scenes into illustration scenes.")
    return errs


def _near(name: str, known: set[str], n: int = 5) -> list[str]:
    import difflib
    return difflib.get_close_matches(name, sorted(known), n=n, cutoff=0.35)


def summarise(board: dict) -> str:
    scenes = board["scenes"]
    kinds: dict[str, int] = {}
    for s in scenes:
        kinds[s.get("type", "?")] = kinds.get(s.get("type", "?"), 0) + 1
    beats = sum(len(s.get("narration", [])) for s in scenes)
    words = sum(len(" ".join(s.get("narration", [])).split()) for s in scenes)
    kind = ", ".join(f"{v}x {k}" for k, v in sorted(kinds.items()))
    return (f"{len(scenes)} scenes, {beats} narration beats, {words} words "
            f"(~{words / 150:.1f} min)\n      {kind}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Draft a paper2video storyboard from a PDF using Claude.")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output JSON (default: <pdf stem>_storyboard.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key-env", default="ANTHROPIC_API_KEY")
    ap.add_argument("--max-chars", type=int, default=90000,
                    help="how much of the paper to send")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--repair", type=int, default=2,
                    help="rounds of validate-and-fix before giving up")
    ap.add_argument("--assets", type=Path, default=None,
                    help="directory of page-N.png renders; enables figure "
                         "scenes that reuse the paper's own diagrams")
    ap.add_argument("--extra", default=None,
                    help="one extra instruction, e.g. 'focus on the method'")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt and exit without calling the API")
    a = ap.parse_args(argv)

    if not a.pdf.exists():
        sys.exit(f"no such PDF: {a.pdf}")
    out = a.out or a.pdf.with_name(a.pdf.stem + "_storyboard.json")

    text = pdf_text(a.pdf, a.max_chars)
    n_pages = pdf_pages(a.pdf)
    figures, known = catalogue()

    if a.assets and a.assets.exists():
        have = len(sorted(a.assets.glob("page-*.png")))
        assets_note = textwrap.dedent(f"""\
            === THE PAPER'S OWN FIGURES ===
            {have} page renders are available as page-1 .. page-{have}. When the
            paper already has a good figure for an idea, prefer a `figure` scene
            over redrawing it:
              {{"type": "figure", "image": "page:7",
                "crop": [0.05, 0.18, 0.95, 0.72], "heading": "...",
                "caption": "credit the source", "narration": [...]}}
            crop is [x0, y0, x1, y1] as fractions of the page, y from the top.
            Crop tightly: exclude the slide or page title, footers and logos.""")
    else:
        assets_note = ("=== THE PAPER'S OWN FIGURES ===\n"
                       "No page renders supplied, so do not emit `figure` "
                       "scenes. Draw everything with `illustration` scenes.")

    prompt = build_prompt(text, figures, assets_note, a.extra)
    if a.dry_run:
        print(prompt)
        return 0

    key = os.environ.get(a.key_env)
    if not key:
        sys.exit(f"{a.key_env} is not set")

    print(f"plan  {a.pdf.name} — {n_pages} pages, {len(text):,} chars sent")
    print(f"      model {a.model}, {len(known)} figures offered")

    messages = [{"role": "user", "content": prompt}]
    board = None
    for attempt in range(a.repair + 1):
        raw = ask(messages, a.model, key, a.max_tokens)
        try:
            board = parse_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            errs = [f"The reply was not valid JSON: {e}"]
        else:
            errs = validate(board, known, n_pages)
        if not errs:
            break
        print(f"      round {attempt + 1}: {len(errs)} problem(s)")
        for e in errs[:6]:
            print(f"        - {e}")
        if len(errs) > 6:
            print(f"        ... and {len(errs) - 6} more")
        if attempt == a.repair:
            print("      giving up on repair; writing the draft anyway")
            break
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user",
                      "content": REPAIR.format(errors="\n".join(f"- {e}" for e in errs))}]

    if board is None:
        sys.exit("no usable storyboard was produced")
    out.write_text(json.dumps(board, indent=2, ensure_ascii=False))
    print(f"\ndone  {summarise(board)}")
    print(f"      -> {out}")
    print("      this is a draft: read the caveat scene and check every number "
          "against the paper before rendering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())