#!/usr/bin/env python3
"""
storyboard2pptx.py — turn a paper2video storyboard into a PowerPoint deck.

The deck is rendered by the *same* matplotlib code that draws the video, so a
slide and the corresponding video frame are pixel-identical. Narration becomes
speaker notes.

    python storyboard2pptx.py lacot_storyboard.json -o lacot.pptx

Two modes:

    --per-beat   (default) one slide per narration line, each showing the
                 figure exactly as it looks when that line finishes. Advancing
                 the deck reproduces the video's reveals as PowerPoint builds.
    --per-scene  one slide per scene, showing its final state. Shorter deck;
                 all of the scene's narration goes into that slide's notes.

Needs paper2video.py and illustrations.py beside it (it imports the renderer),
plus python-pptx:  pip install python-pptx

Slides are 16:9. By default the *figure* is a picture and the *text* around it
(heading, caption) is real PowerPoint text — editable, searchable, and readable
by screen readers.

By default the figure is drawn with real PowerPoint shapes, tables and charts
wherever pptx_native.py has an emitter for it; anything without one falls back
to a rendered picture, and the run reports the split.

    --no-native draw every figure as a picture. Slower to edit, but pixel-
                identical to the video, which native shapes are not.
    --flat      bake heading and caption into the picture too, so the slide
                has no selectable text at all.

The plot itself is always a picture. The figures are matplotlib drawings with
no native PowerPoint equivalent, and the one vector route python-pptx accepts
(SVG -> LibreOffice -> EMF) does not round-trip faithfully: it shifts the rule
under the heading and recolours some axis lines. Raster at --dpi 140 is sharp
to well past 100% zoom, so that is what this does.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

# ── the renderer, borrowed wholesale ────────────────────────────────────────
try:
    import paper2video as P
except ImportError:
    sys.exit("storyboard2pptx: put this next to paper2video.py "
             "(it reuses the video's renderer so slides match the frames)")

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt
except ImportError:
    sys.exit("storyboard2pptx: pip install python-pptx")

try:
    import pptx_native as NATIVE
except ImportError:
    NATIVE = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402


W_IN, H_IN = 13.3333, 7.5                # 16:9 at PowerPoint's wide default
SU_W, SU_H = 16.0, 9.0                   # the renderer's scene units


def _x(u):                               # scene units -> inches from the left
    return Inches(u / SU_W * W_IN)


def _y(v):                               # scene units are y-up, slides y-down
    return Inches((SU_H - v) / SU_H * H_IN)


def _rgb(hex_str):
    return RGBColor.from_string(hex_str.lstrip("#").upper()[:6])


def add_textbox(slide, text, left, top, width, height, size, color,
                bold=False, align=None):
    from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    if align:
        p.alignment = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT}[align]
    for run in p.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = _rgb(color)
    return box


class _Args:
    """The handful of fields plan_timing reads. Kept separate from argparse so
    the timing matches the video defaults unless the caller overrides them."""

    def __init__(self, wpm, beat_gap, min_scene, max_scene):
        self.wpm = wpm
        self.beat_gap = beat_gap
        self.min_scene = min_scene
        self.max_scene = max_scene


def frames_for(board, args) -> list[tuple[dict, float, float, list[str]]]:
    """(scene, u, scene_duration, notes) for every slide we intend to draw.

    `u` is the moment within the scene to freeze. Beat boundaries come from
    plan_timing with no audio, so they land where the video's reveals land
    when narration is read at `--wpm`.
    """
    scenes = board["scenes"]
    timing = P.plan_timing(scenes, [[] for _ in scenes], args)
    out = []
    for sc, tm in zip(scenes, timing):
        lines = P.narration_lines(sc)
        if args.per_scene or not tm.beats:
            out.append((sc, max(tm.dur - 0.25, 0.1), tm.dur, lines))
            continue
        for b, beat in enumerate(tm.beats):
            # freeze just after this line finishes, so its reveal has landed
            u = min(beat.start + beat.dur + 0.25, tm.dur - 0.05)
            out.append((sc, u, tm.dur, [lines[b]] if b < len(lines) else []))
    return out


def render_frames(board, args, pages, tmp: Path) -> list[tuple[Path, list[str]]]:
    plan = frames_for(board, args)
    scenes = board["scenes"]
    idx_of = {id(sc): i for i, sc in enumerate(scenes)}
    r = P.Renderer(board.get("theme", "paper"), args.base, 30,
                   (args.width, int(round(args.width * 9 / 16))))
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    made = []
    for n, (sc, u, dur, notes) in enumerate(plan):
        tm = P.plan_timing([sc], [[]], args)[0]
        r.beats = tm.beats
        ax = r.axes(fig)
        draw = sc if args.flat else {k: v for k, v in sc.items()
                                     if k not in ("heading", "caption")}
        try:
            r.scene(ax, draw, u, dur, idx_of[id(sc)], pages)
        except Exception as e:                     # never lose the whole deck
            print(f"      slide {n + 1}: {type(e).__name__}: {e}")
        img = tmp / f"slide{n:03d}.png"
        fig.savefig(img, dpi=args.dpi, facecolor=r.c["bg"])
        made.append((img, notes, sc))
        if (n + 1) % 10 == 0 or n + 1 == len(plan):
            print(f"      {n + 1}/{len(plan)} slides rendered")
    plt.close(fig)
    return made


def build(board, frames, out: Path, args) -> None:
    pres = Presentation()
    pres.slide_width, pres.slide_height = Inches(W_IN), Inches(H_IN)
    blank = pres.slide_layouts[6]                  # blank: no placeholders
    theme = P.THEMES[board.get("theme", "paper")]
    native_ok = 0
    for img, notes, sc in frames:
        slide = pres.slides.add_slide(blank)
        drew = False
        if args.native and sc.get("type") == "illustration":
            drew = NATIVE.emit(slide, sc.get("figure", ""),
                               sc.get("opts"), theme)
            native_ok += bool(drew)
        if not drew:
            slide.shapes.add_picture(str(img), 0, 0,
                                     width=Inches(W_IN), height=Inches(H_IN))
        else:
            # the picture normally carries the header rule and scene number,
            # so a natively drawn slide has to supply them
            NATIVE.header(slide, theme, len(pres.slides._sldIdLst))
        if not args.flat:
            head = sc.get("heading", "")
            if head:
                add_textbox(slide, head, _x(1.65), _y(8.72),
                            _x(13.2), Inches(0.62), 21, theme["ink"], bold=True)
            cap = sc.get("caption", "")
            if cap:
                add_textbox(slide, cap, _x(0.85), _y(2.22),
                            _x(14.30), Inches(0.77), 15,
                            theme.get("accent_ink", theme["ink"]),
                            bold=True, align="center")
        if notes:
            tf = slide.notes_slide.notes_text_frame
            tf.text = "\n".join(notes)
            for para in tf.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(12)
    if args.native:
        n = len(frames)
        print(f"      {native_ok}/{n} slides drawn as PowerPoint objects, "
              f"{n - native_ok} as pictures")
    pres.save(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render a paper2video storyboard as a PowerPoint deck.")
    ap.add_argument("storyboard", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output .pptx (default: alongside the storyboard)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--per-beat", dest="per_scene", action="store_false",
                   help="one slide per narration line (default) — reveals "
                        "become PowerPoint builds")
    g.add_argument("--per-scene", dest="per_scene", action="store_true",
                   help="one slide per scene, final state only")
    ap.set_defaults(per_scene=False)
    ap.add_argument("--no-native", dest="native", action="store_false",
                    help="draw every figure as a picture instead of shapes; "
                         "pixel-identical to the video")
    ap.set_defaults(native=True)
    ap.add_argument("--flat", action="store_true",
                    help="bake heading and caption into the picture too; "
                         "no selectable text anywhere on the slide")
    ap.add_argument("--assets", type=Path, default=None,
                    help="directory of page-N.png for 'figure' scenes")
    ap.add_argument("--dpi", type=int, default=140,
                    help="render dpi; 140 gives ~2240x1260 per slide")
    ap.add_argument("--width", type=int, default=1920,
                    help="layout width in px (font sizes are absolute, so "
                         "this only affects the drawing grid)")
    ap.add_argument("--wpm", type=int, default=150,
                    help="reading pace used to place the reveals")
    ap.add_argument("--beat-gap", type=float, default=0.45)
    ap.add_argument("--min-scene", type=float, default=4.0)
    ap.add_argument("--max-scene", type=float, default=26.0)
    ap.add_argument("--keep-frames", type=Path, default=None,
                    help="also write the slide PNGs here")
    a = ap.parse_args()

    if not a.storyboard.exists():
        sys.exit(f"no such storyboard: {a.storyboard}")
    board = json.loads(a.storyboard.read_text())
    out = a.out or a.storyboard.with_suffix(".pptx")

    a.base = a.assets or a.storyboard.parent
    pages = (sorted(a.assets.glob("page-*.png"))
             if a.assets and a.assets.exists() else [])

    scenes = board["scenes"]
    n_beats = sum(len(P.narration_lines(s)) for s in scenes)
    mode = "per scene" if a.per_scene else "per narration beat"
    print(f"plan  {len(scenes)} scenes, {n_beats} narration beats")
    if a.flat:
        a.native = False                     # --flat means one picture, period
    if a.native and NATIVE is None:
        print("      pptx_native.py not found — falling back to pictures")
        a.native = False
    layer = ("everything baked into one picture" if a.flat
             else "native shapes where available, pictures elsewhere"
             if a.native
             else "figure as a picture, heading and caption as real text")
    print(f"      {layer}")
    print(f"      one slide {mode}"
          f"{'' if not pages else f', {len(pages)} paper pages available'}")

    tmp = Path(tempfile.mkdtemp(prefix="sb2pptx-"))
    try:
        frames = render_frames(board, a, pages, tmp)
        build(board, frames, out, a)
        if a.keep_frames:
            a.keep_frames.mkdir(parents=True, exist_ok=True)
            for img, _, _sc in frames:
                shutil.copy2(img, a.keep_frames / img.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    mb = out.stat().st_size / 1e6
    print(f"\ndone  {len(frames)} slides  ->  {out}  ({mb:.1f} MB)")
    print("      narration is in the speaker notes"
          + ("" if a.flat else "; heading and caption are editable text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())