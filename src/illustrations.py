"""
illustrations.py — the animated-figure library for paper2video.

Two layers, and the split is the whole point.

    PRIMITIVES   generic, fully opts-driven drawing functions. No paper-specific
                 strings or numbers anywhere. `pipeline` draws a box-and-arrow
                 flow for any architecture; `curve_family` plots any scaling
                 law. These are what you reuse.

    PRESETS      a named primitive plus a bag of opts. "scaled_dot_product" is
                 just `dataflow` with the Transformer paper's nodes filled in.
                 Presets are data, not code.

A storyboard names either one:

    {"type": "illustration", "figure": "pipeline",
     "opts": {"stages": [["encoder","in"], ["projector","hot"], ["LLM","mid"]]}}

Layout
------
    illustrations.py         this file: helpers, the common primitives, and the
                             preset dictionary covering the papers below
    illustrations_extra.py   the long tail: rarer primitives and their presets,
                             merged in automatically if the file is present

Adding a paper
--------------
1. Try to express each figure as an existing primitive plus opts, and add the
   result to PRESETS. Most figures land here and need no new code.
2. If the *shape* is genuinely new — new motion or geometry, not new labels —
   add a primitive. Common shapes go here; niche ones go in the extra file.
   Name the shape, never the paper: `stacked_blocks`, not `transformer_encoder`.

Contract for a primitive:  draw(R, ax, u, d, o)
    R   the Renderer: R.c colours, R.txt/R.box, R.rt(k, ...) beat timing
    ax  axes in scene units, 0..16 x 0..9, aspect equal
    u   seconds into the scene;  d  scene duration;  o  the opts dict
Honour _box(o) so the figure also works beside an equation in a half-width
panel, and use R.rt(k, fallback, step) so reveals land on narration beats.

`python paper2video.py --figures` prints the live catalogue.
"""

import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)   # noqa: F401

W, H = 16.0, 9.0



# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _fade(u, start, dur=0.55):
    return _ease((u - start) / dur) if dur > 0 else float(u >= start)


def _box(o, default=(0.85, 2.40, 14.30, 5.20)):
    """Drawing rectangle (x, y, w, h). Figures honour this so the same figure
    can fill the frame or sit beside an equation in a half-width panel."""
    b = o.get("box")
    return tuple(b) if b else default


def _pal(R):
    """Two extra semantic colours: the visual modality and the text modality."""
    dark = R.c["bg"].lower() in ("#11151c",)
    return {
        "vis": "#4FD1C5" if dark else "#0E8A7D",
        "txt": "#B794F4" if dark else "#7C4DBE",
        "grid": R.c["line"],
    }


def _col(R, key, default="dim"):
    """Resolve a colour name: hot / in / out / pos / neg / mid / dim / ink,
    or a literal hex string."""
    C, P = R.c, _pal(R)
    if isinstance(key, str) and key.startswith("#"):
        return key
    return {"hot": C["accent"], "in": P["vis"], "out": P["txt"],
            "pos": C["pos"], "neg": C["neg"], "mid": C["panel"],
            "dim": C["dim"], "ink": C["ink"], "muted": C["muted"]
            }.get(key or default, C["dim"])


def _tex(expr: str) -> str:
    """Rewrite macros mathtext does not know, then verify the expression
    parses; fall back to stripped plain text rather than raising.

    A single unsupported macro must never abort a long render.
    """
    if not expr:
        return ""
    import re as _re
    out = expr
    for bad, good in ((r"\tfrac", r"\frac"), (r"\dfrac", r"\frac"),
                      (r"\cfrac", r"\frac"), (r"\Bigl", r"\left"),
                      (r"\Bigr", r"\right"), (r"\bigl", r"\left"),
                      (r"\bigr", r"\right"), (r"\Big", ""), (r"\big", ""),
                      (r"\text", r"\mathrm"), (r"\mbox", r"\mathrm"),
                      (r"\nonumber", "")):
        out = out.replace(bad, good)
    try:
        from matplotlib import mathtext
        mathtext.MathTextParser("path").parse(out, 100, None)
        return out
    except Exception:
        return _re.sub(r"\\[a-zA-Z]+", "", expr).replace("$", "")


def _tail(R, ax, o, u, beat, x, y, w, key="verdict"):
    """The one-line conclusion most figures end on."""
    t = o.get(key)
    if not t:
        return
    a = _fade(u, R.rt(beat, 4.0, 0.9), 0.7)
    R.txt(ax, x + w / 2, y + 0.22, t, 20, R.c["accent"], a, ha="center",
          weight="bold")


# --------------------------------------------------------------------------- #
# structure and process
# --------------------------------------------------------------------------- #


def _arrow(ax, p0, p1, color, lw=2.6, a=1.0, ms=16, ls="-", z=8):
    if a <= 0.01:
        return
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 lw=lw, color=color, alpha=min(a, 1.0),
                                 linestyle=ls, zorder=z,
                                 shrinkA=0, shrinkB=0))


def _proj3(p, cx, cy, s=1.0):
    """Oblique projection of (x, y, z); the plane z=0 reads as a receding plane."""
    x, y, z = p
    return (cx + s * (x + 0.52 * y), cy + s * (0.34 * y + z))


def _chip(R, ax, x, y, w, h, label, fc, a=1.0, size=14):
    if a <= 0.01:
        return
    R.box(ax, x, y, w, h, a=a, fc=fc, r=0.10, z=6)
    R.txt(ax, x + w / 2, y + h / 2, label, size, "#FFFFFF", a, ha="center", z=9)


# --------------------------------------------------------------------------- #
# 1. open_world — the problem setup
# --------------------------------------------------------------------------- #


# default cluster geometry for cluster_walk
_ID_C = [(-1.15, 0.85), (1.05, 1.00), (1.20, -0.55)]
_OOD_C = (-1.10, -1.05)
_ID_T = [(-1.46, 1.28), (1.41, 1.36), (1.56, -0.97)]
_OOD_T = (-1.54, -1.44)


def _stream(n=160, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = i % 4
        if k < 3:
            c = _ID_C[k]
            out.append((c[0] + rng.normal(0, .30), c[1] + rng.normal(0, .30),
                        True, float(rng.uniform(.80, .99))))
        else:
            c = _OOD_C
            out.append((c[0] + rng.normal(0, .34), c[1] + rng.normal(0, .34),
                        False, float(rng.uniform(.01, .30))))
    return out




# --------------------------------------------------------------------------- #
# shared model of a fitted scaling law (used by curve_family presets)
# --------------------------------------------------------------------------- #

ALPHA, BETA, K, D0 = 0.077, 0.015, 2.43, 0.10
NS = [0.5, 1.8, 4.0, 7.0]                    # LLM params, billions
VS = [1, 4, 16, 36, 64, 144, 576]            # visual tokens


def _err(n_b, v, alpha=ALPHA, beta=BETA):
    return K / ((n_b * 1e9) ** alpha * v ** beta) + D0


def _fl(n_b, v, q=0):
    return n_b * (q + v)                     # O(N(Q+V)), arbitrary units


# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# PRIMITIVES
# --------------------------------------------------------------------------- #

def annotated_equation(R, ax, u, d, o):
    """A formula with callouts pointing at individual terms, revealed one per
    beat. The clearest way to explain what each symbol is doing.

    opts: latex, terms [{text, at (0..1 across the formula), note, colour}],
          verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ey = y + h * 0.72
    R.txt(ax, x + w / 2, ey, _tex(o.get("latex", "")), 30, C["ink"], a0,
          ha="center")

    terms = o.get("terms", [])
    for k, t in enumerate(terms):
        a = _fade(u, R.rt(k + 1, 1.4 + k * 1.2, 0.7), 0.6)
        if a <= 0.01:
            continue
        col = _col(R, t.get("colour", "hot"))
        tx = x + 0.8 + float(t.get("at", (k + 0.5) / max(len(terms), 1))) * (w - 1.6)
        ty = ey - 1.15 - (k % 2) * 1.35
        ax.plot([tx, tx], [ey - 0.45, ty + 0.42], lw=2.0, color=col,
                alpha=min(a, 1.0) * 0.8, zorder=7)
        ax.plot([tx], [ey - 0.42], "o", ms=8, color=col, alpha=min(a, 1.0),
                zorder=8)
        R.txt(ax, tx, ty, t.get("text", ""), 19, col, a, ha="center",
              weight="bold")
        if t.get("note"):
            R.txt(ax, tx, ty - 0.45, t["note"], 15, C["muted"], a, ha="center")
    _tail(R, ax, o, u, len(terms) + 1, x, y, w)


def area_compare(R, ax, u, d, o):
    """FLOPs = N x T. Two very different configurations, identical area."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    unit = min(w / 8.2, h / 6.2)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, x + w / 2, y + h - 0.15, o.get("title", ""),
          24, C["ink"], a0, ha="center", weight="bold")

    cfgs = [tuple(c) for c in o.get("configs", [
        ["4B LLM", "576 tokens", 1.0, 3.4, C["dim"], 1],
        ["7B LLM", "36 tokens", 1.75, 1.94, C["accent"], 2]])]
    for i, (nl, tl, ww, hh, col, beat) in enumerate(cfgs):
        a = _fade(u, R.rt(beat, 1.2 + i * 1.4, 0.7), 0.6)
        if a <= 0.01:
            continue
        bx = x + 0.7 + i * (w / 2 + 0.1)
        by = y + 1.5
        ax.add_patch(Rectangle((bx, by), ww * unit, hh * unit, fc=col,
                               ec="none", alpha=min(a, 1.0) * 0.85, zorder=6))
        R.txt(ax, bx + ww * unit / 2, by - 0.40, tl, 17, C["muted"], a,
              ha="center")
        R.txt(ax, bx - 0.22, by + hh * unit / 2, nl, 17, C["ink"], a,
              ha="right", rot=90)

    a3 = _fade(u, R.rt(3, 4.2, 0.8), 0.7)
    R.txt(ax, x + w / 2, y + 0.35, o.get("verdict", ""), 20,
          C["accent"], a3, ha="center", weight="bold")


def attractor(R, ax, u, d, o):
    """Theorem 1: own-class images pull the prototype in, others push it out."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    cx, cy = x + w / 2, y + h / 2 + 0.25
    rng = np.random.default_rng(5)

    own = [(cx + 1.15 + rng.normal(0, .42), cy + .35 + rng.normal(0, .55))
           for _ in range(8)]
    oth = [(cx - 1.45 + rng.normal(0, .48), cy - .30 + rng.normal(0, .70))
           for _ in range(10)]

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for px, py in own:
        ax.plot([px], [py], "o", ms=9, color=C["pos"], alpha=min(a0, 1.) * .8,
                zorder=5)
    a1 = _fade(u, R.rt(1, 1.4, 0.6), 0.6)
    for px, py in oth:
        ax.plot([px], [py], "o", ms=9, color=C["dim"], alpha=min(a1, 1.) * .8,
                zorder=5)

    # the prototype slides toward its own class as the pulls are applied
    t2 = R.rt(2, 2.6, 0.8)
    p = _ease(min(1.0, max(0.0, (u - t2) / max(d - t2 - 1.2, 1.5))))
    wx, wy = cx - 0.55 + 1.25 * p, cy - 0.05 + 0.30 * p

    ap = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    if _fade(u, t2, 0.5) > 0.02:
        for px, py in own[:5]:
            _arrow(ax, (wx, wy), (px, py), C["pos"], 1.6,
                   _fade(u, t2, 0.6) * 0.5, 10)
        for px, py in oth[:5]:
            _arrow(ax, (wx, wy), (2 * wx - px, 2 * wy - py), C["neg"], 1.6,
                   _fade(u, R.rt(3, 3.6, 0.8), 0.6) * 0.45, 10)
    ax.plot([wx], [wy], "*", ms=28, color=P["vis"], alpha=min(ap, 1.0), zorder=9)
    R.txt(ax, wx + 0.30, wy - 0.45, o.get("marker_label", ""), 24, P["vis"], ap,
          weight="bold")

    R.txt(ax, x, y + 0.72, o.get("pull_label", ""), 17,
          C["pos"], _fade(u, t2, 0.6))
    R.txt(ax, x, y + 0.28, o.get("push_label", ""), 17,
          C["neg"], _fade(u, R.rt(3, 3.6, 0.8), 0.6))


_STREAM = _stream()


def bars(R, ax, u, d, o):
    """Horizontal bars, one per beat, with an optional highlighted row.

    Accepts either the compact form
        "items": [[label, value] | [label, value, colour]]
    or the older parallel arrays "names" and "values", in which case the
    largest bar is highlighted automatically.

    opts: items | (names, values), title, fmt, max, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    items = o.get("items")
    if items:
        rows = [(str(i[0]), float(i[1]),
                 _col(R, i[2]) if len(i) > 2 else C["dim"],
                 len(i) > 2 and i[2] == "hot") for i in items]
    else:
        names = o.get("names", ["cat", "dog", "bus", "mushroom"])
        vals = [float(v) for v in o.get("values", [0.71, 0.44, 0.22, 0.10])]
        best = int(np.argmax(vals)) if vals else -1
        rows = [(str(nm), v, C["accent"] if k == best else C["dim"], k == best)
                for k, (nm, v) in enumerate(zip(names, vals))]
    if not rows:
        return
    mx = float(o.get("max") or max(v for _, v, _, _ in rows) * 1.06) or 1.0
    fmt = o.get("fmt")
    labw = max(len(nm) for nm, _, _, _ in rows)
    bx = x + min(w * 0.45, 0.7 + labw * 0.17)
    bw = w - (bx - x) - (1.5 if fmt else 0.5)
    gap = min(0.86, (h - 1.7) / len(rows))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.5)
    R.txt(ax, x, y + h - 0.15, o.get("title", ""), 18, C["muted"], a0)
    for k, (nm, v, col, hot) in enumerate(rows):
        t0 = R.rt(k, 1.0, 0.5)
        a = _fade(u, t0, 0.45)
        if a <= 0.01:
            continue
        g = max(_ease((u - t0) / 0.7), 0.0)
        yy = y + h - 0.95 - k * gap
        ax.add_patch(Rectangle((bx, yy - 0.20), bw * (v / mx) * g, 0.40,
                               fc=col, ec="none", alpha=min(a, 1.0), zorder=6))
        R.txt(ax, bx - 0.25, yy, nm, 16, C["ink"] if hot else C["muted"], a,
              ha="right", weight="bold" if hot else "normal")
        if fmt:
            R.txt(ax, bx + bw * (v / mx) * g + 0.20, yy, fmt.format(v), 15,
                  C["ink"] if hot else C["muted"], a * g,
                  weight="bold" if hot else "normal")
    am = _fade(u, R.rt(len(rows), 2.6, 0.8), 0.6)
    import textwrap as _tw
    for j, ln in enumerate(_tw.wrap(o.get("verdict", ""), 74)[:2]):
        R.txt(ax, x + w / 2, y + 0.42 - j * 0.42, ln, 20, C["accent"], am,
              ha="center", weight="bold")

def before_after(R, ax, u, d, o):
    """Two grids side by side with an arrow between — any transformation of a
    structured input: compression, denoising, masking, editing.

    opts: left {label, n, density}, right {label, n, density}, arrow_label,
          verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    cy = y + h * 0.58
    side = min(w * 0.28, h * 0.62)

    for i, (key, beat) in enumerate((("left", 0), ("right", 2))):
        spec = o.get(key, {})
        a = _fade(u, R.rt(beat, 0.4 + i * 1.8, 0.6), 0.6)
        if a <= 0.01:
            continue
        n = int(spec.get("n", 12))
        cell = side / n
        gx = x + (0.9 if i == 0 else w - side - 0.9)
        gy = cy + side / 2
        rng = np.random.default_rng(4 + i)
        for r_ in range(n):
            for c_ in range(n):
                v = rng.uniform(0.15, 1.0) * float(spec.get("density", 1.0))
                ax.add_patch(Rectangle((gx + c_ * cell, gy - (r_ + 1) * cell),
                                       cell * 0.88, cell * 0.88,
                                       fc=_col(R, spec.get("colour",
                                                           "in" if i == 0 else "hot")),
                                       ec="none", alpha=min(a, 1.0) * v,
                                       zorder=6))
        R.txt(ax, gx + side / 2, gy + 0.42, str(spec.get("label", "")), 18,
              C["ink"], a, ha="center", weight="bold")

    a1 = _fade(u, R.rt(1, 1.6, 0.7), 0.6)
    _arrow(ax, (x + side + 1.3, cy), (x + w - side - 1.3, cy), C["accent"],
           3.0, a1, 18)
    R.txt(ax, x + w / 2, cy + 0.55, o.get("arrow_label", ""), 18, C["accent"],
          a1, ha="center", weight="bold")
    _tail(R, ax, o, u, 3, x, y, w)


def cluster_walk(R, ax, u, d, o):
    C, P = R.c, _pal(R)
    cx, cy, s = 10.9, 4.75, 1.24
    t_stream = R.rt(1, 2.0, 1.2)
    t_move = R.rt(2, 3.2, 1.2)
    span = max(float(o.get("walk_seconds", 0.0)) or (d - t_move - 0.8), 2.0)
    rate = float(o.get("rate", 11.0))

    ap = _fade(u, R.rt(0, 0.5, 0.7), 0.7)
    ax.add_patch(Circle((cx, cy), 2.05 * s, fill=False, ec=P["grid"], lw=2.0,
                        alpha=min(ap, 1.0), zorder=3))
    R.txt(ax, cx, cy + 2.05 * s + 0.42, o.get("space_label", ""), 17, C["muted"],
          ap, ha="center")

    n = min(len(_STREAM), int(max(0.0, u - t_stream) * rate))
    for j in range(n):
        x, y, isid, sc = _STREAM[j]
        ax.plot([cx + x * s], [cy + y * s], "o", ms=6.5,
                color=C["pos"] if isid else C["neg"], alpha=0.32, zorder=5)

    p = _ease(min(1.0, max(0.0, (u - t_move) / span))) if u > t_move else 0.0
    for k in range(3):
        t0, c0 = _ID_T[k], _ID_C[k]
        px, py = t0[0] + (c0[0] - t0[0]) * p, t0[1] + (c0[1] - t0[1]) * p
        ax.plot([cx + t0[0] * s], [cy + t0[1] * s], "x", ms=13, mew=3.0,
                color=P["txt"], alpha=min(ap, 1.0) * .55, zorder=7)
        ax.plot([cx + t0[0] * s, cx + px * s], [cy + t0[1] * s, cy + py * s],
                ls=(0, (2, 3)), lw=1.6, color=C["muted"], alpha=.5, zorder=6)
        ax.plot([cx + px * s], [cy + py * s], "*", ms=22, color=C["pos"],
                alpha=min(ap, 1.0), zorder=9)
    px = _OOD_T[0] + (_OOD_C[0] - _OOD_T[0]) * p
    py = _OOD_T[1] + (_OOD_C[1] - _OOD_T[1]) * p
    ax.plot([cx + _OOD_T[0] * s], [cy + _OOD_T[1] * s], "x", ms=13, mew=3.0,
            color=P["txt"], alpha=min(ap, 1.0) * .55, zorder=7)
    ax.plot([cx + _OOD_T[0] * s, cx + px * s],
            [cy + _OOD_T[1] * s, cy + py * s], ls=(0, (2, 3)), lw=1.6,
            color=C["muted"], alpha=.5, zorder=6)
    ax.plot([cx + px * s], [cy + py * s], "*", ms=22, color=C["neg"],
            alpha=min(ap, 1.0), zorder=9)

    # routing bar on the left
    ab = _fade(u, t_stream - 0.4, 0.6)
    bx, by, bw = 1.0, 4.75, 5.6
    if ab > 0.01:
        beta = float(o.get("beta", 0.95))
        ax.add_patch(Rectangle((bx, by), bw * (1 - beta), .44, fc=C["neg"],
                               ec="none", alpha=min(ab, 1.) * .75, zorder=5))
        ax.add_patch(Rectangle((bx + bw * (1 - beta), by), bw * (2 * beta - 1),
                               .44, fc=C["dim"], ec="none",
                               alpha=min(ab, 1.) * .5, zorder=5))
        ax.add_patch(Rectangle((bx + bw * beta, by), bw * (1 - beta), .44,
                               fc=C["pos"], ec="none", alpha=min(ab, 1.) * .75,
                               zorder=5))
        R.txt(ax, bx + bw * (1 - beta), by - .36, r"$1-\beta$", 15, C["neg"],
              ab, ha="center")
        R.txt(ax, bx + bw * beta, by - .36, r"$\beta$", 15, C["pos"], ab,
              ha="center")
        R.txt(ax, bx + bw * .5, by + .22, o.get("skip_label", ""), 15, C["muted"],
              ab, ha="center")
        R.txt(ax, bx, by + .95, o.get("route_label", ""), 19,
              C["ink"], ab, weight="bold")
        if n:
            sc = _STREAM[min(n - 1, len(_STREAM) - 1)][3]
            ax.plot([bx + bw * sc], [by + .78], "v", ms=13, color=C["ink"],
                    alpha=min(ab, 1.0), zorder=9)

    al = _fade(u, R.rt(2, 1.6, 1.0), 0.6)
    ax.plot([1.15], [3.30], "x", ms=12, mew=3.0, color=P["txt"],
            alpha=min(al, 1.0))
    R.txt(ax, 1.50, 3.30, o.get("legend_a", ""), 16, C["muted"], al)
    ax.plot([1.15], [2.78], "*", ms=19, color=C["pos"], alpha=min(al, 1.0))
    R.txt(ax, 1.50, 2.78, o.get("legend_b", ""), 16, C["muted"], al)
    if n:
        R.txt(ax, 1.15, 2.20, f"{o.get('counter_label', 'seen')}: {n * 40:,}", 18, C["ink"],
              1.0, weight="bold")


# --------------------------------------------------------------------------- #
# 5. regret_curve
# --------------------------------------------------------------------------- #


def curve_family(R, ax, u, d, o):
    """A family of curves plus an optional Pareto frontier — the shape every
    scaling-law or trade-off-front figure takes.

    opts: series [{name, colour, points [[x, y] | [x, y, marker_size]]}],
          xlabel, ylabel, hi_label, lo_label, logx,
          frontier ("min"|"max"|null), frontier_label, note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    series = o.get("series", [])
    allp = [p for sr in series for p in sr.get("points", [])]
    if not allp:
        return
    xs = [float(p[0]) for p in allp]
    ys = [float(p[1]) for p in allp]
    logx = bool(o.get("logx", True))
    fx = (lambda v: np.log10(max(float(v), 1e-9))) if logx else float
    xlo, xhi = fx(min(xs)), fx(max(xs))
    ylo, yhi = min(ys), max(ys)
    pad = (yhi - ylo) * 0.12 or 1.0
    ylo, yhi = ylo - pad, yhi + pad

    def PX(v):
        return x + 0.95 + (fx(v) - xlo) / max(xhi - xlo, 1e-9) * (w - 1.4)

    def PY(v):
        return y + 0.95 + (float(v) - ylo) / max(yhi - ylo, 1e-9) * (h - 1.7)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([x + 0.95, x + 0.95], [y + 0.95, y + h - 0.45], color=C["line"],
            lw=1.6, alpha=min(a0, 1.0))
    ax.plot([x + 0.95, x + w - 0.45], [y + 0.95, y + 0.95], color=C["line"],
            lw=1.6, alpha=min(a0, 1.0))
    R.txt(ax, x + w / 2, y + 0.42, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.42, y + h / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)
    R.txt(ax, x + 1.15, y + h - 0.30, o.get("hi_label", ""), 14, C["muted"], a0)
    R.txt(ax, x + 1.15, y + 1.15, o.get("lo_label", ""), 14, C["muted"], a0)

    shades = ["#F2C879", "#EE9B4E", "#DE5F3C", "#9E2B25", "#5B9CF8"]
    for i, sr in enumerate(series):
        a = _fade(u, R.rt(1, 1.2, 0.6) + i * 0.30, 0.5)
        if a <= 0.01:
            continue
        col = _col(R, sr["colour"]) if sr.get("colour") else shades[i % len(shades)]
        pts = sr.get("points", [])
        px = [PX(p[0]) for p in pts]
        py = [PY(p[1]) for p in pts]
        ax.plot(px, py, lw=2.4, color=col, alpha=min(a, 1.0), zorder=6)
        for k, p in enumerate(pts):
            ms = 4 + 1.6 * np.log2(float(p[2]) + 1) if len(p) > 2 else 7
            ax.plot([px[k]], [py[k]], "o", ms=ms, color=col, alpha=min(a, 1.0),
                    zorder=7)
        if sr.get("name"):
            if px[-1] > x + w - 1.7:      # would overflow: label inside instead
                R.txt(ax, px[-1] - 0.18, py[-1], sr["name"], 15, col, a,
                      ha="right")
            else:
                R.txt(ax, px[-1] + 0.18, py[-1], sr["name"], 15, col, a)

    mode = o.get("frontier")
    if mode:
        ap = _fade(u, R.rt(2, 3.4, 0.8), 0.7)
        if ap > 0.01:
            pairs = sorted((float(p[0]), float(p[1])) for p in allp)
            env, best = [], (9e9 if mode == "min" else -9e9)
            for fv, e in pairs:
                if (e < best) if mode == "min" else (e > best):
                    best = e
                    env.append((fv, e))
            g = _ease((u - R.rt(2, 3.4, 0.8)) / 1.4)
            m = max(2, int(len(env) * g))
            ax.plot([PX(p) for p, _ in env[:m]], [PY(q) for _, q in env[:m]],
                    ls=(0, (2, 3)), lw=3.0, color=C["ink"], alpha=min(ap, 1.0),
                    zorder=9)
            R.txt(ax, x + w - 0.5, y + 1.35,
                  o.get("frontier_label", "frontier"), 17, C["ink"], ap,
                  ha="right", weight="bold")
    if o.get("note"):
        R.txt(ax, x + 3.0, y + h - 0.30, o["note"], 17, C["accent"],
              _fade(u, R.rt(3, 5.0, 0.8), 0.7))
    _tail(R, ax, o, u, 4, x, y, w)

def dataflow(R, ax, u, d, o):
    """A small DAG of operations laid out on a grid, with multi-input nodes —
    the shape for "here is exactly what this operator computes".

    opts: nodes [{label, row, col, colour}], edges [[from, to]], title,
          rows, cols, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    nodes = o.get("nodes", [])
    if not nodes:
        return
    rows = int(o.get("rows") or max(n.get("row", 0) for n in nodes) + 1)
    cols = int(o.get("cols") or max(n.get("col", 0) for n in nodes) + 1)
    cw = (w - 1.2) / max(cols, 1)
    rh = (h - 2.3) / max(rows, 1)
    bw, bh = min(cw * 0.78, 3.0), min(rh * 0.56, 0.9)

    def P(nd):
        return (x + 0.6 + (nd.get("col", 0) + 0.5) * cw,
                y + 0.9 + (nd.get("row", 0) + 0.5) * rh)

    pos = [P(nd) for nd in nodes]
    for k, (i, j) in enumerate(o.get("edges", [])):
        beat = max(nodes[j].get("row", 0), 1)
        a = _fade(u, R.rt(beat, 0.8 + beat * 0.9, 0.5), 0.45)
        if a <= 0.01:
            continue
        _arrow(ax, (pos[i][0], pos[i][1] + bh / 2),
               (pos[j][0], pos[j][1] - bh / 2), C["muted"], 1.8, a * 0.85, 12)

    for k, nd in enumerate(nodes):
        beat = nd.get("row", 0)
        a = _fade(u, R.rt(beat, 0.4 + beat * 0.9, 0.5), 0.5)
        if a <= 0.01:
            continue
        col = _col(R, nd.get("colour", "mid"))
        light = col == C["panel"]
        px, py = pos[k]
        R.box(ax, px - bw / 2, py - bh / 2, bw, bh, a=a, fc=col, r=0.10, z=7)
        R.txt(ax, px, py, _tex(str(nd.get("label", ""))), 15,
              C["ink"] if light else "#FFFFFF", a, ha="center", z=9)

    R.txt(ax, x + 0.3, y + h - 0.18, o.get("title", ""), 19, C["muted"],
          _fade(u, R.rt(0, 0.4, 0.6), 0.6))
    _tail(R, ax, o, u, rows, x, y, w)


def density_pair(R, ax, u, d, o):
    """Two 1-D densities that start overlapping and separate — score
    distributions, ID vs OOD, before vs after.

    opts: a_label, b_label, from_gap, to_gap, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    bx, by, bw, bh = x + 0.9, y + 1.3, w - 1.8, h - 2.4
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([bx, bx + bw], [by, by], lw=1.8, color=C["line"],
            alpha=min(a0, 1.0))

    t1 = R.rt(1, 1.6, 0.9)
    prog = _ease(min(1.0, max(0.0, (u - t1) / max(d - t1 - 1.0, 1.5))))
    g0, g1 = float(o.get("from_gap", 0.10)), float(o.get("to_gap", 0.42))
    gap = g0 + (g1 - g0) * prog
    xs = np.linspace(0, 1, 220)
    for sign, col, lab, beat in ((-1, C["pos"], o.get("a_label", ""), 0),
                                 (+1, C["neg"], o.get("b_label", ""), 1)):
        a = _fade(u, R.rt(beat, 0.4 + beat * 1.0, 0.6), 0.6)
        mu = 0.5 + sign * gap
        ys = np.exp(-((xs - mu) ** 2) / (2 * 0.09 ** 2))
        ax.fill_between(bx + xs * bw, by, by + ys * bh * 0.8, color=col,
                        alpha=min(a, 1.0) * 0.35, zorder=5)
        ax.plot(bx + xs * bw, by + ys * bh * 0.8, lw=2.6, color=col,
                alpha=min(a, 1.0), zorder=6)
        R.txt(ax, bx + mu * bw, by + bh * 0.88, lab, 18, col, a, ha="center",
              weight="bold")
    _tail(R, ax, o, u, 2, x, y, w)


def distribution_pair(R, ax, u, d, o):
    """Eq. 7: the one-hot label is replaced by CLIP's own soft prediction."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    names = o.get("names", ["cat", "dog", "bus", "neg-1", "neg-2"])
    hard = [1, 0, 0, 0, 0]
    soft = o.get("soft", [0.62, 0.19, 0.09, 0.06, 0.04])
    bw = (w - 1.2) / len(names)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, x + 0.2, y + h - 0.2, o.get("top_title", ""),
          17, C["muted"], a0)
    for k, v in enumerate(hard):
        ax.add_patch(Rectangle((x + 0.3 + k * bw, y + h - 2.5), bw * 0.62,
                               1.9 * v, fc=C["dim"], ec="none",
                               alpha=min(a0, 1.0) * 0.7, zorder=6))
    if _fade(u, R.rt(1, 1.6, 0.7), 0.5) > 0.4:
        ax.plot([x + 0.3, x + w - 0.6], [y + h - 1.5, y + h - 1.5], lw=3,
                color=C["neg"], alpha=0.8, zorder=9)

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.6)
    R.txt(ax, x + 0.2, y + 2.35, o.get("bottom_title", ""),
          17, C["muted"], a2)
    for k, (nm, v) in enumerate(zip(names, soft)):
        g = _ease((u - R.rt(2, 2.8, 0.7) - k * 0.12) / 0.8)
        ax.add_patch(Rectangle((x + 0.3 + k * bw, y + 0.55), bw * 0.62,
                               2.9 * v * max(g, 0), fc=C["pos"], ec="none",
                               alpha=min(a2, 1.0), zorder=6))
        R.txt(ax, x + 0.3 + k * bw + bw * 0.31, y + 0.25, nm, 14, C["muted"],
              a2, ha="center")


def diverging_bars(R, ax, u, d, o):
    """Figure 4: same compute, reasoning benchmarks up, OCR benchmarks down."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    names = o.get("labels", ["GQA","MMB","MME","MMMU","POPE","SQA","DocVQA","TextVQA"])
    vals = o.get("values", [5.5, 34, 22, 19.5, -2, 33.5, -11, -6])
    zero = y + h * 0.52
    bw = (w - 1.4) / len(names)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([x + 0.6, x + w - 0.5], [zero, zero], color=C["line"], lw=1.6,
            alpha=min(a0, 1.0))
    R.txt(ax, x + 0.6, y + h - 0.25, o.get("caption", ""),
          18, C["muted"], a0)

    for k, (nm, v) in enumerate(zip(names, vals)):
        ocr = k >= 6
        t0 = R.rt(2 if ocr else 1, 1.2, 0.5) + (k if not ocr else k - 6) * 0.18
        a = _fade(u, t0, 0.45)
        if a <= 0.01:
            continue
        g = _ease((u - t0) / 0.7)
        bx = x + 0.75 + k * bw
        hh = (h * 0.36) * (v / 34.0) * g
        col = C["pos"] if v > 0 else C["neg"]
        ax.add_patch(Rectangle((bx, zero if v > 0 else zero + hh),
                               bw * 0.66, abs(hh), fc=col, ec="none",
                               alpha=min(a, 1.0), zorder=6))
        R.txt(ax, bx + bw * 0.33, zero - 0.30 if v > 0 else zero + 0.28, nm, 14,
              C["muted"], a, ha="center")

    a1 = _fade(u, R.rt(1, 1.2, 0.5) + 1.4, 0.6)
    R.txt(ax, x + 0.75 + 3 * bw, y + h - 0.85, o.get("group_a", ""),
          17, C["pos"], a1, ha="center", weight="bold")
    a2 = _fade(u, R.rt(2, 3.4, 0.8) + 0.6, 0.6)
    R.txt(ax, x + 0.75 + 7 * bw, y + 0.55, o.get("group_b", ""), 17,
          C["neg"], a2, ha="center", weight="bold")


def flow_split(R, ax, u, d, o):
    """One input fanning out to several branches, optionally with one chosen —
    routing, mixture of experts, beam search, ablation trees.

    opts: source, branches [{label, colour, weight, chosen}], sink, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    br = o.get("branches", [])
    n = max(len(br), 1)
    sx, cy = x + 1.0, y + h / 2

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, sx, cy - 0.55, 2.2, 1.1, a=a0, fc=_col(R, "in"), r=0.12, z=7)
    R.txt(ax, sx + 1.1, cy, o.get("source", ""), 16, "#FFFFFF", a0,
          ha="center", z=9)

    bx = x + w * 0.52
    span = h - 1.6
    for k, b in enumerate(br):
        a = _fade(u, R.rt(min(k + 1, 3), 1.2 + k * 0.5, 0.5), 0.5)
        if a <= 0.01:
            continue
        by = cy + span / 2 - k * (span / max(n - 1, 1))
        chosen = bool(b.get("chosen"))
        col = _col(R, b.get("colour", "hot" if chosen else "dim"))
        lw = 1.6 + 2.6 * float(b.get("weight", 0.4))
        _arrow(ax, (sx + 2.3, cy), (bx - 0.08, by), col, lw,
               a * (1.0 if chosen else 0.55), 13)
        R.box(ax, bx, by - 0.42, 2.9, 0.84, a=a * (1.0 if chosen else 0.7),
              fc=col, r=0.12, z=7)
        R.txt(ax, bx + 1.45, by, str(b.get("label", "")), 15, "#FFFFFF", a,
              ha="center", z=9)
        if o.get("sink") and chosen:
            _arrow(ax, (bx + 3.0, by), (x + w - 1.9, cy), C["accent"], 2.4, a, 14)

    if o.get("sink"):
        a2 = _fade(u, R.rt(3, 3.4, 0.8), 0.6)
        R.box(ax, x + w - 1.8, cy - 0.55, 1.6, 1.1, a=a2, fc=_col(R, "hot"),
              r=0.12, z=7)
        R.txt(ax, x + w - 1.0, cy, o["sink"], 15, "#FFFFFF", a2, ha="center", z=9)
    _tail(R, ax, o, u, n + 1, x, y, w)


def grid_collapse(R, ax, u, d, o):
    """The sqrt(n) x sqrt(n) patch grid collapsing s x s regions into one token."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n, s = int(o.get("n", 12)), int(o.get("s", 4))
    cell = min((w - 4.6) / n, (h - 2.2) / n)
    gx, gy = x + 0.5, y + h - 1.0

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((gx + j * cell, gy - (i + 1) * cell),
                                   cell * 0.88, cell * 0.88, fc=P["vis"],
                                   ec="none", alpha=min(a0, 1.0) * 0.45,
                                   zorder=5))
    R.txt(ax, gx + n * cell / 2, gy + 0.35, o.get("grid_label", ""),
          18, P["vis"], a0, ha="center")

    a1 = _fade(u, R.rt(1, 1.6, 0.7), 0.6)
    for bi in range(n // s):
        for bj in range(n // s):
            ax.add_patch(Rectangle((gx + bj * s * cell, gy - (bi + 1) * s * cell),
                                   s * cell * 0.94, s * cell * 0.94, fill=False,
                                   ec=C["accent"], lw=2.2, alpha=min(a1, 1.0),
                                   zorder=7))
    R.txt(ax, gx, y + 0.35, o.get("region_label", ""), 18,
          C["accent"], a1)

    a2 = _fade(u, R.rt(2, 3.0, 0.7), 0.6)
    ox = gx + n * cell + 1.5
    oy = gy - 1.4
    for bi in range(n // s):
        for bj in range(n // s):
            _arrow(ax, (gx + (bj + 0.5) * s * cell, gy - (bi + 0.5) * s * cell),
                   (ox + bj * 0.62, oy - bi * 0.62), C["accent"], 1.4,
                   a2 * 0.35, 9)
            ax.add_patch(Rectangle((ox + bj * 0.62, oy - bi * 0.62), 0.5, 0.5,
                                   fc=C["accent"], ec="none",
                                   alpha=min(a2, 1.0), zorder=8))
    R.txt(ax, ox + 0.9, oy + 0.85, o.get("out_label", ""), 18, C["accent"], a2, ha="center",
          weight="bold")

    a3 = _fade(u, R.rt(3, 4.2, 0.8), 0.7)
    R.txt(ax, ox - 0.3, y + 0.95,
          o.get("note", ""), 17, C["muted"], a3)


def growth_curve(R, ax, u, d, o):
    C, P = R.c, _pal(R)
    x0, y0, w, h = 3.4, 2.4, 9.2, 4.6
    a = _fade(u, R.rt(0, 0.4, 0.7), 0.6)
    ax.plot([x0, x0], [y0, y0 + h], color=P["grid"], lw=1.8, alpha=min(a, 1.0))
    ax.plot([x0, x0 + w], [y0, y0], color=P["grid"], lw=1.8, alpha=min(a, 1.0))
    R.txt(ax, x0 + w / 2, y0 - 0.5, o.get("xlabel", "steps"), 18,
          C["muted"], a, ha="center")
    R.txt(ax, x0 - 0.5, y0 + h / 2, o.get("ylabel", "cumulative"), 18, C["muted"], a,
          ha="center", va="center", rot=90)

    t1 = R.rt(1, 1.2, 1.2)
    prog = _ease(min(1.0, max(0.0, (u - t1) / max(d - t1 - 1.0, 1.5))))
    m = max(2, int(240 * prog))
    xs = np.linspace(0, 1, 240)[:m]
    ax.plot(x0 + xs * w, y0 + xs * h * 0.94, ls=(0, (5, 4)), lw=2.2,
            color=C["muted"], alpha=min(a, 1.0) * 0.55, zorder=6)
    ax.plot(x0 + xs * w, y0 + np.sqrt(xs) * h * 0.70, lw=3.6, color=C["accent"],
            alpha=min(a, 1.0), zorder=7)

    at = _fade(u, R.rt(2, 3.0, 1.2), 0.6)
    R.txt(ax, x0 + w * 0.60, y0 + h * 0.92, o.get("ref_label", "linear"), 16,
          C["muted"], at)
    R.txt(ax, x0 + w * 0.50, y0 + h * 0.56, o.get("curve_label", ""), 24,
          C["accent"], at, weight="bold")
    R.txt(ax, x0 + w * 0.50, y0 + h * 0.40,
          o.get("note", ""), 16, C["muted"], at)




# --------------------------------------------------------------------------- #
# box-aware figures — these sit beside an equation as well as fill the frame
# --------------------------------------------------------------------------- #


def hierarchy(R, ax, u, d, o):
    """A two- or three-level tree — taxonomies, ablation structure, method
    families.

    opts: root, levels [[label | [label, colour]], ...], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    levels = o.get("levels", [])
    rows = 1 + len(levels)
    dy = (h - 1.2) / max(rows, 1)
    ry = y + h - 0.7

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, x + w / 2 - 1.7, ry - 0.42, 3.4, 0.84, a=a0, fc=_col(R, "hot"),
          r=0.12, z=7)
    R.txt(ax, x + w / 2, ry, o.get("root", ""), 17, "#FFFFFF", a0, ha="center",
          z=9)

    prev = [(x + w / 2, ry)]
    for li, row in enumerate(levels):
        a = _fade(u, R.rt(li + 1, 1.2 + li * 1.2, 0.7), 0.6)
        yy = ry - (li + 1) * dy
        m = max(len(row), 1)
        pts = []
        for k, node in enumerate(row):
            lab = node[0] if isinstance(node, (list, tuple)) else node
            col = _col(R, node[1] if isinstance(node, (list, tuple))
                       and len(node) > 1 else "mid")
            px = x + (w / (m + 1)) * (k + 1)
            bw = min(w / (m + 1) * 0.86, 3.2)
            light = col == C["panel"]
            R.box(ax, px - bw / 2, yy - 0.38, bw, 0.76, a=a, fc=col, r=0.11, z=7)
            R.txt(ax, px, yy, str(lab), 15, C["ink"] if light else "#FFFFFF", a,
                  ha="center", z=9)
            parent = prev[min(k * len(prev) // m, len(prev) - 1)]
            ax.plot([parent[0], px], [parent[1] - 0.42, yy + 0.38], lw=1.8,
                    color=C["line"], alpha=min(a, 1.0) * 0.9, zorder=5)
            pts.append((px, yy))
        prev = pts
    _tail(R, ax, o, u, len(levels) + 1, x, y, w)


def manifold_step(R, ax, u, d, o):
    """Eq. 8: a gradient step leaves the sphere; l2 renormalising is the
    projection that puts it back. That is what makes Theorem 3 apply."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    cx, cy, rad = x + w / 2, y + h / 2, min(w, h) * 0.36

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.add_patch(Circle((cx, cy), rad, fill=False, ec=P["grid"], lw=2.0,
                        alpha=min(a0, 1.0), zorder=4))
    R.txt(ax, cx, cy + rad + 0.42, o.get("set_label", ""), 22, C["muted"], a0,
          ha="center")

    th = 0.62
    p0 = (cx + rad * np.cos(th), cy + rad * np.sin(th))
    ax.plot([p0[0]], [p0[1]], "o", ms=13, color=P["txt"],
            alpha=min(a0, 1.0), zorder=8)
    R.txt(ax, p0[0] + 0.30, p0[1] + 0.22, o.get("start_label", ""), 19, P["txt"],
          a0)

    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    # step along the tangent so the result is visibly off the sphere
    tang = np.array([-np.sin(th), np.cos(th)])
    p1 = (p0[0] + tang[0] * rad * 1.05, p0[1] + tang[1] * rad * 1.05)
    _arrow(ax, p0, p1, C["neg"], 2.6, a1)
    R.txt(ax, (p0[0] + p1[0]) / 2 + 0.15, (p0[1] + p1[1]) / 2 + 0.42,
          o.get("step_label", ""), 19, C["neg"], a1, ha="center")
    ax.plot([p1[0]], [p1[1]], "o", ms=11, color=C["neg"], alpha=min(a1, 1.) * .9,
            zorder=8)

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.6)
    v = np.array([p1[0] - cx, p1[1] - cy])
    v = v / np.linalg.norm(v) * rad
    p2 = (cx + v[0], cy + v[1])
    if a2 > 0.01:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], ls=(0, (3, 3)), lw=2.0,
                color=C["accent"], alpha=min(a2, 1.0), zorder=7)
        ax.plot([p2[0]], [p2[1]], "o", ms=13, color=P["vis"],
                alpha=min(a2, 1.0), zorder=9)
        R.txt(ax, p2[0] - 0.34, p2[1] + 0.10, o.get("end_label", ""), 19,
              P["vis"], a2, ha="right")
    a3 = _fade(u, R.rt(3, 4.0, 0.8), 0.7)
    R.txt(ax, x + 0.2, y + 0.30,
          o.get("verdict", ""), 18,
          C["accent"], a3, weight="bold")


def matrix_grid(R, ax, u, d, o):
    """A matrix or heatmap with optional highlighted rows, columns or blocks —
    attention maps, confusion matrices, masks.

    opts: rows, cols, values [[..]], row_label, col_label,
          highlight {row, col, block [r0,c0,r1,c1]}, cmap_colour, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rows, cols = int(o.get("rows", 8)), int(o.get("cols", 8))
    vals = o.get("values")
    cell = min((w - 4.0) / cols, (h - 2.0) / rows)
    gx = x + 1.6
    gy = y + h - 0.9
    base = _col(R, o.get("cmap_colour", "in"))

    rng = np.random.default_rng(int(o.get("seed", 3)))
    for i in range(rows):
        for j in range(cols):
            a = _fade(u, R.rt(0, 0.4, 0.6) + (i + j) * 0.012, 0.35)
            v = (float(vals[i][j]) if vals else
                 float(rng.uniform(0.12, 0.95)))
            ax.add_patch(Rectangle((gx + j * cell, gy - (i + 1) * cell),
                                   cell * 0.9, cell * 0.9, fc=base, ec="none",
                                   alpha=min(a, 1.0) * v, zorder=5))
    R.txt(ax, gx + cols * cell / 2, gy + 0.32, o.get("col_label", ""), 17,
          C["muted"], _fade(u, R.rt(0, 0.4, 0.6), 0.6), ha="center")
    R.txt(ax, gx - 0.35, gy - rows * cell / 2, o.get("row_label", ""), 17,
          C["muted"], _fade(u, R.rt(0, 0.4, 0.6), 0.6), ha="center",
          va="center", rot=90)

    hl = o.get("highlight", {})
    a1 = _fade(u, R.rt(1, 1.8, 0.7), 0.6)
    if a1 > 0.01 and hl:
        if hl.get("row") is not None:
            i = int(hl["row"])
            ax.add_patch(Rectangle((gx, gy - (i + 1) * cell), cols * cell,
                                   cell * 0.9, fill=False, ec=C["accent"],
                                   lw=2.6, alpha=min(a1, 1.0), zorder=8))
        if hl.get("col") is not None:
            j = int(hl["col"])
            ax.add_patch(Rectangle((gx + j * cell, gy - rows * cell),
                                   cell * 0.9, rows * cell, fill=False,
                                   ec=C["accent"], lw=2.6, alpha=min(a1, 1.0),
                                   zorder=8))
        if hl.get("block"):
            r0, c0, r1, c1 = hl["block"]
            ax.add_patch(Rectangle((gx + c0 * cell, gy - (r1 + 1) * cell),
                                   (c1 - c0 + 1) * cell, (r1 - r0 + 1) * cell,
                                   fill=False, ec=C["accent"], lw=2.6,
                                   alpha=min(a1, 1.0), zorder=8))
    _tail(R, ax, o, u, 2, x, y, w)


def panels(R, ax, u, d, o):
    """Two or three side-by-side panels, each a title, some labelled bars and a
    verdict. The shape for any "regime A versus regime B" comparison.

    A panel is either a dict
        {"title": str, "bars": [[label, value, colour], ...], "verdict": str}
    or the older compact form
        [title, value_a, value_b, verdict, beat]
    which is read as two bars labelled alpha and beta.

    opts: panels [...], fmt, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    raw = o.get("panels", [
        ["visual reasoning", ALPHA, BETA, "1 token is enough", 0],
        ["OCR / documents", 0.029, 0.048, "tokens dominate", 1]])

    def _norm(p, i):
        if isinstance(p, dict):
            return (p.get("title", ""),
                    [(str(b[0]), float(b[1]),
                      _col(R, b[2]) if len(b) > 2 else (C["accent"] if j == 0
                                                       else P["vis"]))
                     for j, b in enumerate(p.get("bars", []))],
                    p.get("verdict", ""), int(p.get("beat", i)))
        p = tuple(p)
        return (p[0], [(r"$\alpha$", float(p[1]), C["accent"]),
                       (r"$\beta$", float(p[2]), P["vis"])],
                p[3] if len(p) > 3 else "", int(p[4]) if len(p) > 4 else i)

    ps = [_norm(p, i) for i, p in enumerate(raw)]
    n = max(len(ps), 1)
    pw = (w - 0.7 * (n - 1)) / n
    fmt = o.get("fmt", "{:.3f}")

    for i, (title, bars, verdict, beat) in enumerate(ps):
        t0 = R.rt(beat, 0.5 + i * 1.5, 0.7)
        a = _fade(u, t0, 0.6)
        if a <= 0.01:
            continue
        bx = x + i * (pw + 0.7)
        R.box(ax, bx, y + 0.9, pw, h - 1.4, a=a * 0.55, fc=C["panel"], r=0.14,
              z=3)
        R.txt(ax, bx + pw / 2, y + h - 0.85, title, 19, C["ink"], a,
              ha="center", weight="bold")
        mx = (max(v for _, v, _ in bars) * 1.2) if bars else 1.0
        for j, (nm, v, col) in enumerate(bars):
            yy = y + h - 1.75 - j * 0.95
            g = max(_ease((u - t0 - j * 0.25) / 0.8), 0.0)
            ax.add_patch(Rectangle((bx + 0.85, yy - 0.24),
                                   (pw - 1.9) * v / mx * g, 0.48, fc=col,
                                   ec="none", alpha=min(a, 1.0), zorder=6))
            R.txt(ax, bx + 0.70, yy, nm, 17, col, a, ha="right")
            R.txt(ax, bx + 0.85 + (pw - 1.9) * v / mx * g + 0.15, yy,
                  fmt.format(v), 15, C["muted"], a * g)
        av = _fade(u, t0 + 1.0, 0.6)
        R.txt(ax, bx + pw / 2, y + 1.35, verdict, 18,
              C["accent"] if i == 0 else C["neg"], av, ha="center",
              weight="bold")

    a3 = _fade(u, R.rt(n + 1, 4.2, 0.9), 0.7)
    R.txt(ax, x + w / 2, y + 0.28, o.get("verdict", ""), 20, C["accent"], a3,
          ha="center", weight="bold")


def parallel_heads(R, ax, u, d, o):
    """One input projected into h parallel paths that run independently and are
    then concatenated and projected back — multi-head anything, ensembles,
    grouped convolutions.

    opts: input_label, heads, head_label, concat_label, output_label,
          per_head_labels, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    hn = int(o.get("heads", 8))
    cy = y + h * 0.60
    lane = min((h - 2.4) / hn, 0.62)
    hx = x + w * 0.36
    hw = w * 0.24

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, x + 0.5, cy - 0.62, 2.1, 1.24, a=a0, fc=_col(R, "in"), r=0.12, z=7)
    R.txt(ax, x + 1.55, cy, o.get("input_label", ""), 16, "#FFFFFF", a0,
          ha="center", z=9)

    labels = o.get("per_head_labels", [])
    for k in range(hn):
        a = _fade(u, R.rt(1, 1.3, 0.6) + k * 0.10, 0.45)
        if a <= 0.01:
            continue
        yy = cy + (hn - 1) / 2 * lane - k * lane
        _arrow(ax, (x + 2.65, cy), (hx - 0.06, yy), C["muted"], 1.5, a * 0.7, 10)
        R.box(ax, hx, yy - lane * 0.38, hw, lane * 0.76, a=a,
              fc=_col(R, "hot"), r=0.09, z=7)
        lab = labels[k] if k < len(labels) else o.get("head_label", "")
        if lab:
            R.txt(ax, hx + hw / 2, yy, str(lab), 12, "#FFFFFF", a, ha="center",
                  z=9)
        _arrow(ax, (hx + hw + 0.06, yy), (x + w * 0.70, cy), C["muted"], 1.5,
               a * 0.7, 10)
    R.txt(ax, hx + hw / 2, cy + (hn / 2) * lane + 0.45,
          o.get("heads_label", f"{hn} heads"), 18, _col(R, "hot"),
          _fade(u, R.rt(1, 1.3, 0.6), 0.6), ha="center", weight="bold")

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.6)
    R.box(ax, x + w * 0.70, cy - 0.62, 1.5, 1.24, a=a2, fc=C["panel"], r=0.12, z=7)
    R.txt(ax, x + w * 0.70 + 0.75, cy, o.get("concat_label", "concat"), 14,
          C["ink"], a2, ha="center", z=9)
    a3 = _fade(u, R.rt(3, 3.8, 0.7), 0.6)
    _arrow(ax, (x + w * 0.70 + 1.55, cy), (x + w - 2.3, cy), C["muted"], 2.0, a3)
    R.box(ax, x + w - 2.2, cy - 0.62, 1.9, 1.24, a=a3, fc=_col(R, "out"),
          r=0.12, z=7)
    R.txt(ax, x + w - 1.25, cy, o.get("output_label", ""), 15, "#FFFFFF", a3,
          ha="center", z=9)
    _tail(R, ax, o, u, 4, x, y, w)


def pipeline(R, ax, u, d, o):
    """A left-to-right flow of labelled boxes — any architecture diagram.

    opts: stages ["label" | ["label", colour] | ["label", colour, beat]],
          per_beat (stages per narration beat when no beat is given),
          verdict, note
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cy = y + h * 0.60
    bw, bh = (w - 1.6) / 5.2, 1.25
    per = float(o.get("per_beat", 2.0))

    def _stage(t):
        """Accept "label", ["label"], ["label", colour] or
        ["label", colour, beat]; missing fields fall back."""
        t = tuple(t) if isinstance(t, (list, tuple)) else (t,)
        lab = t[0]
        col = _col(R, t[1]) if len(t) > 1 and isinstance(t[1], str) else C["panel"]
        beat = next((v for v in t[1:] if isinstance(v, (int, float))), None)
        return (lab, col, beat)

    stages = [_stage(t) for t in o.get("stages", [
        ["text\nquery", P["txt"], 0], ["LLM\nhidden state", C["panel"], 0],
        ["linear\nprojection", C["panel"], 1], ["+ visual\ntokens", P["vis"], 1],
        ["conv\ndownsample", C["accent"], 2], ["cross-\nattention", C["accent"], 2],
        ["MLP", C["panel"], 3]])]
    n = len(stages)
    sw = (w - 0.8) / n
    for k, (lab, col, beat) in enumerate(stages):
        if beat is None:                      # no explicit beat: pace by `per`
            beat = int(k / max(per, 1))
        a = _fade(u, R.rt(int(beat), 0.5 + k * 0.45, 0.5), 0.45)
        if a <= 0.01:
            continue
        bx = x + 0.35 + k * sw
        light = col in (C["panel"],)
        R.box(ax, bx, cy - bh / 2, sw * 0.82, bh, a=a, fc=col, r=0.12, z=6)
        R.txt(ax, bx + sw * 0.41, cy, lab, 14,
              C["ink"] if light else "#FFFFFF", a, ha="center", z=9)
        if k:
            _arrow(ax, (bx - sw * 0.16, cy), (bx - 0.03, cy), C["muted"], 1.8, a)

    a4 = _fade(u, R.rt(3, 3.6, 0.8), 0.7)
    R.txt(ax, x + 0.35, cy - bh / 2 - 0.75,
          o.get("verdict", ""), 20, C["accent"], a4,
          weight="bold")
    a5 = _fade(u, R.rt(4, 4.8, 0.8), 0.7)
    R.txt(ax, x + 0.35, cy - bh / 2 - 1.30,
          o.get("note", ""), 18, C["muted"], a5)


def ratio_slider(R, ax, u, d, o):
    """As the text prompt grows, the optimal visual token count rises."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    qs = [0, 10, 25, 50, 100]
    vstar = [1, 4, 9, 16, 36]

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, x + 0.2, y + h - 0.2, o.get("a_label", ""), 18, P["txt"], a0)
    R.txt(ax, x + 0.2, y + h - 0.75, o.get("b_label", ""), 18,
          P["vis"], a0)

    t1 = R.rt(1, 1.4, 0.8)
    prog = _ease(min(1.0, max(0.0, (u - t1) / max(d - t1 - 1.4, 1.5))))
    idx = min(int(prog * (len(qs) - 0.001)), len(qs) - 1)
    frac = prog * (len(qs) - 1) - idx
    qv = qs[idx] + (qs[min(idx + 1, len(qs) - 1)] - qs[idx]) * frac
    vv = vstar[idx] + (vstar[min(idx + 1, len(vstar) - 1)] - vstar[idx]) * frac

    a1 = _fade(u, t1, 0.6)
    bx, bw = x + 0.4, w - 1.0
    yy = y + h * 0.55
    ax.add_patch(Rectangle((bx, yy), bw * qv / 130.0, 0.6, fc=P["txt"],
                           ec="none", alpha=min(a1, 1.0) * 0.9, zorder=6))
    ax.add_patch(Rectangle((bx + bw * qv / 130.0, yy), bw * vv / 130.0, 0.6,
                           fc=P["vis"], ec="none", alpha=min(a1, 1.0), zorder=6))
    R.txt(ax, bx, yy - 0.55, f"{o.get('a_short','A')} = {qv:.0f}", 20, P["txt"], a1, weight="bold")
    R.txt(ax, bx + bw * 0.42, yy - 0.55, f"{o.get('b_short','B')} = {vv:.0f}", 20, P["vis"], a1,
          weight="bold")

    a2 = _fade(u, R.rt(2, 3.2, 0.9), 0.7)
    R.txt(ax, x + 0.2, y + 0.85,
          o.get("note_a", ""), 18, C["muted"], a2)
    R.txt(ax, x + 0.2, y + 0.35,
          o.get("note_b", ""), 19, C["accent"], a2,
          weight="bold")


def scale_ladder(R, ax, u, d, o):
    """Model or dataset sizes as stacked blocks, one per beat — the visual for
    "we scaled this up".

    opts: rungs [{label, size, note, colour}], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rungs = o.get("rungs", [])
    n = max(len(rungs), 1)
    mx = max(float(r.get("size", 1)) for r in rungs) if rungs else 1.0
    bw = (w - 1.6) / n

    for k, r_ in enumerate(rungs):
        a = _fade(u, R.rt(k, 0.5 + k * 0.8, 0.6), 0.5)
        if a <= 0.01:
            continue
        g = _ease((u - R.rt(k, 0.5 + k * 0.8, 0.6)) / 0.8)
        hh = (h - 2.0) * float(r_.get("size", 1)) / mx * g
        px = x + 0.8 + k * bw
        R.box(ax, px, y + 1.2, bw * 0.72, max(hh, 0.05), a=a,
              fc=_col(R, r_.get("colour", "hot" if k == n - 1 else "dim")),
              r=0.10, z=6)
        R.txt(ax, px + bw * 0.36, y + 0.82, str(r_.get("label", "")), 17,
              C["ink"] if k == n - 1 else C["muted"], a, ha="center",
              weight="bold" if k == n - 1 else "normal")
        if r_.get("note"):
            R.txt(ax, px + bw * 0.36, y + 1.35 + hh, r_["note"], 15,
                  C["muted"], a * g, ha="center")
    _tail(R, ax, o, u, n, x, y, w)


# --------------------------------------------------------------------------- #
# data and geometry
# --------------------------------------------------------------------------- #


def scatter_boundary(R, ax, u, d, o):
    """Two classes of points with a decision boundary that appears and then
    improves — classifiers, margins, separability arguments.

    opts: a_label, b_label, boundary ("line"|"curve"), bad_first, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cx, cy = x + w / 2, y + h / 2 + 0.2
    rad = min(w * 0.20, h * 0.36)
    rng = np.random.default_rng(int(o.get("seed", 9)))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    A = [(cx - rad * 0.55 + rng.normal(0, rad * .26),
          cy + rad * 0.42 + rng.normal(0, rad * .26)) for _ in range(26)]
    B = [(cx + rad * 0.60 + rng.normal(0, rad * .28),
          cy - rad * 0.45 + rng.normal(0, rad * .28)) for _ in range(26)]
    for px, py in A:
        ax.plot([px], [py], "o", ms=9, color=C["pos"], alpha=min(a0, 1.) * .85,
                zorder=6)
    a1 = _fade(u, R.rt(1, 1.4, 0.6), 0.6)
    for px, py in B:
        ax.plot([px], [py], "o", ms=9, color=C["neg"], alpha=min(a1, 1.) * .85,
                zorder=6)

    t2 = R.rt(2, 2.6, 0.8)
    a2 = _fade(u, t2, 0.6)
    if a2 > 0.01:
        prog = _ease(min(1.0, max(0.0, (u - t2) / max(d - t2 - 1.0, 1.5))))
        ang0, ang1 = float(o.get("from_angle", -0.95)), float(o.get("to_angle", -0.55))
        th = ang0 + (ang1 - ang0) * (prog if o.get("bad_first", True) else 1.0)
        L = rad * 1.5
        ax.plot([cx - L * np.cos(th), cx + L * np.cos(th)],
                [cy - L * np.sin(th), cy + L * np.sin(th)], lw=3.2,
                color=C["accent"], alpha=min(a2, 1.0), zorder=8)
    R.txt(ax, cx - rad * 1.15, cy + rad * 1.0, o.get("a_label", ""), 18,
          C["pos"], a0, ha="right")
    R.txt(ax, cx + rad * 1.15, cy - rad * 1.05, o.get("b_label", ""), 18,
          C["neg"], a1)
    _tail(R, ax, o, u, 3, x, y, w)


def sequence_tokens(R, ax, u, d, o):
    """A token sequence with a mask or attention span sweeping along it —
    causal masking, context windows, sliding attention.

    opts: tokens [str], window, causal, note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    toks = o.get("tokens", [])
    n = max(len(toks), 1)
    cw = min((w - 1.6) / n, 1.5)
    bx = x + (w - cw * n) / 2
    cy = y + h * 0.60

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    t1 = R.rt(1, 1.6, 0.8)
    cur = int(min(n - 1, max(0, (u - t1) / max(float(o.get("rate", 0.7)), 0.05))))
    win = int(o.get("window", 0))

    for k, t in enumerate(toks):
        active = (u > t1) and (k <= cur) and (not win or k > cur - win)
        col = _col(R, "hot") if (u > t1 and k == cur) else (
            _col(R, "in") if active else C["dim"])
        a = a0 if u <= t1 else 1.0
        R.box(ax, bx + k * cw, cy - 0.45, cw * 0.88, 0.9, a=a * (1 if active or u <= t1 else 0.35),
              fc=col, r=0.10, z=6)
        R.txt(ax, bx + k * cw + cw * 0.44, cy, str(t), 14, "#FFFFFF",
              a * (1 if active or u <= t1 else 0.5), ha="center", z=9)

    a2 = _fade(u, R.rt(2, 3.2, 0.8), 0.6)
    R.txt(ax, x + 0.3, cy - 1.25, o.get("note", ""), 18, C["muted"], a2)
    _tail(R, ax, o, u, 3, x, y, w)


def split_bar(R, ax, u, d, o):
    """Theorem 2: the distance splits into a closable part and a floor."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    bw, bx = w - 1.0, x + 0.3
    frac = float(o.get("orthogonal", 0.42))
    yy = y + h - 2.2

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, bx, yy + 1.05, o.get("title", ""), 22,
          C["ink"], a0)
    ax.add_patch(Rectangle((bx, yy), bw, 0.8, fc=C["dim"], ec="none",
                           alpha=min(a0, 1.0) * 0.5, zorder=5))

    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    g = _ease((u - R.rt(1, 1.4, 0.7)) / 0.9)
    ax.add_patch(Rectangle((bx, yy), bw * (1 - frac) * g, 0.8, fc=P["vis"],
                           ec="none", alpha=min(a1, 1.0), zorder=6))
    R.txt(ax, bx, yy - 0.65, o.get("a_label", ""), 17, P["vis"], a1)

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.6)
    g2 = _ease((u - R.rt(2, 2.8, 0.7)) / 0.9)
    ax.add_patch(Rectangle((bx + bw * (1 - frac), yy), bw * frac * g2, 0.8,
                           fc=C["neg"], ec="none", alpha=min(a2, 1.0), zorder=6))
    R.txt(ax, bx, yy - 1.20, o.get("b_label", ""), 17, C["neg"],
          a2)

    a3 = _fade(u, R.rt(3, 4.0, 0.8), 0.7)
    R.txt(ax, bx, y + 0.40,
          o.get("verdict", ""), 19, C["accent"], a3,
          weight="bold")


def stacked_bar(R, ax, u, d, o):
    """NegLabel: what fraction of the probability mass stays on the ID side."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    frac_id = float(o.get("id_fraction", 0.72))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.5)
    R.txt(ax, x, y + h - 0.2, o.get("total_label", ""), 18, C["muted"], a0)
    bx, by, bw = x, y + h - 1.5, w - 0.4
    ax.add_patch(Rectangle((bx, by), bw, 0.62, fc=C["dim"], ec="none",
                           alpha=min(a0, 1.0) * 0.55, zorder=5))

    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    g = _ease((u - R.rt(1, 1.4, 0.7)) / 0.9)
    ax.add_patch(Rectangle((bx, by - 1.5), bw * frac_id * g, 0.62, fc=C["pos"],
                           ec="none", alpha=min(a1, 1.0), zorder=6))
    R.txt(ax, bx, by - 1.95, o.get("a_label", "{pct}%").format(pct=int(frac_id*100)), 17,
          C["pos"], a1)

    a2 = _fade(u, R.rt(2, 2.6, 0.7), 0.6)
    g2 = _ease((u - R.rt(2, 2.6, 0.7)) / 0.9)
    ax.add_patch(Rectangle((bx + bw * frac_id, by - 2.9),
                           bw * (1 - frac_id) * g2, 0.62, fc=C["neg"],
                           ec="none", alpha=min(a2, 1.0), zorder=6))
    R.txt(ax, bx, by - 3.35, o.get("b_label", ""), 17,
          C["neg"], a2)

    a3 = _fade(u, R.rt(3, 3.8, 0.8), 0.6)
    R.txt(ax, x, y + 0.30, o.get("verdict", ""), 19,
          C["accent"], a3, weight="bold")


def stacked_blocks(R, ax, u, d, o):
    """A repeated block with internal sublayers and residual bypasses, drawn
    with ghost copies behind it to show the stack depth.

    opts: title, repeat_label, sublayers [{label, colour, residual}],
          depth (ghost copies), input_label, output_label, verdict, side
    """
    C = R.c
    x, y, w, h = _box(o)
    subs = o.get("sublayers", [])
    n = max(len(subs), 1)
    bw = min(w * 0.52, 7.4)
    bx = x + (w - bw) / 2
    top = y + h - 0.95
    sh = min(1.05, (h - 3.4) / n)
    bh = n * sh + 0.55

    # ghost copies behind, to read as a stack
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for g in range(int(o.get("depth", 2)), 0, -1):
        ax.add_patch(FancyBboxPatch((bx + g * 0.22, top - bh - g * 0.22), bw, bh,
                                    boxstyle="round,pad=0,rounding_size=0.16",
                                    fc=C["panel"], ec=C["line"], lw=1.2,
                                    alpha=min(a0, 1.0) * 0.45, zorder=3))
    ax.add_patch(FancyBboxPatch((bx, top - bh), bw, bh,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                fc=C["panel"], ec=C["line"], lw=1.6,
                                alpha=min(a0, 1.0) * 0.9, zorder=4))
    R.txt(ax, bx + bw / 2, top + 0.35, o.get("title", ""), 20, C["ink"], a0,
          ha="center", weight="bold")
    if o.get("repeat_label"):
        R.txt(ax, bx + bw + 0.55, top - bh / 2, o["repeat_label"], 22,
              C["accent"], a0, weight="bold")

    for k, sl in enumerate(subs):
        a = _fade(u, R.rt(k + 1, 1.2 + k * 1.0, 0.6), 0.55)
        if a <= 0.01:
            continue
        yy = top - 0.55 - (k + 1) * sh + sh * 0.5
        col = _col(R, sl.get("colour", "hot"))
        R.box(ax, bx + 1.25, yy - sh * 0.34, bw - 2.5, sh * 0.68, a=a, fc=col,
              r=0.11, z=7)
        R.txt(ax, bx + bw / 2 + 0.35, yy, str(sl.get("label", "")), 16,
              "#FFFFFF", a, ha="center", z=9)
        if sl.get("residual", True):
            lx = bx + 0.72
            ax.plot([lx, lx], [yy - sh * 0.46, yy + sh * 0.46], lw=2.0,
                    color=C["muted"], alpha=min(a, 1.0) * 0.9, zorder=6)
            _arrow(ax, (lx, yy + sh * 0.46), (bx + 1.20, yy + sh * 0.30),
                   C["muted"], 1.8, a * 0.9, 11)
            R.txt(ax, lx - 0.18, yy, "+", 20, C["muted"], a, ha="right")

    a2 = _fade(u, R.rt(n + 1, 1.2 + n * 1.0, 0.7), 0.6)
    if o.get("input_label"):
        _arrow(ax, (bx + bw / 2, top - bh - 0.62), (bx + bw / 2, top - bh - 0.08),
               C["muted"], 2.2, a2)
        R.txt(ax, bx + bw / 2, top - bh - 0.92, o["input_label"], 17, C["muted"],
              a2, ha="center")
    if o.get("output_label"):
        _arrow(ax, (bx + bw / 2, top + 0.05), (bx + bw / 2, top + 0.75),
               C["muted"], 2.2, a2)
    _tail(R, ax, o, u, n + 2, x, y, w)


def stream_router(R, ax, u, d, o):
    C, P = R.c, _pal(R)
    items = o.get("items", [["cat", 0], ["beetle", 1], ["dog", 0],
                            ["honeycomb", 1], ["bus", 0], ["mushroom", 1]])
    rate = float(o.get("rate", 1.25))
    start = R.rt(1, 2.0, 1.0)

    a = _fade(u, R.rt(0, 0.6, 0.6), 0.5)
    R.box(ax, 6.55, 4.15, 2.60, 2.05, a=a, fc=C["panel"], r=0.18)
    R.txt(ax, 7.85, 5.52, o.get("box_title", ""), 25, C["ink"], a, ha="center", weight="bold")
    R.txt(ax, 7.85, 4.98, o.get("box_sub1", ""), 16, C["muted"], a, ha="center")
    R.txt(ax, 7.85, 4.55, o.get("box_sub2", ""), 16, C["muted"], a, ha="center")

    ab = _fade(u, R.rt(0, 1.0, 0.6), 0.5)
    R.box(ax, 11.55, 5.55, 3.55, 1.35, a=ab * 0.45, fc=C["pos"], r=0.16)
    R.txt(ax, 11.80, 6.62, o.get("bin_top", ""), 18, C["pos"], ab, weight="bold")
    R.box(ax, 11.55, 3.10, 3.55, 1.35, a=ab * 0.45, fc=C["neg"], r=0.16)
    R.txt(ax, 11.80, 2.82, o.get("bin_bottom", ""), 18, C["neg"], ab, weight="bold")
    _arrow(ax, (9.25, 5.20), (11.35, 6.20), C["pos"], 2.0, ab * 0.5)
    _arrow(ax, (9.25, 5.20), (11.35, 4.20), C["neg"], 2.0, ab * 0.5)

    idn = oodn = 0
    for k, (name, ood) in enumerate(items):
        t0 = start + k * rate
        p = (u - t0) / (rate * 1.25)
        if p < 0:
            continue
        col = C["neg"] if ood else C["pos"]
        if p >= 1.0:
            if ood:
                x, y = 11.75 + oodn * 1.14, 3.48
                oodn += 1
            else:
                x, y = 11.75 + idn * 1.14, 5.93
                idn += 1
            _chip(R, ax, x, y, 1.02, 0.62, name, col, 1.0, 13)
            continue
        x = 1.05 + _ease(p) * 10.4
        known = x > 9.2
        y = 5.20 + ((0.46 if not ood else -0.46) * _ease((x - 9.2) / 2.3)
                    if known else 0.0)
        _chip(R, ax, x, y - 0.31, 1.02, 0.62, name,
              col if known else C["dim"], 1.0, 13)


# --------------------------------------------------------------------------- #
# 2. shared_space — one sphere, two clusters
# --------------------------------------------------------------------------- #


def swap(R, ax, u, d, o):
    """Eq. 14: same formula as NegLabel, with w in place of r."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    cy = y + h / 2

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, x + w / 2, cy + 1.55, o.get("context", ""), 19, C["muted"], a0, ha="center")
    R.txt(ax, x + w / 2, cy + 0.85, o.get("before", ""), 24,
          P["txt"], a0, ha="center", weight="bold")

    a1 = _fade(u, R.rt(1, 1.6, 0.7), 0.6)
    _arrow(ax, (x + w / 2, cy + 0.35), (x + w / 2, cy - 0.55), C["accent"],
           3.0, a1, 18)

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.7)
    R.txt(ax, x + w / 2, cy - 1.05, o.get("after", ""),
          24, P["vis"], a2, ha="center", weight="bold")
    R.txt(ax, x + w / 2, cy - 1.70, o.get("note", ""), 18,
          C["muted"], a2, ha="center")






# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def threshold_line(R, ax, u, d, o):
    """Scores on a line with a threshold splitting them — any accept/reject
    rule, calibration, FPR/TPR argument.

    opts: threshold (0..1), lo_label, hi_label, rule, n, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    bx, by, bw = x + 0.9, y + h * 0.55, w - 1.8
    thr = float(o.get("threshold", 0.7))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([bx, bx + bw], [by, by], lw=3.0, color=C["line"],
            alpha=min(a0, 1.0), zorder=4)
    rng = np.random.default_rng(int(o.get("seed", 12)))
    pts = list(rng.beta(2, 5, int(o.get("n", 30)) // 2)) + \
        list(rng.beta(6, 2, int(o.get("n", 30)) // 2))
    for k, p in enumerate(pts):
        a = _fade(u, R.rt(0, 0.4, 0.6) + k * 0.02, 0.35)
        col = C["pos"] if p >= thr else C["neg"]
        show = _fade(u, R.rt(1, 1.8, 0.7), 0.5) > 0.5
        ax.plot([bx + p * bw], [by + 0.30 + (k % 4) * 0.16], "o", ms=9,
                color=col if show else C["dim"], alpha=min(a, 1.0) * 0.85,
                zorder=6)

    a1 = _fade(u, R.rt(1, 1.8, 0.7), 0.6)
    ax.plot([bx + thr * bw, bx + thr * bw], [by - 0.5, by + 1.5], lw=3.0,
            ls=(0, (4, 3)), color=C["accent"], alpha=min(a1, 1.0), zorder=8)
    R.txt(ax, bx + thr * bw, by + 1.7, o.get("rule", ""), 18, C["accent"], a1,
          ha="center", weight="bold")
    R.txt(ax, bx, by - 0.85, o.get("lo_label", ""), 17, C["neg"], a1)
    R.txt(ax, bx + bw, by - 0.85, o.get("hi_label", ""), 17, C["pos"], a1,
          ha="right")
    _tail(R, ax, o, u, 2, x, y, w)


def timeline(R, ax, u, d, o):
    """Ordered phases along a line — training stages, a protocol, a history.

    opts: events [{label, note, colour}], axis_label, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    ev = o.get("events", [])
    n = max(len(ev), 1)
    ly = y + h * 0.55

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([x + 0.6, x + w - 0.6], [ly, ly], lw=3.0, color=C["line"],
            alpha=min(a0, 1.0), zorder=4)
    for k, e in enumerate(ev):
        a = _fade(u, R.rt(k, 0.6 + k * 0.9, 0.55), 0.5)
        if a <= 0.01:
            continue
        px = x + 0.6 + (w - 1.2) * (k + 0.5) / n
        col = _col(R, e.get("colour", "hot"))
        ax.plot([px], [ly], "o", ms=15, color=col, alpha=min(a, 1.0), zorder=7)
        up = k % 2 == 0
        R.txt(ax, px, ly + (0.75 if up else -0.85), str(e.get("label", "")), 18,
              C["ink"], a, ha="center", weight="bold")
        if e.get("note"):
            R.txt(ax, px, ly + (1.25 if up else -1.35), e["note"], 15,
                  C["muted"], a, ha="center")
    R.txt(ax, x + w / 2, y + 0.65, o.get("axis_label", ""), 16, C["muted"], a0,
          ha="center")
    _tail(R, ax, o, u, n, x, y, w)


def token_compare(R, ax, u, d, o):
    """576 visual tokens versus a handful of text tokens: what the LLM eats."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cy = y + h * 0.62

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, x + 0.2, cy - 0.75, 1.7, 1.5, a=a0, fc=P["vis"], r=0.14, z=6)
    R.txt(ax, x + 1.05, cy, o.get("source", ""), 17, "#FFFFFF", a0, ha="center", z=9)
    _arrow(ax, (x + 2.05, cy), (x + 3.0, cy), C["muted"], 2.0, a0)
    R.box(ax, x + 3.1, cy - 0.75, 2.3, 1.5, a=a0, fc=C["panel"], r=0.14, z=6)
    R.txt(ax, x + 4.25, cy + 0.18, o.get("encoder", ""), 16, C["ink"], a0, ha="center", z=9)
    R.txt(ax, x + 4.25, cy - 0.32, o.get("encoder_note", ""), 14, C["muted"], a0, ha="center", z=9)

    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    gx, gy, cell = x + 6.0, cy + 1.15, 0.185
    for i in range(24):
        for j in range(24):
            g = _fade(u, R.rt(1, 1.4, 0.7) + (i + j) * 0.006, 0.3)
            ax.add_patch(Rectangle((gx + j * cell, gy - i * cell),
                                   cell * 0.82, cell * 0.82, fc=P["vis"],
                                   ec="none", alpha=min(g, 1.0) * 0.85, zorder=6))
    R.txt(ax, gx + 12 * cell, gy + 0.55, o.get("grid_label", ""), 19, P["vis"], a1,
          ha="center", weight="bold")

    a2 = _fade(u, R.rt(2, 3.0, 0.7), 0.6)
    tx = gx + 12 * cell - 2.2
    for k in range(10):
        ax.add_patch(Rectangle((tx + k * 0.44, y + h * 0.16), 0.34, 0.34,
                               fc=P["txt"], ec="none",
                               alpha=min(a2, 1.0) * 0.9, zorder=6))
    R.txt(ax, tx + 2.2, y + h * 0.16 - 0.45, o.get("few_label", ""), 19, P["txt"],
          a2, ha="center", weight="bold")

    a3 = _fade(u, R.rt(3, 4.2, 0.8), 0.7)
    R.txt(ax, x + 0.2, y + 0.30,
          o.get("verdict", ""), 20, C["accent"],
          a3, weight="bold")


def two_clusters(R, ax, u, d, o):
    C, P = R.c, _pal(R)
    cx, cy, rad = 8.0, 4.9, 2.45
    a = _fade(u, R.rt(0, 0.4, 0.7), 0.6)
    ax.add_patch(Circle((cx, cy), rad, fill=False, ec=P["grid"], lw=2.0,
                        alpha=min(a, 1.0), zorder=4))
    R.txt(ax, cx, cy + rad + 0.5, o.get("space_label", ""),
          18, C["muted"], a, ha="center")

    rng = np.random.default_rng(int(o.get("seed", 4)))
    at = _fade(u, R.rt(1, 1.4, 0.8), 0.6)
    for ang in rng.uniform(0.45, 1.35, 14):
        rr = rad * rng.uniform(0.97, 1.03)
        ax.plot([cx + rr * np.cos(ang)], [cy + rr * np.sin(ang)], "o", ms=10,
                color=P["txt"], alpha=min(at, 1.0) * 0.9, zorder=6)
    av = _fade(u, R.rt(1, 1.4, 0.8), 0.6)
    for ang in rng.uniform(3.6, 4.8, 20):
        rr = rad * rng.uniform(0.97, 1.03)
        ax.plot([cx + rr * np.cos(ang)], [cy + rr * np.sin(ang)], "o", ms=10,
                color=P["vis"], alpha=min(av, 1.0) * 0.9, zorder=6)

    R.txt(ax, cx + 2.35, cy + 2.05, o.get("a_label", ""), 18, P["txt"], at)
    R.txt(ax, cx - 2.40, cy - 2.20, o.get("b_label", ""), 18, P["vis"], av,
          ha="right")

    # the separating band
    ag = _fade(u, R.rt(2, 2.6, 0.9), 0.7)
    if ag > 0.01:
        t = np.linspace(2.1, 5.4, 60)
        ax.plot(cx + rad * 1.16 * np.cos(t), cy + rad * 1.16 * np.sin(t),
                ls=(0, (4, 4)), lw=2.2, color=C["neg"], alpha=min(ag, 1.0) * 0.8,
                zorder=5)
        R.txt(ax, cx - 3.85, cy + 0.15, o.get("gap_label", ""), 20, C["neg"], ag,
              ha="center", weight="bold")


# --------------------------------------------------------------------------- #
# 3. subspace_gap — the core geometric argument
# --------------------------------------------------------------------------- #


def vector_decompose(R, ax, u, d, o):
    C, P = R.c, _pal(R)
    cx, cy, s = float(o.get("cx", 8.4)), float(o.get("cy", 4.65)), \
        float(o.get("scale", 1.62))

    ap = _fade(u, R.rt(0, 0.4, 0.7), 0.7)
    if ap > 0.01:
        corners = [(-2.1, -1.8, 0), (2.1, -1.8, 0), (2.1, 1.8, 0), (-2.1, 1.8, 0)]
        pts = [_proj3(c, cx, cy, s) for c in corners]
        ax.add_patch(Polygon(pts, closed=True, fc=P["vis"], ec=P["vis"], lw=1.6,
                             alpha=min(ap, 1.0) * 0.14, zorder=3))
        R.txt(ax, *_proj3((-1.95, 1.50, 0), cx, cy, s), o.get("plane_label", ""), 26,
              P["vis"], ap, weight="bold")
        R.txt(ax, *_proj3((-1.05, 1.50, 0), cx, cy, s),
              o.get("plane_note", ""), 15, P["vis"], ap)
        rng = np.random.default_rng(11)
        for _ in range(30):
            X, Y = _proj3((rng.uniform(-1.85, 1.85), rng.uniform(-1.5, 1.5), 0),
                          cx, cy, s)
            ax.plot([X], [Y], "o", ms=7, color=P["vis"],
                    alpha=min(ap, 1.0) * 0.5, zorder=5)

    O = _proj3((0, 0, 0), cx, cy, s)
    wtip = _proj3((1.45, -0.35, 0), cx, cy, s)
    aw = _fade(u, R.rt(1, 1.4, 0.8), 0.6)
    _arrow(ax, O, wtip, P["vis"], 3.2, aw)
    R.txt(ax, wtip[0] + 0.28, wtip[1] - 0.36, o.get("in_label", ""), 26, P["vis"],
          aw, weight="bold")

    rvec = (0.62, 0.92, 1.42)
    rtip = _proj3(rvec, cx, cy, s)
    ar = _fade(u, R.rt(2, 2.4, 0.8), 0.7)
    _arrow(ax, O, rtip, P["txt"], 3.2, ar)
    R.txt(ax, rtip[0] + 0.26, rtip[1] + 0.16, o.get("out_label", ""), 26, P["txt"],
          ar, weight="bold")

    rproj = _proj3((rvec[0], rvec[1], 0), cx, cy, s)
    apj = _fade(u, R.rt(3, 3.4, 0.8), 0.7)
    if apj > 0.01:
        ax.plot([rtip[0], rproj[0]], [rtip[1], rproj[1]], ls=(0, (3, 3)),
                lw=2.2, color=P["txt"], alpha=min(apj, 1.0) * 0.85, zorder=7)
        _arrow(ax, O, rproj, P["txt"], 2.0, apj * 0.55, 12)
        R.txt(ax, rtip[0] + 0.30, (rtip[1] + rproj[1]) / 2,
              o.get("out_note", ""), 15, P["txt"], apj)
        R.txt(ax, rproj[0] - 0.10, rproj[1] - 0.44,
              o.get("proj_label", ""), 17, P["txt"],
              apj, ha="center")

    agp = _fade(u, R.rt(4, 4.4, 0.8), 0.6)
    if agp > 0.01:
        ax.plot([wtip[0], rproj[0]], [wtip[1], rproj[1]], lw=3.0, color=C["neg"],
                alpha=min(agp, 1.0), zorder=9)
        R.txt(ax, (wtip[0] + rproj[0]) / 2 + 0.62,
              (wtip[1] + rproj[1]) / 2 + 0.34, o.get("gap_label", ""), 18, C["neg"], agp,
              ha="center", weight="bold")


# --------------------------------------------------------------------------- #
# 4. prototype_walk — the method, animated
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# PRIMITIVES — added while working through SynOOD (Li et al., ICCV 2025)
# --------------------------------------------------------------------------- #

def feedback_loop(R, ax, u, d, o):
    """A forward chain with a gradient or signal fed back to an earlier stage.

    The shape for anything iterative-with-a-critic: adversarial attacks,
    guided generation, RL, actor-critic, GAN training.

    opts: stages [[label, colour]], feedback_from, feedback_to,
          feedback_label, iterations_label, verdict, note
    """
    C = R.c
    x, y, w, h = _box(o)
    stages = [tuple(s) if isinstance(s, (list, tuple)) else (s, "mid")
              for s in o.get("stages", [])]
    n = max(len(stages), 1)
    sw = (w - 0.8) / n
    bh = min(1.20, h * 0.26)
    cy = y + h * 0.66

    for k, st in enumerate(stages):
        a = _fade(u, R.rt(k, 0.5 + k * 0.7, 0.55), 0.5)
        if a <= 0.01:
            continue
        col = _col(R, st[1] if len(st) > 1 else "mid")
        light = col == C["panel"]
        bx = x + 0.35 + k * sw
        R.box(ax, bx, cy - bh / 2, sw * 0.82, bh, a=a, fc=col, r=0.12, z=7)
        R.txt(ax, bx + sw * 0.41, cy, str(st[0]), 14,
              C["ink"] if light else "#FFFFFF", a, ha="center", z=9)
        if k:
            _arrow(ax, (bx - sw * 0.16, cy), (bx - 0.03, cy), C["muted"], 1.9, a)

    fi = int(o.get("feedback_from", n - 1))
    fj = int(o.get("feedback_to", 0))
    ab = _fade(u, R.rt(n, 0.5 + n * 0.7, 0.8), 0.6)
    if ab > 0.01 and 0 <= fi < n and 0 <= fj < n:
        x1 = x + 0.35 + fi * sw + sw * 0.41
        x2 = x + 0.35 + fj * sw + sw * 0.41
        dip = cy - bh / 2 - 1.05
        ax.plot([x1, x1], [cy - bh / 2, dip], lw=2.6, color=C["neg"],
                alpha=min(ab, 1.0), zorder=6)
        ax.plot([x1, x2], [dip, dip], lw=2.6, color=C["neg"],
                alpha=min(ab, 1.0), zorder=6)
        _arrow(ax, (x2, dip), (x2, cy - bh / 2 - 0.05), C["neg"], 2.6, ab, 16)
        R.txt(ax, (x1 + x2) / 2, dip - 0.38, o.get("feedback_label", ""), 18,
              C["neg"], ab, ha="center", weight="bold")
    if o.get("iterations_label"):
        R.txt(ax, x + w - 0.3, cy + bh / 2 + 0.45, o["iterations_label"], 18,
              C["accent"], ab, ha="right", weight="bold")
    _tail(R, ax, o, u, n + 1, x, y, w)


def boundary_band(R, ax, u, d, o):
    """Two populations with the margin between them highlighted, and new
    points appearing inside that margin.

    The shape for hard-example mining, boundary sampling, active learning,
    adversarial examples near a decision surface.

    opts: a_label, b_label, band_label, synth_label, n_a, n_b, n_synth,
          verdict, seed
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cx, cy = x + w / 2, y + h * 0.56
    sc = min(w * 0.11, h * 0.26)
    rng = np.random.default_rng(int(o.get("seed", 17)))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for px, py in [(cx - 2.0 * sc + rng.normal(0, sc * .42),
                    cy + rng.normal(0, sc * .55))
                   for _ in range(int(o.get("n_a", 34)))]:
        ax.plot([px], [py], "o", ms=9, color=C["pos"], alpha=min(a0, 1.) * .8,
                zorder=6)
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    for px, py in [(cx + 2.1 * sc + rng.normal(0, sc * .46),
                    cy + rng.normal(0, sc * .58))
                   for _ in range(int(o.get("n_b", 34)))]:
        ax.plot([px], [py], "o", ms=9, color=C["neg"], alpha=min(a1, 1.) * .8,
                zorder=6)
    R.txt(ax, cx - 2.0 * sc, cy - sc * 1.48, o.get("a_label", ""), 18,
          C["pos"], a0, ha="center", weight="bold")
    R.txt(ax, cx + 2.1 * sc, cy - sc * 1.48, o.get("b_label", ""), 18,
          C["neg"], a1, ha="center", weight="bold")

    a2 = _fade(u, R.rt(2, 2.4, 0.7), 0.6)
    bw = sc * 0.92
    ax.add_patch(Rectangle((cx - bw / 2, cy - sc * 1.5), bw, sc * 3.0,
                           fc=C["accent"], ec="none", alpha=min(a2, 1.0) * 0.22,
                           zorder=4))
    ax.plot([cx, cx], [cy - sc * 1.5, cy + sc * 1.5], ls=(0, (4, 3)), lw=2.4,
            color=C["accent"], alpha=min(a2, 1.0), zorder=5)
    R.txt(ax, cx, cy + sc * 1.66, o.get("band_label", ""), 18,
          C["accent"], a2, ha="center", weight="bold")

    t3 = R.rt(3, 3.6, 0.8)
    ns = int(o.get("n_synth", 14))
    grown = int(min(ns, max(0.0, (u - t3)) * 6.0))
    for k in range(grown):
        px = cx + rng.uniform(-bw * 0.46, bw * 0.46)
        py = cy + rng.uniform(-sc * 1.25, sc * 1.25)
        ax.plot([px], [py], "*", ms=17, color=C["accent"], alpha=0.95, zorder=8)
    if grown:
        R.txt(ax, cx, cy + sc * 1.28, o.get("synth_label", ""), 17,
              C["accent"], 1.0, ha="center")
    _tail(R, ax, o, u, 4, x, y, w)


def dual_space_align(R, ax, u, d, o):
    """Two feature spaces stacked, with correspondence lines between them that
    start crossed and straighten out.

    The shape for any alignment claim: contrastive learning, retrieval,
    cross-modal matching, before/after fine-tuning.

    opts: top_label, bottom_label, links [[bottom_i, top_i_before, top_i_after]],
          n, before_note, after_note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    ty, by = y + h * 0.82, y + h * 0.40
    bw = w - 4.2
    bx = x + 3.3
    links = o.get("links", [])
    n = max(len(links), 1)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for yy, lab, col in ((ty, o.get("top_label", ""), P["txt"]),
                         (by, o.get("bottom_label", ""), P["vis"])):
        ax.add_patch(FancyBboxPatch((bx - 0.5, yy - 0.62), bw + 1.0, 1.24,
                                    boxstyle="round,pad=0,rounding_size=0.6",
                                    fc=col, ec="none", alpha=min(a0, 1.0) * 0.14,
                                    zorder=3))
        R.txt(ax, bx - 0.7, yy, lab, 17, col, a0, ha="right", weight="bold")

    t2 = R.rt(2, 2.6, 0.9)
    prog = _ease(min(1.0, max(0.0, (u - t2) / max(d - t2 - 0.9, 1.2))))
    for k, L in enumerate(links):
        a = _fade(u, R.rt(1, 1.2, 0.6) + k * 0.12, 0.5)
        if a <= 0.01:
            continue
        b_i, t_before, t_after = (list(L) + [L[1]])[:3]
        px = bx + float(b_i) * bw
        qx = bx + (float(t_before) + (float(t_after) - float(t_before)) * prog) * bw
        crossed = abs(float(t_before) - float(b_i)) > 0.12
        col = C["neg"] if (crossed and prog < 0.5) else C["accent"]
        ax.plot([px, qx], [by + 0.55, ty - 0.55], lw=2.2, color=col,
                alpha=min(a, 1.0) * 0.85, zorder=6)
        ax.plot([px], [by], "o", ms=11, color=P["vis"], alpha=min(a, 1.0), zorder=8)
        ax.plot([qx], [ty], "s", ms=10, color=P["txt"], alpha=min(a, 1.0), zorder=8)

    R.txt(ax, x + w / 2, y + 0.85, o.get("before_note", ""), 18, C["neg"],
          _fade(u, R.rt(1, 1.2, 0.6), 0.6) * (1 - prog), ha="center",
          weight="bold")
    R.txt(ax, x + w / 2, y + 0.85, o.get("after_note", ""), 18, C["accent"],
          _fade(u, t2, 0.6) * prog, ha="center", weight="bold")
    _tail(R, ax, o, u, 3, x, y, w)


def module_states(R, ax, u, d, o):
    """Modules tagged frozen or trainable, so it is obvious what actually gets
    updated. The shape for adapters, LoRA, prompt tuning, partial fine-tuning.

    opts: modules [{label, state: "frozen"|"train", note}], legend, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    mods = o.get("modules", [])
    n = max(len(mods), 1)
    mw = (w - 0.8) / n
    cy = y + h * 0.58
    bh = min(1.5, h * 0.32)

    for k, m in enumerate(mods):
        a = _fade(u, R.rt(k, 0.5 + k * 0.8, 0.6), 0.55)
        if a <= 0.01:
            continue
        train = str(m.get("state", "frozen")).startswith("t")
        col = _col(R, "hot" if train else "dim")
        bx = x + 0.4 + k * mw
        R.box(ax, bx, cy - bh / 2, mw * 0.84, bh, a=a, fc=col, r=0.12, z=7)
        R.txt(ax, bx + mw * 0.42, cy + 0.16, str(m.get("label", "")), 15,
              "#FFFFFF", a, ha="center", z=9)
        R.txt(ax, bx + mw * 0.42, cy - 0.34,
              "trainable" if train else "frozen", 13,
              "#FFFFFF" if train else C["ink"], a * 0.95, ha="center", z=9)
        if m.get("note"):
            R.txt(ax, bx + mw * 0.42, cy - bh / 2 - 0.40, m["note"], 14,
                  C["muted"], a, ha="center")
        if k:
            _arrow(ax, (bx - mw * 0.14, cy), (bx - 0.03, cy), C["muted"], 1.8, a)
    _tail(R, ax, o, u, n, x, y, w)


def pair_strip(R, ax, u, d, o):
    """Two rows of paired qualitative examples with column headings — the
    "before / after" figure every vision paper ends on.

    opts: columns [str], row_a, row_b, colour_a, colour_b, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    cols = o.get("columns", [])
    n = max(len(cols), 1)
    cw = (w - 2.6) / n
    gx = x + 2.4
    tile = min(cw * 0.86, (h - 2.2) / 2 * 0.92)
    ya = y + h - 1.3 - tile
    yb = ya - tile - 0.35
    rng = np.random.default_rng(21)

    for r, (lab, key, beat) in enumerate(((o.get("row_a", ""), "colour_a", 0),
                                          (o.get("row_b", ""), "colour_b", 1))):
        a = _fade(u, R.rt(beat, 0.5 + beat * 1.4, 0.6), 0.6)
        if a <= 0.01:
            continue
        yy = ya if r == 0 else yb
        col = _col(R, o.get(key, "in" if r == 0 else "hot"))
        R.txt(ax, gx - 0.3, yy + tile / 2, lab, 17, C["ink"], a, ha="right",
              weight="bold")
        for k in range(n):
            g = _fade(u, R.rt(beat, 0.5 + beat * 1.4, 0.6) + k * 0.07, 0.35)
            for _ in range(9):          # a few blocks, so a tile reads as an image
                ox = rng.uniform(0, tile * 0.7)
                oy = rng.uniform(0, tile * 0.7)
                ax.add_patch(Rectangle((gx + k * cw + ox, yy + oy),
                                       tile * 0.3, tile * 0.3, fc=col,
                                       ec="none", alpha=min(g, 1.0) * 0.30,
                                       zorder=5))
            ax.add_patch(Rectangle((gx + k * cw, yy), tile, tile, fill=False,
                                   ec=col, lw=1.8, alpha=min(g, 1.0), zorder=6))
            if r == 0 and k < len(cols):
                R.txt(ax, gx + k * cw + tile / 2, ya + tile + 0.28,
                      str(cols[k]), 13, C["muted"], g, ha="center")
    _tail(R, ax, o, u, 2, x, y, w)


def ablation_grid(R, ax, u, d, o):
    """A component-ablation table: ticks for what is switched on, metrics on
    the right, best row highlighted. One row per beat.

    opts: components [str], metrics [str], rows [{marks [bool], values [..],
          best}], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    comps = o.get("components", [])
    mets = o.get("metrics", [])
    rows = o.get("rows", [])
    nc, nm, nr = len(comps), len(mets), max(len(rows), 1)
    cw = (w * 0.52) / max(nc, 1)
    mw = (w * 0.34) / max(nm, 1)
    gx = x + 0.4
    mx = x + 0.4 + w * 0.56
    top = y + h - 1.0
    rh = min(0.72, (h - 1.9) / nr)

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for k, c in enumerate(comps):
        R.txt(ax, gx + k * cw + cw / 2, top + 0.42, str(c), 15, C["muted"], a0,
              ha="center")
    for k, m in enumerate(mets):
        R.txt(ax, mx + k * mw + mw / 2, top + 0.42, str(m), 15, C["muted"], a0,
              ha="center")
    ax.plot([gx - 0.2, x + w - 0.3], [top + 0.16, top + 0.16], lw=1.4,
            color=C["line"], alpha=min(a0, 1.0))

    for r, row in enumerate(rows):
        a = _fade(u, R.rt(min(r + 1, 5), 0.9 + r * 0.6, 0.5), 0.45)
        if a <= 0.01:
            continue
        yy = top - 0.35 - r * rh
        best = bool(row.get("best"))
        if best:
            ax.add_patch(Rectangle((gx - 0.25, yy - rh * 0.45),
                                   w - 0.35, rh * 0.9, fc=C["accent"],
                                   ec="none", alpha=min(a, 1.0) * 0.16, zorder=4))
        for k in range(nc):
            on = bool(row.get("marks", [])[k]) if k < len(row.get("marks", [])) else False
            R.txt(ax, gx + k * cw + cw / 2, yy, "✓" if on else "–", 19,
                  C["accent"] if on else C["dim"], a, ha="center",
                  weight="bold" if on else "normal")
        for k in range(nm):
            v = row.get("values", [])[k] if k < len(row.get("values", [])) else ""
            R.txt(ax, mx + k * mw + mw / 2, yy, str(v), 17,
                  C["ink"] if best else C["muted"], a, ha="center",
                  weight="bold" if best else "normal")
    _tail(R, ax, o, u, min(nr + 1, 6), x, y, w)


def peak_curve(R, ax, u, d, o):
    """A curve that improves, peaks, and then declines, with the optimum marked
    and the over-shoot region shaded. Any "more is better, until it isn't".

    opts: xs [str], ys [float], peak, xlabel, ylabel, peak_label,
          overshoot_label, higher_is_better, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    xs = o.get("xs", [])
    ys = [float(v) for v in o.get("ys", [])]
    if len(ys) < 3:
        return
    pk = int(o.get("peak", int(np.argmax(ys) if o.get("higher_is_better", True)
                              else np.argmin(ys))))
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.25 or 1.0
    px0, py0 = x + 1.5, y + 1.2
    pw, ph = w - 2.2, h - 2.4

    def PX(i):
        return px0 + i / max(len(ys) - 1, 1) * pw

    def PY(v):
        return py0 + (v - lo + pad / 2) / (hi - lo + pad) * ph

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    R.txt(ax, px0 + pw / 2, y + 0.55, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.75, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)

    t1 = R.rt(1, 1.3, 0.8)
    prog = _ease(min(1.0, max(0.0, (u - t1) / max(d - t1 - 1.4, 1.2))))
    m = max(2, int(len(ys) * prog))
    ax.plot([PX(i) for i in range(m)], [PY(v) for v in ys[:m]], lw=3.2,
            color=C["accent"], alpha=min(_fade(u, t1, 0.5), 1.0), zorder=7)
    for i in range(m):
        ax.plot([PX(i)], [PY(ys[i])], "o", ms=8, color=C["accent"],
                alpha=min(_fade(u, t1, 0.5), 1.0), zorder=8)
        if i < len(xs):
            R.txt(ax, PX(i), py0 - 0.32, str(xs[i]), 13, C["muted"],
                  min(_fade(u, t1, 0.5), 1.0), ha="center")

    a2 = _fade(u, R.rt(2, 3.0, 0.8), 0.6)
    if a2 > 0.01 and pk < len(ys):
        ax.plot([PX(pk)], [PY(ys[pk])], "o", ms=17, color=C["pos"],
                alpha=min(a2, 1.0), zorder=9)
        R.txt(ax, PX(pk), PY(ys[pk]) + 0.45, o.get("peak_label", ""), 18,
              C["pos"], a2, ha="center", weight="bold")
    a3 = _fade(u, R.rt(3, 4.2, 0.8), 0.6)
    if a3 > 0.01 and pk < len(ys) - 1:
        ax.add_patch(Rectangle((PX(pk), py0), pw - (PX(pk) - px0), ph,
                               fc=C["neg"], ec="none", alpha=min(a3, 1.0) * 0.13,
                               zorder=3))
        R.txt(ax, (PX(pk) + px0 + pw) / 2, py0 + ph * 0.34,
              o.get("overshoot_label", ""), 17, C["neg"], a3, ha="center")
    _tail(R, ax, o, u, 4, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — added while working through CMA (Kim & Hwang, CVPR 2025)
# --------------------------------------------------------------------------- #

def tradeoff_scatter(R, ax, u, d, o):
    """Methods plotted against two metrics at once, grouped and labelled, with
    optional improvement arrows. The shape for "nobody optimises both".

    opts: xlabel, ylabel, groups [{name, colour, points [[x, y, label]]}],
          arrows [[x0, y0, x1, y1, label]], corner_note, xlim, ylim, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    groups = o.get("groups", [])
    pts = [p for g in groups for p in g.get("points", [])]
    if not pts:
        return
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    x0, x1 = o.get("xlim", [min(xs) - 2, max(xs) + 2])
    y0, y1 = o.get("ylim", [min(ys) - 3, max(ys) + 3])
    px0, py0 = x + 1.6, y + 1.15
    pw, ph = w - 2.4, h - 2.2

    def PX(v):
        return px0 + (float(v) - x0) / max(x1 - x0, 1e-9) * pw

    def PY(v):
        return py0 + (float(v) - y0) / max(y1 - y0, 1e-9) * ph

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    R.txt(ax, px0 + pw / 2, y + 0.52, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.85, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)
    if o.get("corner_note"):
        R.txt(ax, px0 + 0.18, py0 + ph - 0.12, o["corner_note"], 15,
              C["muted"], a0)

    for gi, g in enumerate(groups):
        a = _fade(u, R.rt(gi, 0.6 + gi * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        col = _col(R, g.get("colour", "dim"))
        for p in g.get("points", []):
            hot = len(p) > 3 and bool(p[3])
            ax.plot([PX(p[0])], [PY(p[1])], "o", ms=17 if hot else 12,
                    color=C["accent"] if hot else col, alpha=min(a, 1.0),
                    zorder=8 if hot else 7)
            if len(p) > 2 and p[2]:
                R.txt(ax, PX(p[0]), PY(p[1]) + 0.42, str(p[2]), 14,
                      C["ink"] if hot else C["muted"], a, ha="center",
                      weight="bold" if hot else "normal")
        if g.get("name"):
            R.txt(ax, px0 + 0.18, py0 + 0.30 + (len(groups) - 1 - gi) * 0.46,
                  g["name"], 16, col, a, weight="bold")

    for k, arw in enumerate(o.get("arrows", [])):
        a = _fade(u, R.rt(len(groups), 0.6 + len(groups) * 0.9, 0.8) + k * 0.2, 0.6)
        if a <= 0.01:
            continue
        _arrow(ax, (PX(arw[0]), PY(arw[1])), (PX(arw[2]), PY(arw[3])),
               C["accent"], 2.4, a, 15)
        if len(arw) > 4:
            R.txt(ax, (PX(arw[0]) + PX(arw[2])) / 2, (PY(arw[1]) + PY(arw[3])) / 2
                  + 0.32, str(arw[4]), 15, C["accent"], a, ha="center",
                  weight="bold")
    _tail(R, ax, o, u, len(groups) + 1, x, y, w)


def hypersphere_shift(R, ax, u, d, o):
    """Groups of embeddings on a sphere that migrate from one arrangement to
    another. The shape for any representation-space rearrangement: modality
    gap, collapse, uniformity, alignment.

    opts: sphere_label, groups [{name, colour, from, to, spread, n, marker}],
          before_note, after_note, bars {labels, before, after}, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    has_bars = bool(o.get("bars"))
    cx = x + (w * 0.32 if has_bars else w * 0.5)
    cy = y + h * 0.55
    rad = min(w * 0.16, h * 0.34)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.add_patch(Circle((cx, cy), rad, fill=False, ec=_pal(R)["grid"], lw=2.0,
                        alpha=min(a0, 1.0), zorder=4))
    R.txt(ax, cx, cy + rad + 0.45, o.get("sphere_label", ""), 17, C["muted"],
          a0, ha="center")

    t_shift = R.rt(2, 2.6, 0.9)
    prog = _ease(min(1.0, max(0.0, (u - t_shift) / max(d - t_shift - 1.0, 1.2))))
    rng = np.random.default_rng(int(o.get("seed", 31)))
    for gi, g in enumerate(o.get("groups", [])):
        a = _fade(u, R.rt(min(gi, 1), 0.5 + gi * 0.7, 0.6), 0.55)
        if a <= 0.01:
            continue
        col = _col(R, g.get("colour", "in"))
        th0, th1 = float(g.get("from", 0)), float(g.get("to", g.get("from", 0)))
        th = np.radians(th0 + (th1 - th0) * prog)
        spread = np.radians(float(g.get("spread", 26)))
        for _ in range(int(g.get("n", 14))):
            aang = th + rng.normal(0, spread / 2.2)
            rr = rad * rng.uniform(0.94, 1.06)
            ax.plot([cx + rr * np.cos(aang)], [cy + rr * np.sin(aang)],
                    g.get("marker", "o"), ms=9, color=col, alpha=min(a, 1.) * .9,
                    zorder=7)
        if g.get("name"):
            lr = rad * (1.44 + (gi % 2) * 0.34)
            R.txt(ax, cx + lr * np.cos(th), cy + lr * np.sin(th),
                  g["name"], 15, col, a, ha="center", weight="bold")

    bars = o.get("bars")
    if bars:
        labs = bars.get("labels", [])
        before = [float(v) for v in bars.get("before", [])]
        after = [float(v) for v in bars.get("after", [])]
        bx = x + w * 0.62
        bw = w * 0.30
        ab = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
        for k, lab in enumerate(labs):
            yy = cy + rad * 0.9 - k * 0.86
            v0 = before[k] if k < len(before) else 0.0
            v1 = after[k] if k < len(after) else v0
            v = v0 + (v1 - v0) * prog
            hot = "ood" in str(lab).lower() or "<" in str(lab)
            ax.add_patch(Rectangle((bx, yy - 0.20), bw * v, 0.40,
                                   fc=C["neg"] if hot else C["accent"],
                                   ec="none", alpha=min(ab, 1.0), zorder=6))
            R.txt(ax, bx - 0.22, yy, str(lab), 15, C["muted"], ab, ha="right")
        R.txt(ax, bx, cy + rad * 0.9 + 0.62, bars.get("title", "similarity"),
              16, C["muted"], ab)

    R.txt(ax, x + w / 2, y + 0.78, o.get("before_note", ""), 18, C["neg"],
          _fade(u, R.rt(1, 1.4, 0.6), 0.6) * (1 - prog), ha="center",
          weight="bold")
    R.txt(ax, x + w / 2, y + 0.78, o.get("after_note", ""), 18, C["accent"],
          _fade(u, t_shift, 0.6) * prog, ha="center", weight="bold")
    _tail(R, ax, o, u, 3, x, y, w)


def metric_table(R, ax, u, d, o):
    """A numeric comparison table: methods down the side, metrics across, one
    row per beat, the paper's own row highlighted.

    opts: columns [str], rows [{label, values [..], best}],
          lower_is_better [bool per column], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    cols = o.get("columns", [])
    rows = o.get("rows", [])
    nc, nr = max(len(cols), 1), max(len(rows), 1)
    labw = w * 0.24
    cw = (w - labw - 0.6) / nc
    gx = x + 0.3 + labw
    top = y + h - 0.95
    rh = min(0.68, (h - 1.9) / nr)

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for k, c in enumerate(cols):
        R.txt(ax, gx + k * cw + cw / 2, top + 0.42, str(c), 15, C["muted"], a0,
              ha="center")
    ax.plot([x + 0.3, x + w - 0.3], [top + 0.16, top + 0.16], lw=1.4,
            color=C["line"], alpha=min(a0, 1.0))

    for r, row in enumerate(rows):
        a = _fade(u, R.rt(min(r + 1, 5), 0.8 + r * 0.55, 0.5), 0.45)
        if a <= 0.01:
            continue
        yy = top - 0.32 - r * rh
        best = bool(row.get("best"))
        if best:
            ax.add_patch(Rectangle((x + 0.25, yy - rh * 0.45), w - 0.5,
                                   rh * 0.9, fc=C["accent"], ec="none",
                                   alpha=min(a, 1.0) * 0.16, zorder=4))
        R.txt(ax, gx - 0.25, yy, str(row.get("label", "")), 16,
              C["ink"] if best else C["muted"], a, ha="right",
              weight="bold" if best else "normal")
        for k, v in enumerate(row.get("values", [])[:nc]):
            R.txt(ax, gx + k * cw + cw / 2, yy, str(v), 16,
                  C["ink"] if best else C["muted"], a, ha="center",
                  weight="bold" if best else "normal")
    _tail(R, ax, o, u, min(nr + 1, 6), x, y, w)


def embedding_grid(R, ax, u, d, o):
    """A row of small embedding plots, one per method — the t-SNE/UMAP
    comparison strip that ends most representation papers.

    opts: panels [{label, groups [{colour, angle, spread, n}]}],
          legend [[label, colour]], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    ps = o.get("panels", [])
    n = max(len(ps), 1)
    pw = (w - 0.8) / n
    rad = min(pw * 0.36, (h - 2.3) * 0.42)
    cy = y + h * 0.60
    rng = np.random.default_rng(int(o.get("seed", 41)))

    for k, p in enumerate(ps):
        a = _fade(u, R.rt(min(k, 3), 0.5 + k * 0.8, 0.6), 0.55)
        if a <= 0.01:
            continue
        cx = x + 0.4 + pw * (k + 0.5)
        last = k == n - 1
        ax.add_patch(Circle((cx, cy), rad, fill=False,
                            ec=C["accent"] if last else _pal(R)["grid"],
                            lw=2.4 if last else 1.6, alpha=min(a, 1.0), zorder=4))
        for g in p.get("groups", []):
            col = _col(R, g.get("colour", "in"))
            th = np.radians(float(g.get("angle", 0)))
            sp = np.radians(float(g.get("spread", 40)))
            for _ in range(int(g.get("n", 16))):
                aang = th + rng.normal(0, sp / 2.2)
                rr = rad * rng.uniform(0.30, 1.0)
                ax.plot([cx + rr * np.cos(aang)], [cy + rr * np.sin(aang)],
                        "o", ms=5, color=col, alpha=min(a, 1.) * .85, zorder=6)
        R.txt(ax, cx, cy - rad - 0.42, str(p.get("label", "")), 16,
              C["ink"] if last else C["muted"], a, ha="center",
              weight="bold" if last else "normal")

    leg = o.get("legend", [])
    al = _fade(u, R.rt(min(n, 3), 0.5 + n * 0.8, 0.7), 0.6)
    for k, (lab, colname) in enumerate(leg):
        lx = x + 0.5 + k * (w / max(len(leg), 1))
        ax.plot([lx], [y + 0.85], "o", ms=9, color=_col(R, colname),
                alpha=min(al, 1.0))
        R.txt(ax, lx + 0.28, y + 0.85, str(lab), 15, C["muted"], al)
    _tail(R, ax, o, u, n + 1, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — added while working through GOOD (Gao et al., NeurIPS 2025)
# --------------------------------------------------------------------------- #

def trajectory_guidance(R, ax, u, d, o):
    """A trajectory stepping across nested manifolds, with guidance arrows
    deflecting it off the default path.

    The shape for guided diffusion, but equally for optimisation paths,
    trajectory steering, or any "default dynamics plus a correction" story.

    opts: levels, level_labels [outer, mid, inner], path_label,
          deflections [{at, label, colour}], start_label, end_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("levels", 5))
    # centre the arc family below the box and size it so the whole band fits
    cx, cy = x + w * 0.50, y - h * 1.10
    r0 = (y + h - 0.95) - cy
    r1 = r0 - max(h - 2.5, 1.0)
    th = np.linspace(np.pi / 2 - 0.46, np.pi / 2 + 0.46, 90)

    def _R(k):
        return r0 - (r0 - r1) * k / max(n - 1, 1)

    def _ang(k):
        return np.pi / 2 + 0.34 - 0.68 * k / max(n - 1, 1)

    for k in range(n):
        a = _fade(u, R.rt(0, 0.4, 0.6) + k * 0.10, 0.5)
        if a <= 0.01:
            continue
        rr = _R(k)
        last = k == n - 1
        ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), lw=2.6,
                color=P["vis"] if last else C["dim"],
                alpha=min(a, 1.0) * (1.0 if last else 0.55), zorder=4)

    labs = o.get("level_labels", [])
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for k, lab in enumerate(labs[:3]):
        idx = [0, n // 2, n - 1][k]
        rr = _R(idx)
        ang = th[0]
        R.txt(ax, cx + rr * np.cos(ang) - 0.20, cy + rr * np.sin(ang),
              str(lab), 17, C["muted"], a0, ha="right")

    # the default (unguided) path: one point per manifold
    pts = [(cx + _R(k) * np.cos(_ang(k)), cy + _R(k) * np.sin(_ang(k)))
           for k in range(n)]
    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    for k in range(n - 1):
        _arrow(ax, pts[k], pts[k + 1], _col(R, "in"), 2.4,
               a1 * _fade(u, R.rt(1, 1.4, 0.7) + k * 0.12, 0.4), 14)
    R.txt(ax, pts[0][0] - 0.20, pts[0][1] + 0.40, o.get("start_label", ""), 16,
          C["ink"], a1, ha="center")
    R.txt(ax, pts[-1][0] + 0.20, pts[-1][1] - 0.42, o.get("end_label", ""), 16,
          C["ink"], a1, ha="center")
    R.txt(ax, x + 0.25, y + 0.80, o.get("path_label", ""), 16, _col(R, "in"), a1)

    # the guidance corrections that pull it off that path
    for j, g in enumerate(o.get("deflections", [])):
        a = _fade(u, R.rt(2 + j, 2.6 + j * 1.2, 0.7), 0.6)
        if a <= 0.01:
            continue
        k = int(g.get("at", 1 + j))
        k = max(0, min(k, n - 1))
        px, py = pts[k]
        ang = np.radians(float(g.get("angle", 18 + j * 34)))
        L = float(g.get("length", 1.35))
        col = _col(R, g.get("colour", "hot"))
        _arrow(ax, (px, py), (px + L * np.cos(ang), py + L * np.sin(ang)),
               col, 2.8, a, 16)
        R.txt(ax, px + (L + 0.15) * np.cos(ang), py + (L + 0.15) * np.sin(ang),
              str(g.get("label", "")), 16, col, a, weight="bold")
    _tail(R, ax, o, u, 2 + len(o.get("deflections", [])), x, y, w)


def energy_walk(R, ax, u, d, o):
    """A sample sliding along a density curve from the high-density mode into
    the low-density tail, following the gradient.

    The shape for energy-based reasoning: likelihood, free energy, typicality.

    opts: xlabel, ylabel, peak_label, tail_label, marker_label, arrow_label,
          reverse, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    px0, py0 = x + 1.5, y + 1.2
    pw, ph = w - 2.2, h - 2.3
    xs = np.linspace(0, 1, 240)
    ys = np.exp(-((xs - 0.34) ** 2) / (2 * 0.13 ** 2))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.fill_between(px0 + xs * pw, py0, py0 + ys * ph * 0.86, color=_col(R, "in"),
                    alpha=min(a0, 1.0) * 0.28, zorder=5)
    ax.plot(px0 + xs * pw, py0 + ys * ph * 0.86, lw=2.8, color=_col(R, "in"),
            alpha=min(a0, 1.0), zorder=6)
    R.txt(ax, px0 + pw / 2, y + 0.55, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.8, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    R.txt(ax, px0 + 0.34 * pw, py0 + ph * 0.95, o.get("peak_label", ""), 17,
          _col(R, "in"), a1, ha="center", weight="bold")

    t2 = R.rt(2, 2.6, 0.8)
    prog = _ease(min(1.0, max(0.0, (u - t2) / max(d - t2 - 1.2, 1.2))))
    fx = 0.34 + (0.86 - 0.34) * prog
    fy = np.exp(-((fx - 0.34) ** 2) / (2 * 0.13 ** 2))
    a2 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    ax.plot([px0 + fx * pw], [py0 + fy * ph * 0.86], "o", ms=17,
            color=C["accent"], alpha=min(a2, 1.0), zorder=9)
    R.txt(ax, px0 + fx * pw, py0 + fy * ph * 0.86 + 0.48,
          o.get("marker_label", ""), 17, C["accent"], a2, ha="center",
          weight="bold")
    if prog > 0.05:
        _arrow(ax, (px0 + 0.42 * pw, py0 + ph * 0.30),
               (px0 + 0.86 * pw, py0 + ph * 0.30), C["accent"], 2.6,
               _fade(u, t2, 0.6), 15)
        R.txt(ax, px0 + 0.64 * pw, py0 + ph * 0.42, o.get("arrow_label", ""),
              16, C["accent"], _fade(u, t2, 0.6), ha="center")
    a3 = _fade(u, R.rt(3, 4.0, 0.8), 0.6)
    R.txt(ax, px0 + 0.88 * pw, py0 + ph * 0.14, o.get("tail_label", ""), 17,
          C["neg"], a3, ha="right", weight="bold")
    _tail(R, ax, o, u, 4, x, y, w)


def knn_sparsity(R, ax, u, d, o):
    """A query point moving from a dense region into a sparse one, with its
    k-th nearest neighbour circled so the distance is visible.

    The shape for k-NN scoring, density estimation, coverage arguments.

    opts: k, dense_label, sparse_label, marker_label, radius_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cx, cy = x + w * 0.5, y + h * 0.56
    sc = min(w * 0.16, h * 0.34)
    rng = np.random.default_rng(int(o.get("seed", 23)))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    dense = [(cx - sc * 0.75 + rng.normal(0, sc * .34),
              cy + rng.normal(0, sc * .38)) for _ in range(46)]
    sparse = [(cx + sc * 1.05 + rng.normal(0, sc * .72),
               cy + rng.normal(0, sc * .70)) for _ in range(12)]
    for px, py in dense:
        ax.plot([px], [py], "o", ms=8, color=P["vis"], alpha=min(a0, 1.) * .75,
                zorder=5)
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    for px, py in sparse:
        ax.plot([px], [py], "o", ms=8, color=P["vis"], alpha=min(a1, 1.) * .75,
                zorder=5)
    R.txt(ax, cx - sc * 0.75, cy - sc * 1.15, o.get("dense_label", ""), 17,
          P["vis"], a0, ha="center", weight="bold")
    R.txt(ax, cx + sc * 1.15, cy - sc * 1.30, o.get("sparse_label", ""), 17,
          C["neg"], a1, ha="center", weight="bold")

    t2 = R.rt(2, 2.6, 0.8)
    prog = _ease(min(1.0, max(0.0, (u - t2) / max(d - t2 - 1.2, 1.2))))
    qx = cx - sc * 0.75 + (cx + sc * 1.05 - (cx - sc * 0.75)) * prog
    qy = cy + sc * 0.10
    pool = dense + sparse
    dists = sorted(np.hypot(px - qx, py - qy) for px, py in pool)
    kk = min(int(o.get("k", 5)), len(dists) - 1)
    rad = dists[kk]

    a2 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    ax.add_patch(Circle((qx, qy), rad, fill=False, ec=C["accent"], lw=2.4,
                        ls=(0, (4, 3)), alpha=min(a2, 1.0), zorder=8))
    ax.plot([qx], [qy], "*", ms=24, color=C["accent"], alpha=min(a2, 1.0),
            zorder=9)
    R.txt(ax, qx, qy + rad + 0.32, o.get("marker_label", ""), 17, C["accent"],
          a2, ha="center", weight="bold")
    a3 = _fade(u, R.rt(3, 4.0, 0.8), 0.7)
    R.txt(ax, x + w / 2, y + 0.72, o.get("radius_label", ""), 17,
          C["muted"], a3, ha="center")
    _tail(R, ax, o, u, 4, x, y, w)


def param_grid(R, ax, u, d, o):
    """A grid of small distributions over two swept hyperparameters, so the
    trend across the sweep is visible at a glance.

    opts: rows [labels], cols [labels], row_label, col_label, shift,
          highlight [r, c], note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rows = o.get("rows", [])
    cols = o.get("cols", [])
    nr, nc = max(len(rows), 1), max(len(cols), 1)
    gx, gy = x + 2.0, y + h - 0.95
    cw = (w - 2.6) / nc
    ch = (h - 2.0) / nr
    xs = np.linspace(0, 1, 60)
    shift = float(o.get("shift", 0.55))

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for c, lab in enumerate(cols):
        R.txt(ax, gx + (c + 0.5) * cw, gy + 0.32, str(lab), 14, C["muted"], a0,
              ha="center")
    for r, lab in enumerate(rows):
        R.txt(ax, gx - 0.22, gy - (r + 0.5) * ch, str(lab), 14, C["muted"], a0,
              ha="right")
    R.txt(ax, gx + nc * cw / 2, gy + 0.78, o.get("col_label", ""), 16,
          C["ink"], a0, ha="center")
    R.txt(ax, x + 0.55, gy - nr * ch / 2, o.get("row_label", ""), 16, C["ink"],
          a0, ha="center", va="center", rot=90)

    hl = o.get("highlight")
    for r in range(nr):
        for c in range(nc):
            a = _fade(u, R.rt(min(r, 3), 0.6 + r * 0.8, 0.5) + c * 0.06, 0.4)
            if a <= 0.01:
                continue
            mu = 0.30 + shift * ((r / max(nr - 1, 1) + c / max(nc - 1, 1)) / 2)
            ys = np.exp(-((xs - mu) ** 2) / (2 * 0.12 ** 2))
            bx = gx + c * cw
            by = gy - (r + 1) * ch
            hot = hl and r == int(hl[0]) and c == int(hl[1])
            col = C["accent"] if hot else _col(R, "in")
            ax.fill_between(bx + 0.10 + xs * (cw - 0.25), by + 0.10,
                            by + 0.10 + ys * (ch - 0.30), color=col,
                            alpha=min(a, 1.0) * (0.55 if hot else 0.32), zorder=5)
            ax.plot(bx + 0.10 + xs * (cw - 0.25), by + 0.10 + ys * (ch - 0.30),
                    lw=1.6, color=col, alpha=min(a, 1.0), zorder=6)
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(3, 4.0, 0.8), 0.7))
    _tail(R, ax, o, u, 4, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — added while working through LeJEPA (Balestriero & LeCun, 2025)
# --------------------------------------------------------------------------- #

def sliced_projection(R, ax, u, d, o):
    """A high-dimensional cloud projected onto several 1-D directions, with the
    resulting marginals compared against a target curve.

    The shape for sliced methods generally: sliced Wasserstein, Radon
    transforms, random-projection tests, Cramer-Wold arguments.

    opts: cloud ("x"|"blob"|"ring"|"aniso"), n_dirs, cloud_label, target_label,
          bad_dir, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cx, cy = x + w * 0.22, y + h * 0.56
    sc = min(w * 0.11, h * 0.30)
    rng = np.random.default_rng(int(o.get("seed", 29)))
    kind = o.get("cloud", "x")
    n_d = int(o.get("n_dirs", 5))
    bad = int(o.get("bad_dir", 2))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    if kind == "x":                     # standard marginals, degenerate joint
        t = rng.uniform(-1.6, 1.6, 260)
        s = np.where(rng.random(260) < 0.5, 1.0, -1.0)
        px, py = t, t * s + rng.normal(0, 0.12, 260)
    elif kind == "aniso":
        px, py = rng.normal(0, 1.5, 260), rng.normal(0, 0.35, 260)
    elif kind == "ring":
        th = rng.uniform(0, 2 * np.pi, 260)
        px, py = 1.3 * np.cos(th), 1.3 * np.sin(th)
    else:
        px, py = rng.normal(0, 1.0, 260), rng.normal(0, 1.0, 260)
    for i in range(len(px)):
        ax.plot([cx + px[i] * sc * 0.62], [cy + py[i] * sc * 0.62], "o", ms=5,
                color=P["vis"], alpha=min(a0, 1.0) * 0.55, zorder=5)
    R.txt(ax, cx, cy + sc * 1.50, o.get("cloud_label", ""), 17, P["vis"], a0,
          ha="center", weight="bold")

    # the projection directions
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    angs = [np.pi * k / max(n_d, 1) for k in range(n_d)]
    for k, th in enumerate(angs):
        hot = k == bad
        _arrow(ax, (cx, cy), (cx + sc * 1.28 * np.cos(th),
                              cy + sc * 1.28 * np.sin(th)),
               C["accent"] if hot else C["muted"], 2.4 if hot else 1.5,
               a1 * (1.0 if hot else 0.55), 13)

    # the resulting 1-D marginals, stacked on the right
    gx = x + w * 0.46
    gw = w * 0.48
    lane = (h - 2.0) / max(n_d, 1)
    ts = np.linspace(-3, 3, 120)
    tgt = np.exp(-ts ** 2 / 2)
    for k in range(n_d):
        a = _fade(u, R.rt(2, 2.4, 0.7) + k * 0.14, 0.5)
        if a <= 0.01:
            continue
        yy = y + h - 0.9 - k * lane
        hot = k == bad
        ax.plot(gx + (ts + 3) / 6 * gw, yy + tgt * lane * 0.78, lw=1.8,
                color=C["ink"], alpha=min(a, 1.0) * 0.45, zorder=6)
        if hot:                       # the direction that exposes the defect
            emp = 0.45 * np.exp(-(ts - 1.5) ** 2 / 0.4) + \
                  0.45 * np.exp(-(ts + 1.5) ** 2 / 0.4)
        else:
            emp = tgt * (1.0 + 0.05 * np.sin(3 * ts))
        ax.plot(gx + (ts + 3) / 6 * gw, yy + emp * lane * 0.78, lw=2.6,
                color=C["accent"] if hot else P["vis"], alpha=min(a, 1.0),
                zorder=7)
    R.txt(ax, gx + gw, y + h - 0.35, o.get("target_label", ""), 16, C["ink"],
          _fade(u, R.rt(2, 2.4, 0.7), 0.6), ha="right")
    R.txt(ax, x + 0.25, y + 0.78, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(3, 3.8, 0.8), 0.7))
    _tail(R, ax, o, u, 4, x, y, w)


def distribution_shapes(R, ax, u, d, o):
    """A row of small scatter panels, each a named distribution shape — healthy,
    collapsed to a point, collapsed to a subspace, anisotropic, and so on.

    The shape for failure-mode taxonomies in representation learning.

    opts: shapes [{label, kind: ball|point|line|ring|aniso, colour, good}],
          verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    ss = o.get("shapes", [])
    n = max(len(ss), 1)
    pw = (w - 0.8) / n
    rad = min(pw * 0.34, (h - 2.3) * 0.42)
    cy = y + h * 0.60
    rng = np.random.default_rng(int(o.get("seed", 37)))

    for k, s in enumerate(ss):
        a = _fade(u, R.rt(min(k, 3), 0.5 + k * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        cx = x + 0.4 + pw * (k + 0.5)
        good = bool(s.get("good"))
        col = _col(R, s.get("colour", "in" if good else "neg"))
        ax.add_patch(Circle((cx, cy), rad, fill=False,
                            ec=C["accent"] if good else _pal(R)["grid"],
                            lw=2.4 if good else 1.4, alpha=min(a, 1.0), zorder=4))
        kind = s.get("kind", "ball")
        m = 90
        if kind == "point":
            px, py = rng.normal(0, .05, m), rng.normal(0, .05, m)
        elif kind == "line":
            t = rng.normal(0, .55, m)
            px, py = t, t * 0.85 + rng.normal(0, .05, m)
        elif kind == "ring":
            th = rng.uniform(0, 2 * np.pi, m)
            px, py = .72 * np.cos(th), .72 * np.sin(th)
        elif kind == "aniso":
            px, py = rng.normal(0, .70, m), rng.normal(0, .18, m)
        else:
            px, py = rng.normal(0, .38, m), rng.normal(0, .38, m)
        for i in range(m):
            ax.plot([cx + px[i] * rad], [cy + py[i] * rad], "o", ms=4.5,
                    color=col, alpha=min(a, 1.0) * 0.75, zorder=6)
        R.txt(ax, cx, cy - rad - 0.42, str(s.get("label", "")), 16,
              C["ink"] if good else C["muted"], a, ha="center",
              weight="bold" if good else "normal")
    _tail(R, ax, o, u, n, x, y, w)


def siamese_predict(R, ax, u, d, o):
    """Two views through a shared encoder, with a predictive loss between the
    embeddings — the joint-embedding blueprint.

    opts: view_a, view_b, encoder, loss_label, shared_label, collapse_label,
          verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cy = y + h * 0.58
    dy = min(1.35, h * 0.26)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for i, lab in enumerate([o.get("view_a", ""), o.get("view_b", "")]):
        yy = cy + (dy if i == 0 else -dy)
        R.box(ax, x + 0.4, yy - 0.45, 2.6, 0.9, a=a0, fc=P["vis"], r=0.12, z=7)
        R.txt(ax, x + 1.7, yy, lab, 15, "#FFFFFF", a0, ha="center", z=9)
        _arrow(ax, (x + 3.1, yy), (x + 4.5, cy + (0.35 if i == 0 else -0.35)),
               C["muted"], 2.0, a0)

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    R.box(ax, x + 4.6, cy - 1.05, 2.6, 2.1, a=a1, fc=C["panel"], r=0.14, z=7)
    R.txt(ax, x + 5.9, cy + 0.16, o.get("encoder", ""), 17, C["ink"], a1,
          ha="center", z=9)
    R.txt(ax, x + 5.9, cy - 0.34, o.get("shared_label", ""), 14, C["muted"], a1,
          ha="center", z=9)

    a2 = _fade(u, R.rt(2, 2.4, 0.7), 0.6)
    for i in range(2):
        yy = cy + (dy if i == 0 else -dy)
        _arrow(ax, (x + 7.3, cy + (0.35 if i == 0 else -0.35)),
               (x + 8.6, yy), C["muted"], 2.0, a2)
        ax.plot([x + 8.95], [yy], "o", ms=16, color=_col(R, "out"),
                alpha=min(a2, 1.0), zorder=8)
    ax.plot([x + 8.95, x + 8.95], [cy - dy, cy + dy], ls=(0, (3, 3)), lw=2.2,
            color=C["accent"], alpha=min(a2, 1.0), zorder=7)
    R.txt(ax, x + 9.35, cy, o.get("loss_label", ""), 18, C["accent"], a2,
          weight="bold")

    a3 = _fade(u, R.rt(3, 3.6, 0.8), 0.7)
    if o.get("collapse_label"):
        _arrow(ax, (x + 10.9, cy), (x + 12.1, cy), C["neg"], 2.4, a3, 15)
        ax.plot([x + 12.6], [cy], "o", ms=20, color=C["neg"],
                alpha=min(a3, 1.0), zorder=8)
        R.txt(ax, x + 12.6, cy - 0.75, o["collapse_label"], 16, C["neg"], a3,
              ha="center", weight="bold")
    _tail(R, ax, o, u, 4, x, y, w)


def strike_list(R, ax, u, d, o):
    """A list of items struck through one per beat — "we remove all of these".

    opts: items [str], kept [str], title, kept_title, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    items = o.get("items", [])
    kept = o.get("kept", [])
    n = max(len(items), 1)
    lane = min(0.82, (h - 1.8) / n)
    lx = x + 0.6
    lw_ = w * (0.52 if kept else 0.90)

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    R.txt(ax, lx, y + h - 0.25, o.get("title", ""), 19, C["neg"], a0,
          weight="bold")
    for k, it in enumerate(items):
        t0 = R.rt(min(k, 4), 0.7 + k * 0.55, 0.5)
        a = _fade(u, t0, 0.45)
        if a <= 0.01:
            continue
        yy = y + h - 0.95 - k * lane
        R.txt(ax, lx, yy, str(it), 18, C["muted"], a)
        g = _ease((u - t0 - 0.25) / 0.55)
        if g > 0.02:
            ax.plot([lx - 0.12, lx - 0.12 + (lw_ - 0.8) * g], [yy, yy], lw=2.4,
                    color=C["neg"], alpha=min(a, 1.0) * 0.9, zorder=8)

    if kept:
        ak = _fade(u, R.rt(min(n, 4), 0.7 + n * 0.55, 0.8), 0.6)
        kx = x + w * 0.58
        R.box(ax, kx - 0.3, y + 0.9, w * 0.40, h - 1.6, a=ak * 0.5,
              fc=C["accent"], r=0.14, z=3)
        R.txt(ax, kx, y + h - 0.25, o.get("kept_title", ""), 19, C["accent"],
              ak, weight="bold")
        for k, it in enumerate(kept):
            R.txt(ax, kx, y + h - 0.95 - k * lane, str(it), 18, C["ink"], ak)
    _tail(R, ax, o, u, n + 1, x, y, w)


def boundary_spread(R, ax, u, d, o):
    """Two panels of the same classification task, each showing many decision
    boundaries fit on resampled training sets — tight versus scattered.

    The shape for any estimator-variance argument.

    opts: left {label, spread, note}, right {label, spread, note}, n_lines,
          verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    pw = w / 2 - 0.5
    rad = min(pw * 0.34, (h - 2.2) * 0.40)
    cy = y + h * 0.58
    nl = int(o.get("n_lines", 9))

    for i, key in enumerate(("left", "right")):
        spec = o.get(key, {})
        a = _fade(u, R.rt(i, 0.5 + i * 1.5, 0.7), 0.6)
        if a <= 0.01:
            continue
        cx = x + 0.5 + i * (pw + 1.0) + pw / 2
        rng = np.random.default_rng(50 + i)
        sx = float(spec.get("scale_x", 1.0))
        sy = float(spec.get("scale_y", 1.0))
        for sgn, col in ((-1, C["pos"]), (1, C["neg"])):
            for _ in range(34):
                px = rng.normal(sgn * 0.45, 0.34) * rad * 1.5 * sx
                py = rng.normal(sgn * 0.30, 0.34) * rad * 1.5 * sy
                ax.plot([cx + px], [cy + py], "o", ms=6, color=col,
                        alpha=min(a, 1.0) * 0.65, zorder=5)
        sp = float(spec.get("spread", 0.10))
        ab = _fade(u, R.rt(i, 0.5 + i * 1.5, 0.7) + 0.6, 0.6)
        for j in range(nl):
            th = np.pi / 2 + 0.75 + rng.normal(0, sp)
            L = rad * 1.5
            ax.plot([cx - L * np.cos(th), cx + L * np.cos(th)],
                    [cy - L * np.sin(th), cy + L * np.sin(th)], lw=1.8,
                    color=_col(R, "out"), alpha=min(ab, 1.0) * 0.45, zorder=7)
        R.txt(ax, cx, cy + rad * 1.55, str(spec.get("label", "")), 18,
              C["ink"], a, ha="center", weight="bold")
        if spec.get("note"):
            R.txt(ax, cx, cy - rad * 1.62, spec["note"], 16,
                  C["accent"] if i == 0 else C["neg"], ab, ha="center")
    _tail(R, ax, o, u, 3, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — added while working through Dreamer 4 (Hafner et al., 2025)
# --------------------------------------------------------------------------- #

def step_schedule(R, ax, u, d, o):
    """A generative trajectory from noise to data, drawn twice: many fine steps
    and a few coarse ones, with a bracket showing two half-steps distilled
    into one.

    The shape for step distillation, consistency models, shortcut models — any
    "learn to take bigger steps" argument.

    opts: fine_steps, coarse_steps, start_label, end_label, fine_label,
          coarse_label, distill_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    nf = int(o.get("fine_steps", 16))
    nc = int(o.get("coarse_steps", 4))
    lx, rx = x + 1.9, x + w - 1.2
    y_f = y + h * 0.30
    y_c = y + h * 0.68

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for yy, n, col, lab, beat in ((y_f, nf, C["dim"], o.get("fine_label", ""), 0),
                                  (y_c, nc, C["accent"], o.get("coarse_label", ""), 1)):
        a = _fade(u, R.rt(beat, 0.4 + beat * 1.2, 0.6), 0.55)
        if a <= 0.01:
            continue
        ax.plot([lx, rx], [yy, yy], lw=1.6, color=C["line"], alpha=min(a, 1.0),
                zorder=4)
        for k in range(n):
            x0 = lx + (rx - lx) * k / n
            x1 = lx + (rx - lx) * (k + 1) / n
            g = _fade(u, R.rt(beat, 0.4 + beat * 1.2, 0.6) + k * 0.04, 0.3)
            _arrow(ax, (x0, yy), (x1 - 0.03, yy), col,
                   3.0 if n <= 8 else 1.3, min(g, 1.0),
                   18 if n <= 8 else 7)
        R.txt(ax, lx - 0.22, yy, lab, 17, col, a, ha="right", weight="bold")

    a2 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([lx], [(y_f + y_c) / 2], "o", ms=14, color=P["txt"],
            alpha=min(a2, 1.0), zorder=8)
    ax.plot([rx], [(y_f + y_c) / 2], "o", ms=14, color=P["vis"],
            alpha=min(a2, 1.0), zorder=8)
    R.txt(ax, lx, (y_f + y_c) / 2 + 0.48, o.get("start_label", ""), 17,
          P["txt"], a2, ha="center", weight="bold")
    R.txt(ax, rx, (y_f + y_c) / 2 + 0.48, o.get("end_label", ""), 17, P["vis"],
          a2, ha="center", weight="bold")

    # two half-steps distilled into one
    a3 = _fade(u, R.rt(2, 2.8, 0.8), 0.7)
    if a3 > 0.01:
        seg = (rx - lx) / nc
        bx0 = lx + seg * 1
        bx1 = bx0 + seg
        for xx in (bx0, bx0 + seg / 2, bx1):
            ax.plot([xx, xx], [y_f - 0.16, y_c + 0.16], ls=(0, (2, 3)), lw=1.4,
                    color=C["neg"], alpha=min(a3, 1.0) * 0.7, zorder=6)
        ax.plot([bx0, bx1], [y_c + 0.42, y_c + 0.42], lw=2.4, color=C["neg"],
                alpha=min(a3, 1.0), zorder=7)
        R.txt(ax, (bx0 + bx1) / 2, y_c + 0.72, o.get("distill_label", ""), 17,
              C["neg"], a3, ha="center", weight="bold")
    _tail(R, ax, o, u, 4, x, y, w)


def noise_ladder(R, ax, u, d, o):
    """A sequence of frames each carrying its own signal level, drawn as a
    fill bar inside the frame.

    The shape for diffusion forcing, per-token noise schedules, and any
    "clean history, noisy future" generation pattern.

    opts: frames [{label, tau}], tau_label, note, sweep, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    fr = o.get("frames", [])
    n = max(len(fr), 1)
    fw = min((w - 1.6) / n, 2.2)
    bx = x + (w - fw * n) / 2
    cy = y + h * 0.58
    fh = min(1.9, h * 0.40)

    for k, f in enumerate(fr):
        a = _fade(u, R.rt(min(k, 3), 0.5 + k * 0.42, 0.5), 0.45)
        if a <= 0.01:
            continue
        tau = float(f.get("tau", 1.0))
        px = bx + k * fw
        R.box(ax, px, cy - fh / 2, fw * 0.86, fh, a=a * 0.35, fc=C["dim"],
              r=0.10, z=4)
        ax.add_patch(Rectangle((px, cy - fh / 2), fw * 0.86, fh * tau,
                               fc=P["vis"], ec="none", alpha=min(a, 1.0) * 0.85,
                               zorder=6))
        R.txt(ax, px + fw * 0.43, cy - fh / 2 - 0.34, str(f.get("label", "")),
              14, C["muted"], a, ha="center")
        R.txt(ax, px + fw * 0.43, cy + fh / 2 + 0.30, f"{tau:.1f}", 15,
              C["ink"] if tau > 0.5 else C["neg"], a, ha="center",
              weight="bold")
    R.txt(ax, bx - 0.30, cy + fh / 2 + 0.30, o.get("tau_label", r"$\tau$"), 16,
          C["muted"], _fade(u, R.rt(0, 0.5, 0.5), 0.5), ha="right")
    R.txt(ax, x + 0.25, y + 0.78, o.get("note", ""), 17, C["muted"],
          _fade(u, R.rt(3, 3.6, 0.8), 0.7))
    _tail(R, ax, o, u, 4, x, y, w)


def milestone_chain(R, ax, u, d, o):
    """A chain of dependent milestones with a success rate at each, so the
    drop-off along the chain is visible.

    The shape for long-horizon task progressions, funnels, curricula.

    opts: stages [{label, value, colour}], unit, compare [{name, values}],
          verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    st = o.get("stages", [])
    n = max(len(st), 1)
    sw = (w - 1.0) / n
    base = y + h * 0.30
    top = h * 0.44
    unit = o.get("unit", "%")

    for k, s in enumerate(st):
        t0 = R.rt(min(k, 4), 0.6 + k * 0.5, 0.5)
        a = _fade(u, t0, 0.45)
        if a <= 0.01:
            continue
        g = _ease((u - t0) / 0.7)
        v = float(s.get("value", 0))
        px = x + 0.5 + k * sw
        col = _col(R, s.get("colour", "hot" if k == n - 1 else "in"))
        ax.add_patch(Rectangle((px, base), sw * 0.62, top * (v / 100.0) * g,
                               fc=col, ec="none", alpha=min(a, 1.0), zorder=6))
        R.txt(ax, px + sw * 0.31, base - 0.34, str(s.get("label", "")), 14,
              C["muted"], a, ha="center")
        R.txt(ax, px + sw * 0.31, base + top * (v / 100.0) * g + 0.28,
              f"{v:g}{unit}", 15, C["ink"], a * g, ha="center", weight="bold")
        if k:
            _arrow(ax, (px - sw * 0.30, base - 0.72),
                   (px + sw * 0.10, base - 0.72), C["line"], 1.6, a, 10)

    for ci, cmp_ in enumerate(o.get("compare", [])):
        a = _fade(u, R.rt(min(n, 4) + ci, 0.6 + n * 0.5 + ci * 0.6, 0.6), 0.55)
        if a <= 0.01:
            continue
        vals = cmp_.get("values", [])
        pts = [(x + 0.5 + k * sw + sw * 0.31,
                base + top * (float(v) / 100.0)) for k, v in enumerate(vals)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=2.4,
                ls=(0, (4, 3)), color=C["dim"], alpha=min(a, 1.0), zorder=8)
        R.txt(ax, pts[-1][0] + 0.15, pts[-1][1], str(cmp_.get("name", "")), 15,
              C["muted"], a)
    _tail(R, ax, o, u, min(n + 1, 6), x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — qualitative positioning and architecture internals
# --------------------------------------------------------------------------- #

def quadrant_map(R, ax, u, d, o):
    """Named approaches placed on two qualitative axes, with the desirable
    corner marked. Use this instead of a bar chart when the comparison is
    real but the numbers are not — it states a position without inventing
    precision.

    opts: xlabel, ylabel, lo_x, hi_x, lo_y, hi_y, items [{label, x, y, best}],
          goal_label, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    px0, py0 = x + 2.1, y + 1.5
    pw, ph = w - 3.0, h - 2.6

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.add_patch(Rectangle((px0 + pw / 2, py0 + ph / 2), pw / 2, ph / 2,
                           fc=C["accent"], ec="none", alpha=min(a0, 1.0) * 0.10,
                           zorder=3))
    _arrow(ax, (px0, py0), (px0 + pw, py0), C["line"], 1.8, a0, 14)
    _arrow(ax, (px0, py0), (px0, py0 + ph), C["line"], 1.8, a0, 14)
    R.txt(ax, px0 + pw / 2, y + 0.55, o.get("xlabel", ""), 17, C["ink"], a0,
          ha="center")
    R.txt(ax, x + 0.55, py0 + ph / 2, o.get("ylabel", ""), 17, C["ink"], a0,
          ha="center", va="center", rot=90)
    R.txt(ax, px0 + 0.10, py0 - 0.34, o.get("lo_x", ""), 14, C["muted"], a0)
    R.txt(ax, px0 + pw, py0 - 0.34, o.get("hi_x", ""), 14, C["muted"], a0,
          ha="right")
    R.txt(ax, px0 + 0.12, py0 + 0.18, o.get("lo_y", ""), 14, C["muted"], a0)
    R.txt(ax, px0 + 0.12, py0 + ph - 0.18, o.get("hi_y", ""), 14, C["muted"], a0)
    if o.get("goal_label"):
        R.txt(ax, px0 + pw - 0.15, py0 + ph - 0.28, o["goal_label"], 15,
              C["accent"], a0, ha="right", weight="bold")

    for k, it in enumerate(o.get("items", [])):
        a = _fade(u, R.rt(min(k + 1, 4), 0.9 + k * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        best = bool(it.get("best"))
        cx = px0 + float(it.get("x", 0.5)) * pw
        cy = py0 + float(it.get("y", 0.5)) * ph
        ax.plot([cx], [cy], "o", ms=20 if best else 14,
                color=C["accent"] if best else C["dim"], alpha=min(a, 1.0),
                zorder=8)
        R.txt(ax, cx, cy + (0.52 if best else 0.44), str(it.get("label", "")),
              17 if best else 16, C["ink"] if best else C["muted"], a,
              ha="center", weight="bold" if best else "normal")
        if it.get("note"):
            R.txt(ax, cx, cy - 0.50, it["note"], 14, C["muted"], a, ha="center")
    _tail(R, ax, o, u, 5, x, y, w)


def bottleneck_codec(R, ax, u, d, o):
    """An encoder-decoder over a patch grid with a narrow latent bottleneck,
    showing dropped-out patches on the way in and reconstruction on the way
    out.

    The shape for tokenizers, autoencoders, VQ/continuous bottlenecks, MAE.

    opts: grid, drop, n_latents, in_label, out_label, latent_label,
          bottleneck_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("grid", 6))
    drop = float(o.get("drop", 0.45))
    cell = min((w * 0.16) / n, (h - 2.6) / n)
    cy = y + h * 0.58
    rng = np.random.default_rng(int(o.get("seed", 71)))
    mask = rng.random((n, n)) < drop

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    gx = x + 0.5
    gy = cy + n * cell / 2
    for i in range(n):
        for j in range(n):
            m = mask[i, j]
            ax.add_patch(Rectangle((gx + j * cell, gy - (i + 1) * cell),
                                   cell * 0.88, cell * 0.88,
                                   fc=C["dim"] if m else P["vis"], ec="none",
                                   alpha=min(a0, 1.0) * (0.35 if m else 0.85),
                                   zorder=6))
    R.txt(ax, gx + n * cell / 2, gy + 0.34, o.get("in_label", ""), 16,
          C["muted"], a0, ha="center")
    R.txt(ax, gx + n * cell / 2, gy - n * cell - 0.38,
          f"{int(drop * 100)}% patches dropped", 14, C["neg"], a0, ha="center")

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    ex = gx + n * cell + 0.7
    R.box(ax, ex, cy - 1.15, 1.9, 2.3, a=a1, fc=C["panel"], r=0.14, z=7)
    R.txt(ax, ex + 0.95, cy, "encoder", 16, C["ink"], a1, ha="center", z=9)
    nl = int(o.get("n_latents", 5))
    lx = ex + 2.3
    for k in range(nl):
        yy = cy + (nl - 1) / 2 * 0.42 - k * 0.42
        ax.add_patch(Rectangle((lx, yy - 0.16), 0.55, 0.32, fc=P["txt"],
                               ec="none", alpha=min(a1, 1.0), zorder=7))
    R.txt(ax, lx + 0.28, cy + nl * 0.24 + 0.28, o.get("latent_label", ""), 15,
          P["txt"], a1, ha="center")

    a2 = _fade(u, R.rt(2, 2.5, 0.7), 0.6)
    bx = lx + 1.0
    ax.add_patch(Rectangle((bx, cy - 0.30), 0.42, 0.60, fc=C["accent"],
                           ec="none", alpha=min(a2, 1.0), zorder=7))
    _arrow(ax, (lx + 0.62, cy), (bx - 0.04, cy), C["muted"], 1.8, a2)
    R.txt(ax, bx + 0.21, cy - 0.62, o.get("bottleneck_label", ""), 15,
          C["accent"], a2, ha="center", weight="bold")

    a3 = _fade(u, R.rt(3, 3.6, 0.7), 0.6)
    dx = bx + 1.0
    R.box(ax, dx, cy - 1.15, 1.9, 2.3, a=a3, fc=C["panel"], r=0.14, z=7)
    R.txt(ax, dx + 0.95, cy, "decoder", 16, C["ink"], a3, ha="center", z=9)
    _arrow(ax, (bx + 0.48, cy), (dx - 0.04, cy), C["muted"], 1.8, a3)
    ox = dx + 2.3
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((ox + j * cell, gy - (i + 1) * cell),
                                   cell * 0.88, cell * 0.88, fc=P["vis"],
                                   ec="none", alpha=min(a3, 1.0) * 0.85,
                                   zorder=6))
    _arrow(ax, (dx + 1.94, cy), (ox - 0.10, cy), C["muted"], 1.8, a3)
    R.txt(ax, ox + n * cell / 2, gy + 0.34, o.get("out_label", ""), 16,
          C["muted"], a3, ha="center")
    R.txt(ax, x + 0.3, y + 0.78, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.6, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


def factorized_attention(R, ax, u, d, o):
    """A time-by-space token grid where space attention runs within a column
    and time attention across a row, plus the layer stack showing how often
    each is applied.

    The shape for axial or factorised attention, and for any "we only do the
    expensive operation every N layers" argument.

    opts: times, spaces, every, n_layers, space_label, time_label,
          layer_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    T = int(o.get("times", 6))
    S = int(o.get("spaces", 4))
    cell = min((w * 0.52) / T, (h - 2.6) / S)
    gx, gy = x + 1.6, y + h - 1.2

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for t in range(T):
        for s in range(S):
            ax.add_patch(Rectangle((gx + t * cell, gy - (s + 1) * cell),
                                   cell * 0.84, cell * 0.84, fc=C["dim"],
                                   ec="none", alpha=min(a0, 1.0) * 0.35,
                                   zorder=5))
    R.txt(ax, gx + T * cell / 2, gy + 0.34, "time  \u2192", 16, C["muted"],
          a0, ha="center")
    R.txt(ax, gx - 0.28, gy - S * cell / 2, "space", 16, C["muted"], a0,
          ha="center", va="center", rot=90)

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    tcol = 2
    for s in range(S):
        ax.add_patch(Rectangle((gx + tcol * cell, gy - (s + 1) * cell),
                               cell * 0.84, cell * 0.84, fc=P["vis"], ec="none",
                               alpha=min(a1, 1.0) * 0.9, zorder=6))
    R.txt(ax, gx + (tcol + 0.42) * cell, gy + 0.92,
          o.get("space_label", ""), 16, P["vis"], a1, ha="center",
          weight="bold")

    a2 = _fade(u, R.rt(2, 2.5, 0.7), 0.6)
    srow = 1
    for t in range(T):
        ax.add_patch(Rectangle((gx + t * cell, gy - (srow + 1) * cell),
                               cell * 0.84, cell * 0.84, fc=C["accent"],
                               ec="none", alpha=min(a2, 1.0) * 0.9, zorder=7))
    R.txt(ax, gx + T * cell + 0.22, gy - (srow + 0.5) * cell,
          o.get("time_label", ""), 16, C["accent"], a2, weight="bold")

    # the layer stack: which layers get the expensive temporal attention
    a3 = _fade(u, R.rt(3, 3.6, 0.8), 0.7)
    nl = int(o.get("n_layers", 12))
    every = int(o.get("every", 4))
    sx = x + w - 2.3
    lh = min(0.34, (h - 2.4) / nl)
    for k in range(nl):
        temporal = (k + 1) % every == 0
        yy = gy - k * lh
        ax.add_patch(Rectangle((sx, yy - lh * 0.38), 1.5, lh * 0.72,
                               fc=C["accent"] if temporal else P["vis"],
                               ec="none",
                               alpha=min(a3, 1.0) * (0.95 if temporal else 0.4),
                               zorder=7))
    R.txt(ax, sx + 0.75, gy + 0.34, o.get("layer_label", ""), 15, C["muted"],
          a3, ha="center")
    R.txt(ax, x + w / 2, y + 0.80, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.8, 0.8), 0.7), ha="center")
    _tail(R, ax, o, u, 5, x, y, w)


def modality_attention(R, ax, u, d, o):
    """Named modalities with a directed who-attends-to-whom matrix, so a
    deliberately one-way edge is visible.

    opts: modalities [str], allowed [[from, to], ...] or matrix [[0/1]],
          row_label, col_label, highlight_row, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    mods = o.get("modalities", [])
    n = max(len(mods), 1)
    mat = o.get("matrix")
    if not mat:
        allowed = {tuple(p) for p in o.get("allowed", [])}
        mat = [[1 if (i, j) in allowed else 0 for j in range(n)]
               for i in range(n)]
    cell = min((w * 0.42) / n, (h - 3.0) / n)
    gx = x + w * 0.42
    gy = y + h - 1.5

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for j, m in enumerate(mods):
        R.txt(ax, gx + (j + 0.5) * cell, gy + 0.30, str(m), 14, C["muted"], a0,
              ha="center", rot=32)
    for i, m in enumerate(mods):
        R.txt(ax, gx - 0.22, gy - (i + 0.5) * cell, str(m), 15, C["ink"], a0,
              ha="right")
    R.txt(ax, gx + n * cell / 2, gy + 1.05, o.get("col_label", "attends to"),
          16, C["muted"], a0, ha="center")
    R.txt(ax, gx - 2.6, gy - n * cell / 2, o.get("row_label", "attends from"),
          16, C["muted"], a0, ha="center", va="center", rot=90)

    hl = o.get("highlight_row")
    for i in range(n):
        a = _fade(u, R.rt(min(i, 3), 0.6 + i * 0.7, 0.5), 0.45)
        if a <= 0.01:
            continue
        for j in range(n):
            on = bool(mat[i][j]) if i < len(mat) and j < len(mat[i]) else False
            ax.add_patch(Rectangle((gx + j * cell, gy - (i + 1) * cell),
                                   cell * 0.86, cell * 0.86,
                                   fc=C["accent"] if on else C["dim"],
                                   ec="none",
                                   alpha=min(a, 1.0) * (0.9 if on else 0.22),
                                   zorder=6))
        if hl is not None and i == int(hl):
            ax.add_patch(Rectangle((gx - 0.06, gy - (i + 1) * cell - 0.06),
                                   n * cell + 0.12, cell * 0.86 + 0.12,
                                   fill=False, ec=C["ink"], lw=2.4,
                                   alpha=min(a, 1.0), zorder=9))
    _tail(R, ax, o, u, n, x, y, w)


def interpolation_path(R, ax, u, d, o):
    """A straight path from a noise cloud to a data cloud with intermediate
    points labelled by their mixing coefficient, and a velocity arrow.

    The shape for flow matching, interpolants, and any linear-schedule story.

    opts: start_label, end_label, marks [float], velocity_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cy = y + h * 0.58
    lx, rx = x + 2.2, x + w - 2.2
    rng = np.random.default_rng(int(o.get("seed", 83)))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for px, py in [(lx + rng.normal(0, .55), cy + rng.normal(0, .85))
                   for _ in range(60)]:
        ax.plot([px], [py], "o", ms=6, color=P["txt"], alpha=min(a0, 1.) * .5,
                zorder=5)
    R.txt(ax, lx, cy - 1.65, o.get("start_label", ""), 17, P["txt"], a0,
          ha="center", weight="bold")

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    th = rng.uniform(0, 2 * np.pi, 60)
    rr = rng.uniform(0.55, 0.95, 60)
    for k in range(60):
        ax.plot([rx + rr[k] * np.cos(th[k]) * 0.85],
                [cy + rr[k] * np.sin(th[k]) * 1.05], "o", ms=6, color=P["vis"],
                alpha=min(a1, 1.) * .6, zorder=5)
    R.txt(ax, rx, cy - 1.65, o.get("end_label", ""), 17, P["vis"], a1,
          ha="center", weight="bold")

    a2 = _fade(u, R.rt(2, 2.5, 0.7), 0.6)
    ax.plot([lx, rx], [cy, cy], lw=2.6, color=C["line"], alpha=min(a2, 1.0),
            zorder=6)
    for k, m in enumerate(o.get("marks", [0.0, 0.25, 0.5, 0.75, 1.0])):
        mx = lx + (rx - lx) * float(m)
        g = _fade(u, R.rt(2, 2.5, 0.7) + k * 0.12, 0.4)
        ax.plot([mx], [cy], "o", ms=13, color=C["accent"], alpha=min(g, 1.0),
                zorder=8)
        R.txt(ax, mx, cy + 0.44, f"{float(m):g}", 14, C["accent"], g,
              ha="center")
    R.txt(ax, (lx + rx) / 2, cy + 1.05, o.get("tau_label", r"$\tau$"), 16,
          C["muted"], a2, ha="center")

    a3 = _fade(u, R.rt(3, 3.8, 0.8), 0.7)
    mid = lx + (rx - lx) * 0.45
    _arrow(ax, (mid, cy - 0.75), (mid + 1.5, cy - 0.75), C["neg"], 2.8, a3, 16)
    R.txt(ax, mid + 0.75, cy - 1.12, o.get("velocity_label", ""), 16, C["neg"],
          a3, ha="center", weight="bold")
    R.txt(ax, (lx + rx) / 2, cy + 1.48, o.get("note", ""), 17, C["muted"],
          _fade(u, R.rt(4, 4.8, 0.8), 0.7), ha="center")
    _tail(R, ax, o, u, 5, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — attention mechanics, animated rather than boxed
# --------------------------------------------------------------------------- #

def attention_lookup(R, ax, u, d, o):
    """One query attending over a row of keys: similarity scores grow, softmax
    normalises them into weights, and the output is assembled as a weighted
    blend of the values.

    This is attention as an operation rather than as a box. Also fits any
    soft-lookup, retrieval, or kernel-smoothing story.

    opts: tokens [str], query (index), scores [float], value_label,
          output_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    toks = o.get("tokens", [])
    n = max(len(toks), 1)
    q = int(o.get("query", n - 1))
    raw = [float(v) for v in o.get("scores", [1.0] * n)]
    e = np.exp(np.array(raw) - max(raw))
    wts = e / e.sum()

    cw = min((w - 4.6) / n, 1.9)
    gx = x + 0.6
    ky = y + h * 0.72          # key row
    oy = y + h * 0.30          # output

    # keys, with the query highlighted
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for k, t in enumerate(toks):
        isq = k == q
        R.box(ax, gx + k * cw, ky - 0.34, cw * 0.86, 0.68, a=a0,
              fc=C["accent"] if isq else P["vis"], r=0.10, z=7)
        R.txt(ax, gx + k * cw + cw * 0.43, ky, str(t), 14, "#FFFFFF", a0,
              ha="center", z=9)
    R.txt(ax, gx, ky + 0.62, o.get("key_label", "keys / values"), 16,
          C["muted"], a0)
    R.txt(ax, gx + q * cw + cw * 0.43, ky - 0.68,
          o.get("query_label", "query"), 15, C["accent"], a0, ha="center",
          weight="bold")

    # raw similarity scores
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    mx = max(abs(v) for v in raw) or 1.0
    for k in range(n):
        g = _ease((u - R.rt(1, 1.3, 0.6) - k * 0.05) / 0.6)
        hh = 0.85 * (raw[k] / mx) * max(g, 0)
        ax.add_patch(Rectangle((gx + k * cw + cw * 0.20, ky + 0.44),
                               cw * 0.46, max(hh, 0.01), fc=C["dim"],
                               ec="none", alpha=min(a1, 1.0), zorder=6))
    R.txt(ax, gx + n * cw + 0.25, ky + 0.70, r"$q\cdot k_i/\sqrt{d_k}$", 16,
          C["muted"], a1)

    # softmax weights, drawn as arrow thickness into the output
    a2 = _fade(u, R.rt(2, 2.6, 0.7), 0.7)
    ox = gx + n * cw / 2
    if a2 > 0.01:
        for k in range(n):
            lwid = 0.6 + 5.4 * float(wts[k])
            _arrow(ax, (gx + k * cw + cw * 0.43, ky - 0.40), (ox, oy + 0.42),
                   C["accent"], lwid, min(a2, 1.0) * (0.35 + 0.65 * wts[k]),
                   10 + 8 * wts[k])
            R.txt(ax, gx + k * cw + cw * 0.43, ky - 0.98,
                  f"{wts[k]:.2f}", 13,
                  C["ink"] if wts[k] > 0.2 else C["muted"],
                  min(a2, 1.0), ha="center",
                  weight="bold" if wts[k] > 0.2 else "normal")
        R.txt(ax, ox + n * cw / 2 + 0.1, (ky + oy) / 2,
              o.get("softmax_label", "softmax"), 16, C["accent"], a2)

    a3 = _fade(u, R.rt(3, 3.8, 0.7), 0.7)
    R.box(ax, ox - 2.6, oy - 0.44, 5.2, 0.88, a=a3, fc=P["txt"], r=0.12, z=7)
    R.txt(ax, ox, oy, o.get("output_label", ""), 15, "#FFFFFF", a3,
          ha="center", z=9)
    R.txt(ax, ox, oy - 0.92, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.8, 0.8), 0.7), ha="center")
    _tail(R, ax, o, u, 5, x, y, w)


def head_patterns(R, ax, u, d, o):
    """Several attention heads over the same sentence, each revealed in turn,
    so that different heads attending to different things is visible rather
    than asserted.

    opts: tokens [str], heads [{name, weights [float]}], verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    toks = o.get("tokens", [])
    heads = o.get("heads", [])
    n = max(len(toks), 1)
    m = max(len(heads), 1)
    cw = min((w - 3.6) / n, 1.8)
    gx = x + 3.0
    top = y + h - 0.9
    lane = min(0.86, (h - 1.9) / m)

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for k, t in enumerate(toks):
        R.txt(ax, gx + k * cw + cw * 0.43, top + 0.30, str(t), 14, C["ink"],
              a0, ha="center")

    for i, hd in enumerate(heads):
        a = _fade(u, R.rt(min(i, 4), 0.6 + i * 0.75, 0.5), 0.45)
        if a <= 0.01:
            continue
        yy = top - 0.35 - i * lane
        wts = [float(v) for v in hd.get("weights", [])]
        s = sum(wts) or 1.0
        col = _col(R, hd.get("colour", ["hot", "in", "out", "pos", "neg"][i % 5]))
        for k in range(min(n, len(wts))):
            g = _ease((u - R.rt(min(i, 4), 0.6 + i * 0.75, 0.5) - k * 0.04) / 0.5)
            v = wts[k] / s
            ax.add_patch(Rectangle((gx + k * cw + cw * 0.14, yy - lane * 0.30),
                                   cw * 0.58, lane * 0.60 * v / max(
                                       max(wts) / s, 1e-6) * max(g, 0),
                                   fc=col, ec="none", alpha=min(a, 1.0),
                                   zorder=6))
        R.txt(ax, gx - 0.25, yy, str(hd.get("name", "")), 15, col, a,
              ha="right", weight="bold")
    _tail(R, ax, o, u, m, x, y, w)


def sequential_vs_parallel(R, ax, u, d, o):
    """The same sequence processed two ways: a recurrent chain that advances
    one step at a time, and an all-at-once layer where every position is
    computed simultaneously.

    opts: tokens [str], top_label, bottom_label, top_note, bottom_note,
          rate, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    toks = o.get("tokens", [])
    n = max(len(toks), 1)
    cw = min((w - 3.4) / n, 1.8)
    gx = x + 3.0
    y_r = y + h * 0.74
    y_t = y + h * 0.32
    rate = float(o.get("rate", 0.55))

    # recurrent: a cursor sweeping left to right
    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    t_go = R.rt(1, 1.4, 0.7)
    cur = int(max(0.0, u - t_go) / rate)
    for k, t in enumerate(toks):
        done = k < cur
        R.box(ax, gx + k * cw, y_r - 0.32, cw * 0.82, 0.64, a=a0,
              fc=P["vis"] if done else C["dim"], r=0.10, z=7)
        R.txt(ax, gx + k * cw + cw * 0.41, y_r, str(t), 13,
              "#FFFFFF" if done else C["muted"], a0, ha="center", z=9)
        if k and k <= cur:
            _arrow(ax, (gx + (k - 1) * cw + cw * 0.86, y_r),
                   (gx + k * cw - 0.03, y_r), C["accent"], 2.2, 1.0, 12)
    R.txt(ax, gx - 0.25, y_r, o.get("top_label", ""), 16, C["ink"], a0,
          ha="right", weight="bold")
    R.txt(ax, gx - 0.25, y_r - 0.46, o.get("top_note", ""), 14, C["muted"], a0,
          ha="right")

    # attention: every position at once
    a1 = _fade(u, R.rt(2, 2.8, 0.7), 0.6)
    for k, t in enumerate(toks):
        R.box(ax, gx + k * cw, y_t - 0.32, cw * 0.82, 0.64, a=a1, fc=P["vis"],
              r=0.10, z=7)
        R.txt(ax, gx + k * cw + cw * 0.41, y_t, str(t), 13, "#FFFFFF", a1,
              ha="center", z=9)
    if a1 > 0.01:
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                t0 = np.linspace(0, 1, 16)
                xa = gx + i * cw + cw * 0.41
                xb = gx + j * cw + cw * 0.41
                ax.plot((1 - t0) * xa + t0 * xb,
                        y_t + 0.36 + np.sin(np.pi * t0) * 0.30,
                        lw=1.2, color=C["accent"], alpha=min(a1, 1.0) * 0.30,
                        zorder=5)
    R.txt(ax, gx - 0.25, y_t, o.get("bottom_label", ""), 16, C["ink"], a1,
          ha="right", weight="bold")
    R.txt(ax, gx - 0.25, y_t - 0.46, o.get("bottom_note", ""), 14, C["muted"],
          a1, ha="right")
    _tail(R, ax, o, u, 3, x, y, w)


def score_sharpening(R, ax, u, d, o):
    """The same logits pushed through softmax at two scales, showing one
    distribution saturating into a spike while the other stays informative.

    The shape for temperature, scaling factors, and any saturation argument.

    opts: scores [float], left {label, scale, note}, right {label, scale, note},
          tokens [str], verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    raw = np.array([float(v) for v in o.get("scores", [3.0, 1.2, 0.4, 2.1, 0.1])])
    toks = o.get("tokens", [])
    n = len(raw)
    pw = w / 2 - 0.9

    for i, key in enumerate(("left", "right")):
        spec = o.get(key, {})
        a = _fade(u, R.rt(i, 0.5 + i * 1.4, 0.7), 0.6)
        if a <= 0.01:
            continue
        sc = float(spec.get("scale", 1.0))
        e = np.exp(raw * sc - (raw * sc).max())
        p = e / e.sum()
        bx = x + 0.7 + i * (pw + 1.0)
        by = y + 1.5
        bw = pw / max(n, 1)
        col = C["neg"] if i == 0 else C["accent"]
        for k in range(n):
            g = _ease((u - R.rt(i, 0.5 + i * 1.4, 0.7) - k * 0.06) / 0.7)
            ax.add_patch(Rectangle((bx + k * bw, by), bw * 0.62,
                                   (h - 3.0) * p[k] * max(g, 0), fc=col,
                                   ec="none", alpha=min(a, 1.0), zorder=6))
            if k < len(toks):
                R.txt(ax, bx + k * bw + bw * 0.31, by - 0.32, str(toks[k]), 13,
                      C["muted"], a, ha="center")
        R.txt(ax, bx + pw / 2, y + h - 0.5, str(spec.get("label", "")), 18,
              C["ink"], a, ha="center", weight="bold")
        if spec.get("note"):
            R.txt(ax, bx + pw / 2, y + h - 0.95, spec["note"], 15, col, a,
                  ha="center")
    _tail(R, ax, o, u, 3, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — response curves, residual dataflow, tensor shapes
# --------------------------------------------------------------------------- #

def saturation_curve(R, ax, u, d, o):
    """A squashing response curve with operating points marked and their
    tangents drawn, so a flat region reads directly as a dead gradient.

    The shape for saturation arguments: softmax temperature, sigmoid/tanh
    saturation, clipping, any "we scale to stay in the responsive region".

    opts: xlabel, ylabel, points [{label, x, colour, note}], slope_label,
          curve ("logistic"|"tanh"), note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    px0, py0 = x + 1.7, y + 1.4
    pw, ph = w - 2.6, h - 2.8
    kind = o.get("curve", "logistic")
    lim = float(o.get("xlim", 8.0))
    xs = np.linspace(-lim, lim, 300)
    f = (lambda t: 1 / (1 + np.exp(-t))) if kind == "logistic" else np.tanh
    df = ((lambda t: f(t) * (1 - f(t))) if kind == "logistic"
          else (lambda t: 1 - np.tanh(t) ** 2))
    ys = f(xs)
    lo, hi = (0.0, 1.0) if kind == "logistic" else (-1.0, 1.0)

    def PX(v):
        return px0 + (v + lim) / (2 * lim) * pw

    def PY(v):
        return py0 + (v - lo) / (hi - lo) * ph

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot(PX(xs), PY(ys), lw=3.0, color=C["ink"], alpha=min(a0, 1.0), zorder=6)
    R.txt(ax, px0 + pw / 2, y + 0.55, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.95, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)

    # the flat tails, shaded so "no slope here" is visible before any point lands
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    for x0, x1 in ((-lim, -3.2), (3.2, lim)):
        ax.add_patch(Rectangle((PX(x0), py0), PX(x1) - PX(x0), ph, fc=C["neg"],
                               ec="none", alpha=min(a1, 1.0) * 0.10, zorder=3))
    R.txt(ax, PX(-lim * 0.70), py0 + 0.22, o.get("flat_label", ""), 14,
          C["neg"], a1, ha="center")
    R.txt(ax, PX(lim * 0.70), py0 + ph - 0.24, o.get("flat_label", ""), 14,
          C["neg"], a1, ha="center")

    for k, p in enumerate(o.get("points", [])):
        a = _fade(u, R.rt(2 + k, 2.6 + k * 1.3, 0.7), 0.6)
        if a <= 0.01:
            continue
        vx = float(p.get("x", 0.0))
        vy = f(vx)
        sl = float(df(vx))
        col = _col(R, p.get("colour", "hot"))
        # tangent, length fixed in x so the visual slope is the real slope
        t = np.array([-1.9, 1.9])
        ax.plot(PX(vx + t), PY(vy + sl * t), lw=2.6, color=col,
                alpha=min(a, 1.0), zorder=8)
        ax.plot([PX(vx)], [PY(vy)], "o", ms=16, color=col, alpha=min(a, 1.0),
                zorder=9)
        side = -1.0 if vx > 0 else 1.0          # label away from the curve
        lx = PX(vx) + side * 1.9
        R.txt(ax, lx, PY(vy) + 0.92, str(p.get("label", "")), 17, col, a,
              ha="center", weight="bold")
        R.txt(ax, lx, PY(vy) + 0.50,
              f"{o.get('slope_label', 'slope')} = {sl:.3f}", 15, col, a,
              ha="center")
        if p.get("note"):
            R.txt(ax, PX(vx), py0 + 0.62, p["note"], 14, C["muted"], a,
                  ha="center")
    R.txt(ax, x + 0.3, y + 0.78, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 5.0, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


def residual_stack(R, ax, u, d, o):
    """A vertical dataflow through sublayers with the skip connection drawn as
    an explicit arc around each one, an addition node, and a norm.

    The shape for residual blocks generally — and, with `memory`, for any
    block that also reads from a side input.

    opts: title, repeat_label, sublayers [{label, colour, from_memory}],
          input_label, output_label, memory_label, norm_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    subs = o.get("sublayers", [])
    n = max(len(subs), 1)
    cx = x + w * (0.58 if o.get("memory_label") else 0.50)
    bw = min(w * 0.46, 7.6)
    lane = (h - 2.7) / n
    bot = y + 0.9

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, cx, bot - 0.42, o.get("input_label", ""), 16, C["muted"], a0,
          ha="center")

    mem_x = cx - bw / 2 - 1.9
    if o.get("memory_label"):
        am = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
        R.box(ax, mem_x - 0.75, bot, 1.5, h - 2.7, a=am * 0.5, fc=P["vis"],
              r=0.14, z=3)
        R.txt(ax, mem_x, bot + (h - 2.7) / 2, o["memory_label"], 15, C["ink"],
              am, ha="center", rot=90)

    for k, sl in enumerate(subs):
        a = _fade(u, R.rt(k + 1, 0.9 + k * 1.0, 0.6), 0.55)
        if a <= 0.01:
            continue
        cy = bot + 0.55 + k * lane
        col = _col(R, sl.get("colour", "hot"))
        R.box(ax, cx - bw / 2, cy - 0.34, bw, 0.68, a=a, fc=col, r=0.11, z=7)
        R.txt(ax, cx, cy, str(sl.get("label", "")), 15, "#FFFFFF", a,
              ha="center", z=9)
        # skip arc around this sublayer
        sx = cx + bw / 2 + 0.55
        ax.plot([cx, sx, sx, cx], [cy - 0.62, cy - 0.62, cy + 0.70, cy + 0.70],
                lw=2.0, color=C["muted"], alpha=min(a, 1.0) * 0.9, zorder=6)
        ax.plot([cx], [cy + 0.70], "o", ms=13, fillstyle="none", mew=2.0,
                color=C["muted"], alpha=min(a, 1.0), zorder=8)
        R.txt(ax, cx, cy + 0.70, "+", 16, C["muted"], a, ha="center", z=9)
        R.txt(ax, sx + 0.20, cy, "skip", 13, C["muted"], a)
        R.txt(ax, cx - bw / 2 - 0.25, cy + 0.70, o.get("norm_label", "norm"),
              13, C["muted"], a, ha="right")
        _arrow(ax, (cx, cy - 0.98), (cx, cy - 0.40), C["muted"], 1.8, a)
        if sl.get("from_memory") and o.get("memory_label"):
            _arrow(ax, (mem_x + 0.80, cy), (cx - bw / 2 - 0.04, cy),
                   P["vis"], 2.4, a, 14)

    a2 = _fade(u, R.rt(n + 1, 0.9 + n * 1.0, 0.7), 0.6)
    top = bot + 0.55 + (n - 1) * lane + 1.0
    _arrow(ax, (cx, top), (cx, top + 0.55), C["muted"], 2.0, a2)
    R.txt(ax, cx, top + 0.82, o.get("output_label", ""), 16, C["muted"], a2,
          ha="center")
    if o.get("repeat_label"):
        R.txt(ax, cx + bw / 2 + 1.5, bot + (h - 2.7) / 2, o["repeat_label"], 24,
              C["accent"], a2, weight="bold")
    _tail(R, ax, o, u, n + 2, x, y, w)


def matmul_chain(R, ax, u, d, o):
    """A chain of tensors drawn as rectangles whose proportions match their
    shapes, with the operator between each pair.

    Far more informative than named boxes when the point is what the shapes
    do — attention, projections, reshapes, head splitting.

    opts: items [{label, shape [rows, cols], colour, note}], ops [str],
          unit, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    items = o.get("items", [])
    ops = o.get("ops", [])
    if not items:
        return
    unit = float(o.get("unit", 0.0)) or None
    mr = max(float(i["shape"][0]) for i in items)
    mc = max(float(i["shape"][1]) for i in items)
    total_c = sum(float(i["shape"][1]) for i in items)
    gap = 1.35
    avail_w = w - 1.2 - gap * (len(items) - 1)
    avail_h = h - 2.6
    if unit is None:
        unit = min(avail_w / max(total_c, 1e-6), avail_h / max(mr, 1e-6))
    cy = y + h * 0.56
    cx = x + 0.6

    for k, it in enumerate(items):
        a = _fade(u, R.rt(min(k, 4), 0.5 + k * 0.7, 0.55), 0.5)
        r_, c_ = float(it["shape"][0]) * unit, float(it["shape"][1]) * unit
        if a > 0.01:
            col = _col(R, it.get("colour", "in"))
            ax.add_patch(Rectangle((cx, cy - r_ / 2), c_, r_, fc=col, ec="none",
                                   alpha=min(a, 1.0) * 0.85, zorder=6))
            R.txt(ax, cx + c_ / 2, cy + r_ / 2 + 0.34, str(it.get("label", "")),
                  16, C["ink"], a, ha="center", weight="bold")
            R.txt(ax, cx + c_ / 2, cy - r_ / 2 - 0.36,
                  it.get("dims", f"{it['shape'][0]:g} x {it['shape'][1]:g}"),
                  14, C["muted"], a, ha="center")
            if it.get("note"):
                R.txt(ax, cx + c_ / 2, cy - r_ / 2 - 0.76, it["note"], 13,
                      C["muted"], a, ha="center")
        if k < len(items) - 1:
            ao = _fade(u, R.rt(min(k, 4), 0.5 + k * 0.7, 0.55) + 0.3, 0.4)
            R.txt(ax, cx + c_ + gap / 2, cy,
                  ops[k] if k < len(ops) else r"$\times$", 22, C["accent"], ao,
                  ha="center", weight="bold")
        cx += c_ + gap
    R.txt(ax, x + 0.3, y + 0.78, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.6, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — experimental design, RL internals, probing
# --------------------------------------------------------------------------- #

def design_matrix(R, ax, u, d, o):
    """A factorial study design: conditions crossed on two axes, one labelled
    cell per combination.

    The shape for "we ran every combination of X and Y" — ablation grids of
    conditions, benchmark suites, dataset x method matrices.

    opts: rows [str], cols [str], cells [[str]], row_label, col_label,
          highlight [r, c], note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rows = o.get("rows", [])
    cols = o.get("cols", [])
    nr, nc = max(len(rows), 1), max(len(cols), 1)
    cells = o.get("cells", [])
    gx = x + 3.0
    gy = y + h - 1.25
    cw = (w - 3.6) / nc
    ch = (h - 2.4) / nr

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for c, lab in enumerate(cols):
        R.txt(ax, gx + (c + 0.5) * cw, gy + 0.30, str(lab), 16, C["ink"], a0,
              ha="center", weight="bold")
    for r_, lab in enumerate(rows):
        R.txt(ax, gx - 0.28, gy - (r_ + 0.5) * ch, str(lab), 16, C["ink"], a0,
              ha="right", weight="bold")
    R.txt(ax, gx + nc * cw / 2, gy + 0.86, o.get("col_label", ""), 16,
          C["muted"], a0, ha="center")
    R.txt(ax, x + 0.55, gy - nr * ch / 2, o.get("row_label", ""), 16,
          C["muted"], a0, ha="center", va="center", rot=90)

    hl = o.get("highlight")
    for r_ in range(nr):
        for c in range(nc):
            k = r_ * nc + c
            a = _fade(u, R.rt(min(k, 4), 0.6 + k * 0.5, 0.5), 0.45)
            if a <= 0.01:
                continue
            hot = hl and r_ == int(hl[0]) and c == int(hl[1])
            bx = gx + c * cw
            by = gy - (r_ + 1) * ch
            R.box(ax, bx + 0.10, by + 0.12, cw - 0.24, ch - 0.26,
                  a=a * (0.85 if hot else 0.42),
                  fc=C["accent"] if hot else C["panel"], r=0.12, z=5)
            icons = o.get("icons", [])
            ic = (icons[r_][c] if r_ < len(icons) and c < len(icons[r_])
                  else None)
            if ic:
                _mini_stack(ax, bx + cw / 2, by + ch * 0.46,
                            min(cw, ch) * 0.30, int(ic.get("blocks", 3)),
                            float(ic.get("offset", 0.8)),
                            ic.get("displaced", "top"), a,
                            hot=C["accent"], centre=True, ink=C["ink"])
            txt = ""
            if r_ < len(cells) and c < len(cells[r_]):
                txt = str(cells[r_][c])
            import textwrap as _tw
            width = max(10, int((cw - 0.5) / 0.16))
            for j, ln in enumerate(_tw.wrap(txt, width)[:3]):
                ty = (by + 0.30 - j * 0.34) if ic else \
                     (by + ch / 2 + 0.22 - j * 0.38)
                R.txt(ax, bx + cw / 2, ty, ln, 13,
                      "#FFFFFF" if hot else C["ink"], a, ha="center", z=8)
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.4, 0.8), 0.7))
    _tail(R, ax, o, u, min(nr * nc, 5), x, y, w)


def object_stack(R, ax, u, d, o):
    """A stack of blocks with one displaced, a centre-of-support line, and an
    optional corrective action arrow.

    The shape for stability, balance and support arguments — and for any task
    defined by an offset from an equilibrium.

    opts: blocks (count), offset, displaced ("top"|"side"), centre_label,
          offset_label, action_label, stable, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("blocks", 3))
    off = float(o.get("offset", 0.9))
    where = o.get("displaced", "top")
    bw, bh = 1.5, 0.72
    cx = x + w * 0.40
    base = y + 1.7

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    cols = ["#D96B4A", "#4A93D9", "#5CB87A", "#C9A227"]
    stack = n if where == "side" else n - 1
    for k in range(stack):
        ax.add_patch(Rectangle((cx - bw / 2, base + k * bh), bw, bh * 0.94,
                               fc=cols[k % 4], ec="none", alpha=min(a0, 1.0),
                               zorder=6))
    ax.plot([cx - bw * 1.6, cx + bw * 2.4], [base, base], lw=2.0,
            color=C["line"], alpha=min(a0, 1.0), zorder=4)

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    if where == "top":
        dx, dy = cx + off * bw, base + stack * bh
    else:
        dx, dy = cx + (1.6 + off) * bw, base
    ax.add_patch(Rectangle((dx - bw / 2, dy), bw, bh * 0.94, fc=C["accent"],
                           ec="none", alpha=min(a1, 1.0), zorder=7))
    R.txt(ax, dx, dy + bh + 0.30, o.get("block_label", "displaced block"), 15,
          C["accent"], a1, ha="center", weight="bold")

    a2 = _fade(u, R.rt(2, 2.4, 0.7), 0.6)
    ax.plot([cx, cx], [base - 0.35, base + (stack + 1.6) * bh], ls=(0, (4, 3)),
            lw=2.0, color=C["ink"], alpha=min(a2, 1.0) * 0.8, zorder=5)
    R.txt(ax, cx, base - 0.62, o.get("centre_label", "centre of the tower"),
          15, C["ink"], a2, ha="center")
    ax.plot([cx, dx], [dy - 0.34, dy - 0.34], lw=2.6, color=C["neg"],
            alpha=min(a2, 1.0), zorder=8)
    R.txt(ax, (cx + dx) / 2, dy - 0.74, o.get("offset_label", "offset"), 15,
          C["neg"], a2, ha="center", weight="bold")

    a3 = _fade(u, R.rt(3, 3.6, 0.7), 0.7)
    if o.get("action_label"):
        _arrow(ax, (dx, dy + bh * 1.9), (cx, dy + bh * 1.9), C["accent"], 2.8,
               a3, 16)
        R.txt(ax, (cx + dx) / 2, dy + bh * 1.9 + 0.36, o["action_label"], 16,
              C["accent"], a3, ha="center", weight="bold")
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.6, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


def reward_profile(R, ax, u, d, o):
    """Reward as a function of a continuous action, with the piecewise regimes
    shaded and labelled.

    The shape for shaped-reward designs, penalty structures, and any
    "the objective is discontinuous here" argument.

    opts: peak, regimes [{label, from, to, level, colour, gauss}],
          xlabel, ylabel, optimum_label, note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    px0, py0 = x + 1.8, y + 1.9
    pw, ph = w - 2.6, h - 3.2
    lo = float(o.get("ymin", -6.0))
    hi = float(o.get("ymax", 21.0))
    xs = np.linspace(-3, 3, 300)

    def PX(v):
        return px0 + (v + 3) / 6 * pw

    def PY(v):
        return py0 + (v - lo) / (hi - lo) * ph

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [PY(0), PY(0)], lw=1.4, color=C["line"],
            alpha=min(a0, 1.0))
    R.txt(ax, px0 + pw / 2, y + 0.95, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 1.05, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)
    R.txt(ax, px0 - 0.18, PY(0), "0", 14, C["muted"], a0, ha="right")

    for k, rg in enumerate(o.get("regimes", [])):
        a = _fade(u, R.rt(min(k + 1, 4), 0.9 + k * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        col = _col(R, rg.get("colour", "hot"))
        f, t = float(rg.get("from", -3)), float(rg.get("to", 3))
        if rg.get("gauss"):
            amp = float(rg.get("level", 20.0))
            base = float(rg.get("base", 0.0))
            m = (xs >= f) & (xs <= t)
            ax.plot(PX(xs[m]), PY(amp * np.exp(-xs[m] ** 2) + base), lw=3.0,
                    color=col, alpha=min(a, 1.0), zorder=7)
        else:
            lvl = float(rg.get("level", 0.0))
            ax.plot([PX(f), PX(t)], [PY(lvl), PY(lvl)], lw=3.0, color=col,
                    alpha=min(a, 1.0), zorder=7)
        if rg.get("label"):
            R.txt(ax, PX((f + t) / 2), PY(float(rg.get("label_y",
                  rg.get("level", 0)))) + 0.42, str(rg["label"]), 15, col, a,
                  ha="center", weight="bold")

    a2 = _fade(u, R.rt(4, 4.4, 0.8), 0.7)
    ax.plot([PX(0), PX(0)], [py0, py0 + ph], ls=(0, (4, 3)), lw=2.0,
            color=C["accent"], alpha=min(a2, 1.0), zorder=6)
    R.txt(ax, PX(0), py0 + ph + 0.10, o.get("optimum_label", ""), 16,
          C["accent"], a2, ha="center", weight="bold")
    R.txt(ax, x + 0.3, y + 0.42, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(5, 5.4, 0.8), 0.7))
    _tail(R, ax, o, u, 6, x, y, w)


def group_advantage(R, ax, u, d, o):
    """A group of sampled completions, their rewards, the group mean, and the
    resulting signed advantages.

    The shape for group-relative RL — GRPO, RLOO, GSPO — where the baseline
    is the group itself rather than a learned value function.

    opts: rewards [float], prompt_label, mean_label, adv_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    rw = [float(v) for v in o.get("rewards", [])]
    if not rw:                       # nothing to plot; degrade rather than crash
        return
    n = len(rw)
    bw = (w - 3.4) / n
    gx = x + 2.6
    mid = y + h * 0.58
    mu = float(np.mean(rw)) if rw else 0.0
    sd = float(np.std(rw)) or 1.0
    mx = max(abs(v - mu) for v in rw) or 1.0

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, x + 0.3, mid - 0.45, 2.0, 0.9, a=a0, fc=P["vis"], r=0.12, z=7)
    R.txt(ax, x + 1.3, mid, o.get("prompt_label", "prompt"), 15, "#FFFFFF", a0,
          ha="center", z=9)

    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    for k, v in enumerate(rw):
        g = _fade(u, R.rt(1, 1.3, 0.6) + k * 0.05, 0.35)
        cx = gx + (k + 0.5) * bw
        _arrow(ax, (x + 2.35, mid), (cx, mid + 1.15), C["muted"], 1.2,
               min(g, 1.0) * 0.5, 8)
        ax.add_patch(Rectangle((cx - bw * 0.30, mid + 1.25), bw * 0.60,
                               0.9 * (v - min(rw)) / (max(rw) - min(rw) + 1e-9)
                               + 0.06, fc=C["dim"], ec="none",
                               alpha=min(g, 1.0), zorder=6))
    R.txt(ax, gx, mid + 2.42, f"{n} completions, one reward each", 16,
          C["muted"], a1)

    a2 = _fade(u, R.rt(2, 2.5, 0.7), 0.6)
    ax.plot([gx, gx + n * bw], [mid, mid], ls=(0, (4, 3)), lw=2.2,
            color=C["accent"], alpha=min(a2, 1.0), zorder=7)
    R.txt(ax, gx + n * bw + 0.15, mid, o.get("mean_label", "group mean"), 15,
          C["accent"], a2)

    a3 = _fade(u, R.rt(3, 3.7, 0.7), 0.7)
    for k, v in enumerate(rw):
        adv = (v - mu) / sd
        cx = gx + (k + 0.5) * bw
        hh = 1.15 * (v - mu) / mx
        ax.add_patch(Rectangle((cx - bw * 0.30, mid if hh > 0 else mid + hh),
                               bw * 0.60, abs(hh),
                               fc=C["pos"] if hh > 0 else C["neg"], ec="none",
                               alpha=min(a3, 1.0), zorder=8))
    R.txt(ax, gx, mid - 1.62, o.get("adv_label", ""), 16, C["ink"], a3)
    _tail(R, ax, o, u, 4, x, y, w)


def value_matrix(R, ax, u, d, o):
    """A trained-on by evaluated-on matrix with the number printed in each
    cell and the colour scaled to it, so a strong diagonal and a collapsed
    off-diagonal are visible at once.

    opts: rows [str], cols [str], values [[float]], fmt, row_label, col_label,
          vmin, vmax, diagonal_note, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rows = o.get("rows", [])
    cols = o.get("cols", [])
    vals = o.get("values", [])
    nr, nc = max(len(rows), 1), max(len(cols), 1)
    gx = x + w * 0.30
    gy = y + h - 1.5
    cw = (w - (gx - x) - 0.5) / nc
    ch = (h - 2.7) / nr
    flat = [float(v) for r_ in vals for v in r_] or [0.0, 1.0]
    vmin = float(o.get("vmin", min(flat)))
    vmax = float(o.get("vmax", max(flat)))
    fmt = o.get("fmt", "{:.2f}")

    a0 = _fade(u, R.rt(0, 0.3, 0.5), 0.5)
    for c, lab in enumerate(cols):
        R.txt(ax, gx + (c + 0.5) * cw, gy + 0.26, str(lab), 13, C["muted"], a0,
              ha="center", rot=16)
    for r_, lab in enumerate(rows):
        R.txt(ax, gx - 0.22, gy - (r_ + 0.5) * ch, str(lab), 14, C["ink"], a0,
              ha="right")
    R.txt(ax, gx + nc * cw / 2, gy + 1.05, o.get("col_label", "trained on"),
          16, C["ink"], a0, ha="center", weight="bold")
    R.txt(ax, x + 0.45, gy - nr * ch / 2, o.get("row_label", "evaluated on"),
          16, C["ink"], a0, ha="center", va="center", rot=90)

    for r_ in range(nr):
        a = _fade(u, R.rt(min(r_, 3), 0.6 + r_ * 0.7, 0.5), 0.45)
        if a <= 0.01:
            continue
        for c in range(nc):
            v = float(vals[r_][c]) if r_ < len(vals) and c < len(vals[r_]) else 0.0
            t = (v - vmin) / max(vmax - vmin, 1e-9)
            diag = r_ == c
            ax.add_patch(Rectangle((gx + c * cw, gy - (r_ + 1) * ch),
                                   cw * 0.94, ch * 0.86, fc=C["accent"],
                                   ec="none",
                                   alpha=min(a, 1.0) * (0.12 + 0.82 * t),
                                   zorder=5))
            if diag:
                ax.add_patch(Rectangle((gx + c * cw, gy - (r_ + 1) * ch),
                                       cw * 0.94, ch * 0.86, fill=False,
                                       ec=C["ink"], lw=2.0, alpha=min(a, 1.0),
                                       zorder=8))
            R.txt(ax, gx + c * cw + cw * 0.47, gy - (r_ + 0.5) * ch,
                  fmt.format(v), 14,
                  C["ink"] if (t > 0.55 or diag) else C["muted"], a,
                  ha="center", weight="bold" if diag else "normal", z=9)
    R.txt(ax, x + 0.3, y + 0.72, o.get("diagonal_note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.2, 0.8), 0.7))
    _tail(R, ax, o, u, min(nr + 1, 5), x, y, w)


def probe_layers(R, ax, u, d, o):
    """Probe accuracy read off each layer, against the model's own accuracy on
    the same task — the gap between what is represented and what is used.

    The shape for competence-versus-performance and interpretability probing.

    opts: series [{name, values [float], colour}], behaviour, behaviour_label,
          chance, xlabel, ylabel, gap_label, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    px0, py0 = x + 1.8, y + 1.5
    pw, ph = w - 2.6, h - 2.8
    sers = o.get("series", [])
    n = max(len(sers[0]["values"]) if sers else 1, 1)

    def PX(i):
        return px0 + i / max(n - 1, 1) * pw

    def PY(v):
        return py0 + float(v) * ph

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    R.txt(ax, px0 + pw / 2, y + 0.62, o.get("xlabel", "layer"), 17, C["muted"],
          a0, ha="center")
    R.txt(ax, x + 1.05, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)
    if o.get("chance") is not None:
        ax.plot([px0, px0 + pw], [PY(o["chance"])] * 2, ls=(0, (3, 3)), lw=1.6,
                color=C["muted"], alpha=min(a0, 1.0))
        R.txt(ax, px0 + 0.15, PY(o["chance"]) - 0.30, "chance", 14, C["muted"],
              a0)

    for si, s in enumerate(sers):
        a = _fade(u, R.rt(min(si + 1, 3), 0.9 + si * 0.8, 0.6), 0.55)
        if a <= 0.01:
            continue
        col = _col(R, s.get("colour", ["hot", "in", "out"][si % 3]))
        vals = [float(v) for v in s.get("values", [])]
        g = _ease((u - R.rt(min(si + 1, 3), 0.9 + si * 0.8, 0.6)) / 0.9)
        m = max(2, int(len(vals) * g))
        ax.plot([PX(i) for i in range(m)], [PY(v) for v in vals[:m]], lw=3.0,
                color=col, alpha=min(a, 1.0), zorder=7)
        R.txt(ax, px0 + pw - 0.15, PY(vals[-1]) + 0.30 - si * 0.42,
              s.get("name", ""), 15, col, a, ha="right", weight="bold")

    b = o.get("behaviour")
    if b is not None:
        ab = _fade(u, R.rt(3, 3.6, 0.8), 0.7)
        ax.plot([px0, px0 + pw], [PY(b)] * 2, lw=3.0, color=C["neg"],
                alpha=min(ab, 1.0), zorder=8)
        R.txt(ax, px0 + pw, PY(b) - 0.36, o.get("behaviour_label", ""), 15,
              C["neg"], ab, ha="right", weight="bold")
        if sers:
            top = float(max(sers[0].get("values", [0])))
            ax.plot([px0 + pw * 0.55] * 2, [PY(b), PY(top)], lw=2.4,
                    color=C["ink"], alpha=min(ab, 1.0), zorder=9)
            R.txt(ax, px0 + pw * 0.55 - 0.20, (PY(b) + PY(top)) / 2,
                  o.get("gap_label", ""), 16, C["ink"], ab, ha="right",
                  weight="bold")
    _tail(R, ax, o, u, 4, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — schematic object scenes
# --------------------------------------------------------------------------- #

_STACK_COLS = ["#D96B4A", "#4A93D9", "#5CB87A", "#C9A227"]


def _mini_stack(ax, cx, base, s, blocks=3, offset=0.0, displaced="top",
                a=1.0, hot="#C9A227", ghost=False, centre=False, ink="#12161F"):
    """Draw a small block tower glyph. Returns the displaced block's centre.

    Shared by the schematic primitives so a tower looks the same everywhere.
    `offset` is in block widths; `displaced` is "top", "side" or None.
    """
    bw, bh = 0.62 * s, 0.34 * s
    if displaced == "side":
        cx -= 0.78 * bw          # keep the tower+block pair centred on cx
    stack = blocks if displaced == "side" else max(blocks - 1, 1)
    for k in range(stack):
        ax.add_patch(Rectangle((cx - bw / 2, base + k * bh), bw, bh * 0.9,
                               fc=_STACK_COLS[k % 4], ec="none",
                               alpha=min(a, 1.0) * (0.35 if ghost else 1.0),
                               zorder=6))
    if centre:
        ax.plot([cx, cx], [base - 0.10 * s, base + (stack + 1.4) * bh],
                ls=(0, (3, 3)), lw=1.2, color=ink, alpha=min(a, 1.0) * 0.6,
                zorder=7)
    if displaced is None:
        return None
    if displaced == "top":
        dx, dy = cx + offset * bw, base + stack * bh
    else:
        dx, dy = cx + (1.55 + offset) * bw, base
    ax.add_patch(Rectangle((dx - bw / 2, dy), bw, bh * 0.9, fc=hot, ec="none",
                           alpha=min(a, 1.0) * (0.35 if ghost else 1.0),
                           zorder=7))
    return (dx, dy + bh / 2)


def physics_principles(R, ax, u, d, o):
    """A row of panels, each a schematic of one core physical principle, with
    the expected and the violating outcome side by side.

    The shape for intuitive-physics batteries, violation-of-expectation
    designs, and capability taxonomies illustrated by example.

    opts: principles [{name, kind, note}], verdict
          kind: "support" | "solidity" | "permanence" | "continuity"
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    ps = o.get("principles", [])
    n = max(len(ps), 1)
    pw = (w - 0.8) / n
    s = min(pw * 0.42, (h - 2.6) * 0.30)
    base = y + h * 0.40

    for k, p in enumerate(ps):
        a = _fade(u, R.rt(min(k, 4), 0.5 + k * 0.85, 0.6), 0.55)
        if a <= 0.01:
            continue
        cx = x + 0.4 + pw * (k + 0.5)
        kind = p.get("kind", "support")
        if kind == "support":
            _mini_stack(ax, cx - pw * 0.21, base, s, 3, 0.15, "top", a,
                        hot=C["pos"], ink=C["ink"])
            _mini_stack(ax, cx + pw * 0.21, base, s, 3, 1.15, "top", a,
                        hot=C["neg"], ink=C["ink"])
            R.txt(ax, cx + pw * 0.21, base - 0.34, "falls", 13, C["neg"], a,
                  ha="center")
            R.txt(ax, cx - pw * 0.21, base - 0.34, "holds", 13, C["pos"], a,
                  ha="center")
        elif kind == "solidity":
            ax.add_patch(Rectangle((cx - 0.62 * s, base), 1.24 * s, 0.16 * s,
                                   fc=C["dim"], ec="none", alpha=min(a, 1.0),
                                   zorder=5))
            ax.plot([cx - 0.34 * s], [base + 0.55 * s], "o", ms=13,
                    color=C["pos"], alpha=min(a, 1.0), zorder=7)
            _arrow(ax, (cx - 0.34 * s, base + 0.48 * s),
                   (cx - 0.34 * s, base + 0.22 * s), C["pos"], 2.0, a, 11)
            ax.plot([cx + 0.34 * s], [base - 0.28 * s], "o", ms=13,
                    color=C["neg"], alpha=min(a, 1.0), zorder=7)
            _arrow(ax, (cx + 0.34 * s, base + 0.48 * s),
                   (cx + 0.34 * s, base - 0.18 * s), C["neg"], 2.0, a, 11)
            R.txt(ax, cx + 0.34 * s, base - 0.62 * s, "passes through", 13,
                  C["neg"], a, ha="center")
        elif kind == "permanence":
            ax.plot([cx - 0.55 * s], [base + 0.30 * s], "o", ms=13,
                    color=C["pos"], alpha=min(a, 1.0), zorder=7)
            ax.add_patch(Rectangle((cx - 0.16 * s, base), 0.34 * s, 0.72 * s,
                                   fc=C["dim"], ec="none",
                                   alpha=min(a, 1.0) * 0.8, zorder=8))
            ax.plot([cx + 0.55 * s], [base + 0.30 * s], "o", ms=13,
                    color=C["pos"], alpha=min(a, 1.0) * 0.9, zorder=7)
            _arrow(ax, (cx - 0.42 * s, base + 0.30 * s),
                   (cx + 0.42 * s, base + 0.30 * s), C["muted"], 1.8, a, 11)
            R.txt(ax, cx, base - 0.34, "should reappear", 13, C["muted"], a,
                  ha="center")
        else:                                   # continuity
            t = np.linspace(0, 1, 40)
            ax.plot(cx + (t - 0.5) * 1.2 * s,
                    base + 0.55 * s - 1.4 * (t - 0.5) ** 2 * s, lw=2.4,
                    color=C["pos"], alpha=min(a, 1.0), zorder=6)
            ax.plot([cx - 0.10 * s, cx + 0.10 * s],
                    [base + 0.52 * s, base + 0.10 * s], lw=2.4, ls=(0, (2, 2)),
                    color=C["neg"], alpha=min(a, 1.0), zorder=7)
            R.txt(ax, cx, base - 0.34, "no teleporting", 13, C["neg"], a,
                  ha="center")
        R.txt(ax, cx, base + s * 1.75, str(p.get("name", "")), 17, C["ink"],
              a, ha="center", weight="bold")
        if p.get("note"):
            R.txt(ax, cx, base - 0.74, p["note"], 13, C["muted"], a,
                  ha="center")
    _tail(R, ax, o, u, n, x, y, w)


def shortcut_failure(R, ax, u, d, o):
    """Two scenes that share a surface cue but differ in ground truth, so a
    model reading the cue gets one of them wrong.

    The shape for shortcut learning, spurious correlation and Clever-Hans
    arguments.

    opts: cue_label, left {label, offset, blocks, truth},
          right {label, offset, blocks, truth}, shortcut_says, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    s = min(w * 0.075, (h - 2.8) * 0.38)
    base = y + h * 0.42

    for i, key in enumerate(("left", "right")):
        spec = o.get(key, {})
        a = _fade(u, R.rt(i, 0.5 + i * 1.3, 0.6), 0.55)
        if a <= 0.01:
            continue
        cx = x + w * (0.28 + i * 0.44)
        _mini_stack(ax, cx, base, s, int(spec.get("blocks", 3)),
                    float(spec.get("offset", 0.4)), "top", a,
                    hot=C["accent"], centre=True, ink=C["ink"])
        R.txt(ax, cx, base + s * 1.80, str(spec.get("label", "")), 17, C["ink"],
              a, ha="center", weight="bold")
        truth = spec.get("truth", "")
        R.txt(ax, cx, base - 0.55, truth, 17,
              C["pos"] if "stable" in str(truth).lower()
              and "un" not in str(truth).lower() else C["neg"], a,
              ha="center", weight="bold")
        R.txt(ax, cx, base - 0.95, "ground truth", 13, C["muted"], a,
              ha="center")

    a2 = _fade(u, R.rt(2, 2.7, 0.8), 0.7)
    if a2 > 0.01:
        yy = base - 1.10
        ax.plot([x + w * 0.14, x + w * 0.86], [yy, yy], lw=2.0,
                color=C["line"], alpha=min(a2, 1.0))
        R.txt(ax, x + w * 0.10, yy, o.get("cue_label", ""), 15, C["muted"], a2,
              ha="right")
        for i in range(2):
            R.txt(ax, x + w * (0.28 + i * 0.44), yy - 0.42,
                  o.get("shortcut_says", ""), 16, C["accent"], a2,
                  ha="center", weight="bold")
    _tail(R, ax, o, u, 3, x, y, w)


def task_gallery(R, ax, u, d, o):
    """Cards for a set of task variants, each with a block schematic, a label
    and a score — so the reader sees what each task is, not just its name.

    opts: cards [{label, note, value, blocks, offset, displaced, action,
                  colour, best}], fmt, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    cards = o.get("cards", [])
    n = max(len(cards), 1)
    cw = (w - 0.8) / n
    s = min(cw * 0.40, (h - 3.2) * 0.40)
    base = y + h * 0.50
    fmt = o.get("fmt", "{:.2f}")

    for k, c in enumerate(cards):
        a = _fade(u, R.rt(min(k, 4), 0.5 + k * 0.7, 0.55), 0.5)
        if a <= 0.01:
            continue
        cx = x + 0.4 + cw * (k + 0.5)
        best = bool(c.get("best"))
        R.box(ax, cx - cw * 0.44, base - 1.45, cw * 0.88, s * 1.30 + 2.0,
              a=a * (0.5 if best else 0.28),
              fc=C["accent"] if best else C["panel"], r=0.14, z=3)
        tip = _mini_stack(ax, cx, base, s, int(c.get("blocks", 3)),
                          float(c.get("offset", 0.8)),
                          c.get("displaced", "top"), a, hot=C["accent"],
                          centre=True, ink=C["ink"])
        act = c.get("action")
        if act and tip:
            if act == "binary":
                R.txt(ax, cx, base + s * 0.86, "stable?", 15, C["neg"], a,
                      ha="center", weight="bold")
            elif act == "x":
                _arrow(ax, tip, (cx, tip[1]), C["accent"], 2.2, a, 12)
            elif act == "xy":
                _arrow(ax, tip, (cx, tip[1] + 0.30 * s), C["accent"], 2.2, a, 12)
        R.txt(ax, cx, base - 0.52, str(c.get("label", "")), 15, C["ink"], a,
              ha="center", weight="bold")
        if c.get("note"):
            R.txt(ax, cx, base - 0.88, c["note"], 13, C["muted"], a,
                  ha="center")
        if c.get("value") is not None:
            R.txt(ax, cx, base - 1.28, fmt.format(float(c["value"])), 18,
                  C["accent"] if best else C["ink"], a, ha="center",
                  weight="bold")
    _tail(R, ax, o, u, min(n, 5), x, y, w)


def frame_sequence(R, ax, u, d, o):
    """One or more rows of frames with actions between them, so a single-step
    setup and a multi-step one can be compared directly.

    opts: rows [{label, frames [{offset, displaced, blocks}], actions [str],
                 note}], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    rows = o.get("rows", [])
    nr = max(len(rows), 1)
    lane = (h - 1.8) / nr
    s = min(lane * 0.40, w * 0.036)

    mmax = max((len(r_.get("frames", [])) for r_ in rows), default=1) or 1
    for i, row in enumerate(rows):
        a = _fade(u, R.rt(min(i, 3), 0.5 + i * 1.4, 0.6), 0.55)
        if a <= 0.01:
            continue
        frames = row.get("frames", [])
        gx = x + 3.4
        fw = (w - 4.2) / mmax
        base = y + h - 0.9 - i * lane - lane * 0.55
        R.txt(ax, gx - 0.30, base + s * 0.4, str(row.get("label", "")), 17,
              C["ink"], a, ha="right", weight="bold")
        if row.get("note"):
            R.txt(ax, gx - 0.30, base - 0.05, row["note"], 14, C["muted"], a,
                  ha="right")
        for k, f in enumerate(frames):
            g = _fade(u, R.rt(min(i, 3), 0.5 + i * 1.4, 0.6) + k * 0.35, 0.45)
            cx = gx + fw * (k + 0.5)
            R.box(ax, cx - fw * 0.40, base - 0.30, fw * 0.80, s * 1.40 + 0.6,
                  a=g * 0.28, fc=C["panel"], r=0.10, z=3)
            _mini_stack(ax, cx, base, s, int(f.get("blocks", 3)),
                        float(f.get("offset", 0.8)),
                        f.get("displaced", "side"), g, hot=C["accent"],
                        ink=C["ink"])
            if k < len(row.get("actions", [])):
                _arrow(ax, (cx + fw * 0.42, base + s * 0.35),
                       (cx + fw * 0.60, base + s * 0.35), C["accent"], 2.2,
                       g, 13)
                R.txt(ax, cx + fw * 0.51, base + s * 0.35 + 0.30,
                      str(row["actions"][k]), 13, C["accent"], g, ha="center")
    _tail(R, ax, o, u, nr, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — comparison
# --------------------------------------------------------------------------- #

def compare_pipelines(R, ax, u, d, o):
    """Two pipelines stacked and aligned, with shared stages muted and the
    differing ones highlighted, so where two methods diverge is visible.

    opts: rows [{label, stages [[text, kind]]}], kinds:
          "same" | "diff" | "in" | "out", note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    rows = o.get("rows", [])
    nr = max(len(rows), 1)
    ncols = max((len(r_.get("stages", [])) for r_ in rows), default=1) or 1
    gx = x + 3.0
    sw = (w - 3.6) / ncols
    lane = (h - 2.0) / nr
    bh = min(1.20, lane * 0.52)

    for i, row in enumerate(rows):
        a0 = _fade(u, R.rt(min(i, 3), 0.5 + i * 1.5, 0.6), 0.55)
        cy = y + h - 1.0 - i * lane
        R.txt(ax, gx - 0.30, cy, str(row.get("label", "")), 17, C["ink"], a0,
              ha="right", weight="bold")
        if row.get("note"):
            R.txt(ax, gx - 0.30, cy - 0.46, row["note"], 14, C["muted"], a0,
                  ha="right")
        for k, st in enumerate(row.get("stages", [])):
            txt = st[0] if isinstance(st, (list, tuple)) else str(st)
            kind = st[1] if isinstance(st, (list, tuple)) and len(st) > 1 else "same"
            a = _fade(u, R.rt(min(i, 3), 0.5 + i * 1.5, 0.6) + k * 0.22, 0.45)
            if a <= 0.01:
                continue
            col = {"same": C["dim"], "diff": C["accent"], "in": P["vis"],
                   "out": P["txt"], "neg": C["neg"]}.get(kind, C["dim"])
            bx = gx + k * sw
            R.box(ax, bx, cy - bh / 2, sw * 0.86, bh, a=a, fc=col, r=0.11, z=7)
            import textwrap as _tw
            lines = _tw.wrap(str(txt), max(9, int(sw * 0.86 / 0.15)))[:3]
            for j, ln in enumerate(lines):
                R.txt(ax, bx + sw * 0.43,
                      cy + (len(lines) - 1) * 0.17 - j * 0.34, ln, 13,
                      "#FFFFFF", a, ha="center", z=9)
            if k:
                _arrow(ax, (bx - sw * 0.13, cy), (bx - 0.03, cy), C["muted"],
                       1.6, a)
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(3, 4.0, 0.8), 0.7))
    _tail(R, ax, o, u, nr + 1, x, y, w)


def venn2(R, ax, u, d, o):
    """Two overlapping sets: distinct items listed outside each circle, shared
    items inside the lens.

    opts: left {label, items}, right {label, items}, shared {label, items},
          verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cy = y + h * 0.58
    rad = min(w * 0.155, (h - 2.4) * 0.44)
    lx, rx = x + w / 2 - rad * 0.58, x + w / 2 + rad * 0.58

    for i, (cx, key, col) in enumerate(((lx, "left", P["vis"]),
                                        (rx, "right", P["txt"]))):
        a = _fade(u, R.rt(i, 0.5 + i * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        spec = o.get(key, {})
        ax.add_patch(Circle((cx, cy), rad, fc=col, ec=col, lw=2.0,
                            alpha=min(a, 1.0) * 0.15, zorder=4))
        side = -1 if i == 0 else 1
        R.txt(ax, cx + side * rad * 0.55, cy + rad + 0.40,
              str(spec.get("label", "")), 19, col, a, ha="center",
              weight="bold")
        items = list(spec.get("items", []))[:5]
        for j, it in enumerate(items):
            R.txt(ax, cx - side * -1 * (rad * 1.02) if False else
                  (cx + side * (rad * 1.02)),
                  cy + (len(items) - 1) * 0.24 - j * 0.48, str(it), 15,
                  C["ink"], a, ha="left" if i else "right")

    sh = o.get("shared", {})
    a2 = _fade(u, R.rt(2, 2.4, 0.8), 0.7)
    if a2 > 0.01 and sh:
        items = list(sh.get("items", []))[:4]
        R.txt(ax, x + w / 2, cy + rad * 0.62, str(sh.get("label", "")), 16,
              C["accent"], a2, ha="center", weight="bold")
        for j, it in enumerate(items):
            R.txt(ax, x + w / 2, cy + rad * 0.18 - j * 0.40, str(it), 13,
                  C["accent"], a2, ha="center")
    _tail(R, ax, o, u, 3, x, y, w)

def axis_positions(R, ax, u, d, o):
    """Items placed along one labelled axis, optionally as spans rather than
    points, so two approaches can be compared on a single dimension.

    opts: xlabel, lo_label, hi_label, items [{label, at, to, colour, note}],
          bands [{label, from, to, colour}], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    lx, rx = x + 1.4, x + w - 1.4
    ay = y + h * 0.40

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for bd in o.get("bands", []):
        ax.add_patch(Rectangle((lx + (rx - lx) * float(bd.get("from", 0)),
                                ay - 0.42),
                               (rx - lx) * (float(bd.get("to", 1))
                                            - float(bd.get("from", 0))), 0.84,
                               fc=_col(R, bd.get("colour", "dim")), ec="none",
                               alpha=min(a0, 1.0) * 0.16, zorder=3))
        R.txt(ax, lx + (rx - lx) * (float(bd.get("from", 0))
                                    + float(bd.get("to", 1))) / 2, ay - 0.72,
              str(bd.get("label", "")), 14, C["muted"], a0, ha="center")
    ax.plot([lx, rx], [ay, ay], lw=2.4, color=C["line"], alpha=min(a0, 1.0),
            zorder=4)
    R.txt(ax, (lx + rx) / 2, ay - 1.22, o.get("xlabel", ""), 17, C["ink"], a0,
          ha="center")
    R.txt(ax, lx, ay - 1.22, o.get("lo_label", ""), 14, C["muted"], a0,
          ha="center")
    R.txt(ax, rx, ay - 1.22, o.get("hi_label", ""), 14, C["muted"], a0,
          ha="center")

    for k, it in enumerate(o.get("items", [])):
        a = _fade(u, R.rt(min(k + 1, 4), 1.0 + k * 1.0, 0.6), 0.55)
        if a <= 0.01:
            continue
        col = _col(R, it.get("colour", "hot"))
        p0 = lx + (rx - lx) * float(it.get("at", 0.5))
        yy = ay + 0.95 + k * 1.10
        if it.get("to") is not None:
            p1 = lx + (rx - lx) * float(it["to"])
            ax.plot([p0, p1], [yy, yy], lw=8.0, color=col,
                    alpha=min(a, 1.0) * 0.65, solid_capstyle="round", zorder=7)
            for px in (p0, p1):
                ax.plot([px, px], [ay + 0.06, yy], ls=(0, (2, 3)), lw=1.4,
                        color=col, alpha=min(a, 1.0) * 0.6, zorder=6)
        else:
            ax.plot([p0], [yy], "o", ms=17, color=col, alpha=min(a, 1.0), zorder=8)
            ax.plot([p0, p0], [ay + 0.06, yy], ls=(0, (2, 3)), lw=1.4,
                    color=col, alpha=min(a, 1.0) * 0.6, zorder=6)
        R.txt(ax, (p0 + (p1 if it.get("to") is not None else p0)) / 2,
              yy + 0.34, str(it.get("label", "")), 17, col, a, ha="center",
              weight="bold")
        if it.get("note"):
            R.txt(ax, (p0 + (p1 if it.get("to") is not None else p0)) / 2,
                  yy - 0.42, it["note"], 14, C["muted"], a, ha="center")
    _tail(R, ax, o, u, 5, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — distributions over trajectories
# --------------------------------------------------------------------------- #

def mode_coverage(R, ax, u, d, o):
    """A multimodal target density with two sampling behaviours drawn on it:
    one collapsing onto a single mode, one covering all of them.

    The shape for mode collapse, diversity-seeking objectives, GFlowNets,
    posterior coverage.

    opts: modes [{at, height, label}], collapsed {label, at, n},
          covering {label, n}, xlabel, target_label, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    px0, py0 = x + 1.5, y + 1.9
    pw, ph = w - 2.3, h - 3.2
    modes = o.get("modes", [{"at": 0.2, "height": 0.8},
                            {"at": 0.5, "height": 1.0},
                            {"at": 0.82, "height": 0.6}])
    xs = np.linspace(0, 1, 300)
    dens = np.zeros_like(xs)
    for m in modes:
        dens += float(m.get("height", 1.0)) * np.exp(
            -((xs - float(m.get("at", 0.5))) ** 2) / (2 * 0.055 ** 2))
    dens /= dens.max() or 1.0

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"],
            alpha=min(a0, 1.0))
    ax.fill_between(px0 + xs * pw, py0, py0 + dens * ph, color=C["dim"],
                    alpha=min(a0, 1.0) * 0.30, zorder=4)
    ax.plot(px0 + xs * pw, py0 + dens * ph, lw=2.4, color=C["ink"],
            alpha=min(a0, 1.0), zorder=5)
    R.txt(ax, px0 + pw, py0 + ph + 0.10, o.get("target_label", ""), 16,
          C["ink"], a0, ha="right")
    R.txt(ax, px0 + pw / 2, y + 0.95, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    for m in modes:
        if m.get("label"):
            R.txt(ax, px0 + float(m["at"]) * pw, py0 - 0.34, str(m["label"]),
                  14, C["muted"], a0, ha="center")

    rng = np.random.default_rng(int(o.get("seed", 61)))
    coll = o.get("collapsed", {})
    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    if a1 > 0.01 and coll:
        at = float(coll.get("at", modes[len(modes) // 2].get("at", 0.5)))
        for k in range(int(coll.get("n", 14))):
            px = px0 + (at + rng.normal(0, 0.014)) * pw
            ax.plot([px], [py0 + ph * 0.30 + (k % 4) * 0.16], "o", ms=8,
                    color=C["neg"], alpha=min(a1, 1.0) * 0.85, zorder=7)
        R.txt(ax, px0 + at * pw, py0 + ph * 0.30 + 0.86,
              str(coll.get("label", "")), 16, C["neg"], a1, ha="center",
              weight="bold")

    cov = o.get("covering", {})
    a2 = _fade(u, R.rt(2, 2.7, 0.7), 0.7)
    if a2 > 0.01 and cov:
        n = int(cov.get("n", 26))
        picks = rng.choice(len(xs), size=n, p=dens / dens.sum())
        for k, idx in enumerate(picks):
            ax.plot([px0 + xs[idx] * pw], [py0 + ph * 0.72 + (k % 4) * 0.16],
                    "o", ms=8, color=C["accent"], alpha=min(a2, 1.0) * 0.9,
                    zorder=8)
        R.txt(ax, px0 + pw * 0.06, py0 + ph * 0.72 + 0.86,
              str(cov.get("label", "")), 16, C["accent"], a2, weight="bold")
    _tail(R, ax, o, u, 3, x, y, w)


def trajectory_flow(R, ax, u, d, o):
    """A chain of states with flow values and forward probabilities, and the
    balance condition drawn as a bracket between two states.

    The shape for GFlowNets, flow-matching-on-graphs, and detailed-balance
    style arguments.

    opts: states [str], flows [str], probs [str], balance [i, j],
          balance_label, terminal_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    st = o.get("states", [])
    n = max(len(st), 1)
    sw = (w - 2.0) / n
    gx = x + 1.0
    cy = y + h * 0.56

    for k, s in enumerate(st):
        a = _fade(u, R.rt(min(k, 3), 0.4 + k * 0.45, 0.5), 0.45)
        if a <= 0.01:
            continue
        cx = gx + (k + 0.5) * sw
        last = k == n - 1
        ax.add_patch(Circle((cx, cy), min(sw * 0.24, 0.46),
                            fc=C["accent"] if last else P["vis"], ec="none",
                            alpha=min(a, 1.0), zorder=7))
        R.txt(ax, cx, cy, str(s), 14, "#FFFFFF", a, ha="center", z=9)
        flows = o.get("flows", [])
        if k < len(flows):
            R.txt(ax, cx, cy + 0.72, str(flows[k]), 15, C["ink"], a,
                  ha="center")
        if k:
            _arrow(ax, (gx + (k - 0.5) * sw + 0.42, cy), (cx - 0.42, cy),
                   C["muted"], 2.0, a, 13)
            probs = o.get("probs", [])
            if k - 1 < len(probs):
                R.txt(ax, gx + k * sw, cy - 0.62, str(probs[k - 1]), 14,
                      P["txt"], a, ha="center")
    if o.get("terminal_label"):
        R.txt(ax, gx + (n - 0.5) * sw, cy - 0.98, o["terminal_label"], 14,
              C["accent"], _fade(u, R.rt(3, 1.8, 0.6), 0.6), ha="center")

    bal = o.get("balance")
    ab = _fade(u, R.rt(3, 3.0, 0.8), 0.7)
    if ab > 0.01 and bal:
        i, j = int(bal[0]), int(bal[1])
        x1 = gx + (i + 0.5) * sw
        x2 = gx + (j + 0.5) * sw
        yy = cy + 1.35
        ax.plot([x1, x1, x2, x2], [cy + 0.95, yy, yy, cy + 0.95], lw=2.2,
                color=C["accent"], alpha=min(ab, 1.0), zorder=7)
        R.txt(ax, (x1 + x2) / 2, yy + 0.30, o.get("balance_label", ""), 17,
              C["accent"], ab, ha="center", weight="bold")
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.2, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


def sparse_interpolation(R, ax, u, d, o):
    """A signal evaluated exactly at every k-th point and linearly interpolated
    between, so the saving and the approximation error are both visible.

    The shape for subsampled evaluation, checkpointing, coarse-to-fine
    estimation.

    opts: n, every, xlabel, ylabel, exact_label, interp_label, saving_label,
          verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("n", 33))
    lam = int(o.get("every", 8))
    px0, py0 = x + 1.7, y + 1.6
    pw, ph = w - 2.5, h - 2.9
    rng = np.random.default_rng(9)
    true = np.cumsum(rng.normal(0.9, 0.55, n))
    true = (true - true.min()) / (true.max() - true.min() + 1e-9)

    def PX(i):
        return px0 + i / max(n - 1, 1) * pw

    def PY(v):
        return py0 + float(v) * ph * 0.86

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    ax.plot([px0, px0], [py0, py0 + ph], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    ax.plot([px0, px0 + pw], [py0, py0], lw=1.6, color=C["line"], alpha=min(a0, 1.))
    R.txt(ax, px0 + pw / 2, y + 0.62, o.get("xlabel", ""), 17, C["muted"], a0,
          ha="center")
    R.txt(ax, x + 0.95, py0 + ph / 2, o.get("ylabel", ""), 17, C["muted"], a0,
          ha="center", va="center", rot=90)
    ax.plot([PX(i) for i in range(n)], [PY(v) for v in true], lw=1.8,
            ls=(0, (3, 3)), color=C["dim"], alpha=min(a0, 1.0), zorder=5)

    knots = list(range(0, n, lam))
    if knots[-1] != n - 1:
        knots.append(n - 1)
    a1 = _fade(u, R.rt(1, 1.3, 0.6), 0.6)
    for k, i in enumerate(knots):
        g = _fade(u, R.rt(1, 1.3, 0.6) + k * 0.12, 0.4)
        ax.plot([PX(i)], [PY(true[i])], "o", ms=15, color=C["accent"],
                alpha=min(g, 1.0), zorder=9)
        ax.plot([PX(i), PX(i)], [py0, PY(true[i])], ls=(0, (2, 3)), lw=1.2,
                color=C["accent"], alpha=min(g, 1.0) * 0.5, zorder=4)
    R.txt(ax, px0 + 0.15, py0 + ph, o.get("exact_label", ""), 16, C["accent"],
          a1, weight="bold")

    a2 = _fade(u, R.rt(2, 2.6, 0.7), 0.7)
    if a2 > 0.01:
        ax.plot([PX(i) for i in knots], [PY(true[i]) for i in knots], lw=3.0,
                color=P["vis"], alpha=min(a2, 1.0), zorder=8)
        R.txt(ax, px0 + 0.15, py0 + ph - 0.46, o.get("interp_label", ""), 16,
              P["vis"], a2, weight="bold")
    a3 = _fade(u, R.rt(3, 3.8, 0.8), 0.7)
    R.txt(ax, px0 + pw, py0 + ph, o.get("saving_label", ""), 16, C["ink"], a3,
          ha="right", weight="bold")
    _tail(R, ax, o, u, 4, x, y, w)


def candidate_ranking(R, ax, u, d, o):
    """One query fanning out to several candidates, each scored, with the
    winner selected — and no external judge in the picture.

    The shape for Best-of-N, self-consistency, reranking, marginal-likelihood
    selection.

    opts: query_label, candidates [{label, score, best}], score_label,
          fmt, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cands = o.get("candidates", [])
    n = max(len(cands), 1)
    qx = x + 1.4
    cy = y + h * 0.56
    gx = x + 3.6
    lane = min(1.05, (h - 2.2) / n)
    bw = w - (gx - x) - 3.0
    mx = max([abs(float(c.get("score", 0))) for c in cands] or [1.0]) or 1.0
    fmt = o.get("fmt", "{:.2f}")

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.box(ax, qx - 1.0, cy - 0.55, 2.0, 1.1, a=a0, fc=P["vis"], r=0.12, z=7)
    R.txt(ax, qx, cy, o.get("query_label", ""), 15, "#FFFFFF", a0,
          ha="center", z=9)

    for k, c in enumerate(cands):
        a = _fade(u, R.rt(min(k + 1, 4), 0.9 + k * 0.55, 0.5), 0.45)
        if a <= 0.01:
            continue
        yy = cy + (n - 1) / 2 * lane - k * lane
        best = bool(c.get("best"))
        _arrow(ax, (qx + 1.1, cy), (gx - 0.06, yy), C["muted"], 1.4,
               a * 0.55, 9)
        R.txt(ax, gx - 0.20, yy, str(c.get("label", "")), 15,
              C["ink"] if best else C["muted"], a, ha="right",
              weight="bold" if best else "normal")
        g = _ease((u - R.rt(min(k + 1, 4), 0.9 + k * 0.55, 0.5)) / 0.7)
        val = abs(float(c.get("score", 0))) / mx
        ax.add_patch(Rectangle((gx, yy - 0.18), bw * val * max(g, 0), 0.36,
                               fc=C["accent"] if best else C["dim"], ec="none",
                               alpha=min(a, 1.0), zorder=6))
        R.txt(ax, gx + bw * val * max(g, 0) + 0.20, yy,
              fmt.format(float(c.get("score", 0))), 15,
              C["ink"] if best else C["muted"], a * max(g, 0),
              weight="bold" if best else "normal")
        if best:
            ax.plot([x + w - 0.55], [yy], "*", ms=22, color=C["accent"],
                    alpha=min(a, 1.0), zorder=9)
    R.txt(ax, gx, cy + (n - 1) / 2 * lane + 0.62, o.get("score_label", ""), 16,
          C["muted"], _fade(u, R.rt(1, 0.9, 0.5), 0.5))
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.0, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


# --------------------------------------------------------------------------- #
# PRIMITIVES — diffusion and score teaching figures
# --------------------------------------------------------------------------- #

def diffusion_chain(R, ax, u, d, o):
    """A chain of states from data to noise, with the forward (corrupting)
    process arrowed one way and the reverse (generative) process the other.

    The shape for diffusion, Markov chains, and any pair of opposed processes
    over the same states.

    opts: n, forward_label, reverse_label, forward_eq, reverse_eq,
          data_label, noise_label, t_labels, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("n", 5))
    sw = (w - 1.6) / n
    s = min(sw * 0.46, (h - 3.2) * 0.5)
    gx = x + 0.8
    cy = y + h * 0.52
    rng = np.random.default_rng(int(o.get("seed", 5)))

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for k in range(n):
        cx = gx + (k + 0.5) * sw
        noise = k / max(n - 1, 1)
        R.box(ax, cx - s * 0.62, cy - s * 0.62, s * 1.24, s * 1.24,
              a=a0 * 0.30, fc=C["panel"], r=0.10, z=3)
        m = 34
        for _ in range(m):
            if rng.random() < 1 - noise:      # structured content
                px = cx + rng.normal(0, s * 0.17)
                py = cy + rng.normal(0, s * 0.17)
                col = P["vis"]
            else:                              # noise
                px = cx + rng.uniform(-s * 0.55, s * 0.55)
                py = cy + rng.uniform(-s * 0.55, s * 0.55)
                col = C["dim"]
            ax.plot([px], [py], "o", ms=4, color=col, alpha=min(a0, 1.0) * 0.8,
                    zorder=6)
        tl = o.get("t_labels", [])
        R.txt(ax, cx, cy - s * 0.90, tl[k] if k < len(tl) else f"t={k}", 14,
              C["muted"], a0, ha="center")

    R.txt(ax, gx + 0.5 * sw, cy + s * 1.55, o.get("data_label", ""), 16,
          P["vis"], a0, ha="center", weight="bold")
    R.txt(ax, gx + (n - 0.5) * sw, cy + s * 1.55, o.get("noise_label", ""), 16,
          C["muted"], a0, ha="center", weight="bold")

    a1 = _fade(u, R.rt(1, 1.4, 0.7), 0.6)
    for k in range(n - 1):
        g = _fade(u, R.rt(1, 1.4, 0.7) + k * 0.10, 0.4)
        x0 = gx + (k + 0.5) * sw + s * 0.70
        x1 = gx + (k + 1.5) * sw - s * 0.70
        _arrow(ax, (x0, cy + s * 0.30), (x1, cy + s * 0.30), C["neg"], 2.2,
               min(g, 1.0), 13)
    R.txt(ax, x + w / 2, cy + s * 1.10, o.get("forward_label", ""), 16,
          C["neg"], a1, ha="center", weight="bold")
    R.txt(ax, x + w / 2, y + h - 0.25, o.get("forward_eq", ""), 18, C["neg"],
          a1, ha="center")

    a2 = _fade(u, R.rt(2, 2.8, 0.7), 0.7)
    for k in range(n - 1, 0, -1):
        g = _fade(u, R.rt(2, 2.8, 0.7) + (n - 1 - k) * 0.10, 0.4)
        x0 = gx + (k + 0.5) * sw - s * 0.70
        x1 = gx + (k - 0.5) * sw + s * 0.70
        _arrow(ax, (x0, cy - s * 0.30), (x1, cy - s * 0.30), C["accent"], 2.2,
               min(g, 1.0), 13)
    R.txt(ax, x + w / 2, cy - s * 1.32, o.get("reverse_label", ""), 16,
          C["accent"], a2, ha="center", weight="bold")
    R.txt(ax, x + w / 2, y + 0.86, o.get("reverse_eq", ""), 18, C["accent"],
          a2, ha="center")
    _tail(R, ax, o, u, 3, x, y, w)


def score_field(R, ax, u, d, o):
    """A 2-D density with the score field drawn as arrows pointing uphill, and
    optionally a Langevin trajectory walking up it.

    The shape for score matching, energy landscapes, gradient flows, and the
    regions where an estimated gradient is unreliable.

    opts: modes [{x, y, w}], walk (bool), walk_steps, bad_region [x, y, r],
          bad_label, field_label, walk_label, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    cx, cy = x + w / 2, y + h * 0.54
    ext = min(w * 0.22, (h - 2.4) * 0.46)
    modes = o.get("modes", [{"x": -0.5, "y": 0.3, "w": 0.34},
                            {"x": 0.55, "y": -0.25, "w": 0.30}])

    def logp_grad(px, py):
        gx_ = gy_ = 0.0
        tot = 1e-9
        for m in modes:
            dx, dy = px - m["x"], py - m["y"]
            s2 = m.get("w", 0.3) ** 2
            wgt = np.exp(-(dx * dx + dy * dy) / (2 * s2))
            tot += wgt
            gx_ += wgt * (-dx / s2)
            gy_ += wgt * (-dy / s2)
        return gx_ / tot, gy_ / tot

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for m in modes:
        for rr, al in ((0.6, 0.20), (1.1, 0.13), (1.7, 0.08)):
            ax.add_patch(Circle((cx + m["x"] * ext, cy + m["y"] * ext),
                                m.get("w", 0.3) * rr * ext * 2.0, fc=P["vis"],
                                ec="none", alpha=min(a0, 1.0) * al, zorder=3))

    a1 = _fade(u, R.rt(1, 1.3, 0.7), 0.6)
    if a1 > 0.01:
        for gx_i in np.linspace(-1.0, 1.0, 9):
            for gy_i in np.linspace(-0.75, 0.75, 7):
                vx, vy = logp_grad(gx_i, gy_i)
                nrm = np.hypot(vx, vy) + 1e-9
                sc = min(nrm, 3.0) / 3.0
                if sc < 0.02:
                    continue
                ux, uy = vx / nrm * 0.20 * ext, vy / nrm * 0.20 * ext
                _arrow(ax, (cx + gx_i * ext, cy + gy_i * ext),
                       (cx + gx_i * ext + ux, cy + gy_i * ext + uy),
                       C["accent"], 1.4, min(a1, 1.0) * (0.30 + 0.6 * sc), 8)
        R.txt(ax, x + 0.3, y + h - 0.25, o.get("field_label", ""), 17,
              C["accent"], a1)

    bad = o.get("bad_region")
    ab = _fade(u, R.rt(2, 2.6, 0.7), 0.6)
    if bad and ab > 0.01:
        ax.add_patch(Circle((cx + bad[0] * ext, cy + bad[1] * ext),
                            bad[2] * ext, fill=False, ec=C["neg"], lw=2.4,
                            ls=(0, (4, 3)), alpha=min(ab, 1.0), zorder=8))
        R.txt(ax, cx + bad[0] * ext, cy + (bad[1] - bad[2]) * ext - 0.34,
              o.get("bad_label", ""), 15, C["neg"], ab, ha="center",
              weight="bold")

    if o.get("walk"):
        t3 = R.rt(3, 3.4, 0.8)
        prog = _ease(min(1.0, max(0.0, (u - t3) / max(d - t3 - 0.9, 1.2))))
        rng = np.random.default_rng(int(o.get("seed", 13)))
        steps = int(o.get("walk_steps", 40))
        px, py = float(o.get("start_x", -1.0)), float(o.get("start_y", -0.65))
        pts = [(px, py)]
        for _ in range(steps):
            vx, vy = logp_grad(px, py)
            px += 0.045 * vx + rng.normal(0, 0.055)
            py += 0.045 * vy + rng.normal(0, 0.055)
            pts.append((px, py))
        m = max(2, int(len(pts) * prog))
        ax.plot([cx + p[0] * ext for p in pts[:m]],
                [cy + p[1] * ext for p in pts[:m]], lw=2.2, color=P["txt"],
                alpha=min(_fade(u, t3, 0.5), 1.0), zorder=9)
        ax.plot([cx + pts[m - 1][0] * ext], [cy + pts[m - 1][1] * ext], "o",
                ms=14, color=P["txt"], alpha=min(_fade(u, t3, 0.5), 1.0),
                zorder=10)
        R.txt(ax, x + 0.3, y + h - 0.72, o.get("walk_label", ""), 17, P["txt"],
              _fade(u, t3, 0.6))
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(4, 4.6, 0.8), 0.7))
    _tail(R, ax, o, u, 5, x, y, w)


def unet_shape(R, ax, u, d, o):
    """The U-shaped encoder-decoder with skip connections between matching
    resolutions.

    opts: depth, in_label, out_label, bottleneck_label, skip_label, note,
          verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    D = int(o.get("depth", 4))
    cx = x + w / 2
    top = y + h - 1.2
    step_y = (h - 2.8) / max(D, 1)
    bw0 = min(w * 0.13, 2.0)

    lefts, rights = [], []
    for k in range(D):
        a = _fade(u, R.rt(min(k, 3), 0.4 + k * 0.45, 0.5), 0.45)
        yy = top - k * step_y
        bw = bw0 * (1 - 0.16 * k)
        bh = min(0.62, step_y * 0.62)
        lx = cx - w * 0.17 - (D - k) * 0.22
        rx = cx + w * 0.17 + (D - k) * 0.22
        for side, px in (("l", lx), ("r", rx)):
            R.box(ax, px - bw / 2, yy - bh / 2, bw, bh, a=a,
                  fc=P["vis"] if side == "l" else _col(R, "out"), r=0.09, z=7)
        lefts.append((lx, yy, bw))
        rights.append((rx, yy, bw))
        if k:
            _arrow(ax, (lefts[k - 1][0], lefts[k - 1][1] - bh * 0.75),
                   (lx, yy + bh * 0.62), C["muted"], 1.6, a, 10)
            _arrow(ax, (rx, yy + bh * 0.62),
                   (rights[k - 1][0], rights[k - 1][1] - bh * 0.75),
                   C["muted"], 1.6, a, 10)

    ab = _fade(u, R.rt(3, 2.4, 0.6), 0.6)
    by = top - (D - 1) * step_y - step_y * 0.85
    R.box(ax, cx - bw0 * 0.45, by - 0.30, bw0 * 0.90, 0.60, a=ab,
          fc=C["accent"], r=0.09, z=7)
    R.txt(ax, cx, by, o.get("bottleneck_label", ""), 13, "#FFFFFF", ab,
          ha="center", z=9)
    _arrow(ax, (lefts[-1][0], lefts[-1][1] - 0.36), (cx - bw0 * 0.5, by),
           C["muted"], 1.6, ab, 10)
    _arrow(ax, (cx + bw0 * 0.5, by), (rights[-1][0], rights[-1][1] - 0.36),
           C["muted"], 1.6, ab, 10)

    a4 = _fade(u, R.rt(4, 3.4, 0.8), 0.7)
    for k in range(D):
        lx, yy, bw = lefts[k]
        rx = rights[k][0]
        ax.plot([lx + bw / 2 + 0.06, rx - bw / 2 - 0.06], [yy, yy],
                ls=(0, (3, 3)), lw=1.8, color=C["accent"],
                alpha=min(a4, 1.0) * 0.85, zorder=6)
    R.txt(ax, cx, top + 0.42, o.get("skip_label", ""), 16, C["accent"], a4,
          ha="center", weight="bold")
    R.txt(ax, lefts[0][0], top + 0.42, o.get("in_label", ""), 15, P["vis"],
          _fade(u, R.rt(0, 0.4, 0.5), 0.5), ha="center")
    R.txt(ax, rights[0][0], top + 0.42, o.get("out_label", ""), 15,
          _col(R, "out"), _fade(u, R.rt(0, 0.4, 0.5), 0.5), ha="center")
    R.txt(ax, x + 0.3, y + 0.72, o.get("note", ""), 16, C["muted"],
          _fade(u, R.rt(5, 4.4, 0.8), 0.7))
    _tail(R, ax, o, u, 6, x, y, w)


def noise_scales(R, ax, u, d, o):
    """The same density blurred by increasing amounts of noise, so the
    sharp-but-hard to sharp-but-easy trade-off is visible.

    opts: sigmas [float], labels [str], verdicts [str], xlabel, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    sig = [float(v) for v in o.get("sigmas", [0.03, 0.10, 0.30])]
    n = max(len(sig), 1)
    pw = (w - 0.8) / n
    labs = o.get("labels", [])
    vers = o.get("verdicts", [])
    xs = np.linspace(0, 1, 300)
    base = [(0.20, 0.9), (0.48, 1.0), (0.78, 0.7)]

    for k, s_ in enumerate(sig):
        a = _fade(u, R.rt(min(k, 3), 0.5 + k * 0.9, 0.6), 0.55)
        if a <= 0.01:
            continue
        bx = x + 0.4 + pw * k
        by = y + 1.7
        ph = h - 3.2
        dens = np.zeros_like(xs)
        for mu, hgt in base:
            wid = np.sqrt(0.022 ** 2 + s_ ** 2)
            dens += hgt * np.exp(-((xs - mu) ** 2) / (2 * wid ** 2))
        dens /= dens.max() or 1.0
        col = _col(R, "hot" if k == n // 2 else "dim")
        ax.fill_between(bx + 0.25 + xs * (pw - 0.7), by, by + dens * ph,
                        color=col, alpha=min(a, 1.0) * 0.35, zorder=5)
        ax.plot(bx + 0.25 + xs * (pw - 0.7), by + dens * ph, lw=2.6, color=col,
                alpha=min(a, 1.0), zorder=6)
        ax.plot([bx + 0.25, bx + pw - 0.45], [by, by], lw=1.4, color=C["line"],
                alpha=min(a, 1.0))
        R.txt(ax, bx + pw / 2 - 0.1, by + ph + 0.40,
              labs[k] if k < len(labs) else f"sigma = {s_:g}", 17, C["ink"], a,
              ha="center", weight="bold")
        if k < len(vers):
            R.txt(ax, bx + pw / 2 - 0.1, by - 0.42, vers[k], 15,
                  C["accent"] if k == n // 2 else C["neg"], a, ha="center")
    R.txt(ax, x + w / 2, y + 0.80, o.get("xlabel", ""), 16, C["muted"],
          _fade(u, R.rt(3, 3.4, 0.8), 0.7), ha="center")
    _tail(R, ax, o, u, 4, x, y, w)


# --------------------------------------------------------------------------- #
# PRESETS — the maintained dictionary
#
# Each entry is (primitive, opts). Grouped by paper so you can find and edit
# them; the namespace is flat, so keep names specific. Everything here is data:
# adding a paper normally means adding a block below and nothing else.
# --------------------------------------------------------------------------- #

PRESETS = {

    # === Attention Is All You Need (Vaswani et al., 2017) ==================
    "encoder_stack": ("residual_stack", {
        "title": "Encoder layer", "repeat_label": r"$\times 6$",
        "sublayers": [{"label": "Multi-Head Self-Attention", "colour": "hot"},
                      {"label": "Feed-Forward  512 -> 2048 -> 512", "colour": "in"}],
        "input_label": "input embedding + positional encoding",
        "output_label": "to the next layer",
        "norm_label": "LayerNorm",
        "verdict": r"every sublayer is $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$"}),
    "decoder_stack": ("residual_stack", {
        "title": "Decoder layer", "repeat_label": r"$\times 6$",
        "sublayers": [{"label": "Masked Multi-Head Self-Attention", "colour": "neg"},
                      {"label": "Encoder-Decoder Attention", "colour": "hot",
                       "from_memory": True},
                      {"label": "Position-wise Feed-Forward", "colour": "in"}],
        "memory_label": "encoder output  (keys and values)",
        "input_label": "output embedding, shifted right",
        "output_label": "to the next layer",
        "norm_label": "LayerNorm",
        "verdict": "the middle sublayer is the only place the target reads the source"}),
    "scaled_dot_product": ("matmul_chain", {
        "items": [{"label": "Q", "shape": [7, 4], "dims": "n x d_k", "colour": "in"},
                  {"label": r"$K^{\top}$", "shape": [4, 7], "dims": "d_k x n",
                   "colour": "in"},
                  {"label": "scores", "shape": [7, 7], "dims": "n x n",
                   "colour": "hot", "note": r"scale by $1/\sqrt{d_k}$, mask, softmax"},
                  {"label": "V", "shape": [7, 4], "dims": "n x d_v",
                   "colour": "out"},
                  {"label": "output", "shape": [7, 4], "dims": "n x d_v",
                   "colour": "in"}],
        "ops": [r"$\times$", r"$\rightarrow$", r"$\times$", "="],
        "note": "the n x n score matrix is where the quadratic cost lives",
        "verdict": "two matrix multiplies with a softmax between them"}),
    "multi_head": ("matmul_chain", {
        "items": [{"label": "input", "shape": [6, 8], "dims": "n x 512",
                   "colour": "in"},
                  {"label": "8 heads", "shape": [6, 1], "dims": "n x 64 each",
                   "colour": "hot", "note": "one attention per head"},
                  {"label": "concat", "shape": [6, 8], "dims": "n x 512",
                   "colour": "out"},
                  {"label": r"$W^O$", "shape": [8, 8], "dims": "512 x 512",
                   "colour": "mid"},
                  {"label": "output", "shape": [6, 8], "dims": "n x 512",
                   "colour": "in"}],
        "ops": ["split", "concat", r"$\times$", "="],
        "note": "512 split into 8 x 64, so h heads cost what one full-width head costs",
        "verdict": "the diversity is free; only the projections are learned"}),
    "causal_mask": ("matrix_grid", {
        "rows": 8, "cols": 8, "row_label": "query position",
        "col_label": "key position", "cmap_colour": "hot",
        "values": [[1 if j <= i else 0.06 for j in range(8)] for i in range(8)],
        "verdict": "position i may only attend to positions up to i"}),

    # === BERT (Devlin et al., 2019) =======================================
    "mlm_mask": ("sequence_tokens", {
        "tokens": ["the", "cat", "[MASK]", "on", "the", "[MASK]", "."],
        "note": "15% of tokens are replaced and then predicted",
        "verdict": "the objective forces bidirectional context"}),
    "pretrain_finetune": ("timeline", {
        "events": [{"label": "pre-train", "note": "unlabelled corpus"},
                   {"label": "fine-tune", "note": "one task, few epochs"},
                   {"label": "deploy", "note": "task-specific head"}],
        "axis_label": "one model, two phases",
        "verdict": "the same weights serve every downstream task"}),

    # === ResNet (He et al., 2016) =========================================
    "residual_block": ("stacked_blocks", {
        "title": "Residual block", "repeat_label": r"$\times N$", "depth": 2,
        "sublayers": [{"label": r"conv $3\times3$ + BN + ReLU", "colour": "in"},
                      {"label": r"conv $3\times3$ + BN", "colour": "in"}],
        "input_label": r"$x$",
        "verdict": r"the block learns $F(x)$, the output is $F(x)+x$"}),
    "depth_degradation": ("curve_family", {
        "series": [{"name": "20", "colour": "dim",
                    "points": [[0, 9.0], [1, 6.2], [2, 4.8], [3, 4.1], [4, 3.8], [5, 3.6]]},
                   {"name": "56 plain", "colour": "neg",
                    "points": [[0, 9.5], [1, 7.4], [2, 6.5], [3, 6.0], [4, 5.7], [5, 5.6]]},
                   {"name": "56 res", "colour": "hot",
                    "points": [[0, 8.8], [1, 5.4], [2, 3.9], [3, 3.1], [4, 2.7], [5, 2.5]]}],
        "xlabel": "training iterations", "ylabel": "training error",
        "hi_label": "worse", "lo_label": "better", "logx": False,
        "verdict": "deeper plain networks got worse, not just overfit"}),

    # === GANs (Goodfellow et al., 2014) ===================================
    "adversarial_loop": ("arrow_cycle", {
        "stages": [["sample noise", "in"], ["generate", "hot"],
                   ["discriminate", "neg"], ["update both", "out"]],
        "centre_label": "minimax",
        "verdict": "each network's loss is the other's gain"}),
    "gan_densities": ("density_pair", {
        "a_label": "real data", "b_label": "generated",
        "from_gap": 0.40, "to_gap": 0.04,
        "verdict": "training pulls the two distributions together"}),

    # === VAE (Kingma & Welling, 2014) =====================================
    "vae_flow": ("pipeline", {
        "stages": [["input\nx", "in"], ["encoder", "mid"], [r"$\mu,\sigma$", "hot"],
                   ["sample\nz", "out"], ["decoder", "mid"], ["reconstruct", "in"]],
        "verdict": "the reparameterisation trick makes sampling differentiable",
        "note": r"$z=\mu+\sigma\odot\epsilon$"}),
    "elbo_terms": ("annotated_equation", {
        "latex": r"$\log p(x)\geq \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x)\,\|\,p(z))$",
        "terms": [{"text": "reconstruct", "at": 0.44, "note": "explain the data", "colour": "in"},
                  {"text": "regularise", "at": 0.82, "note": "stay near the prior", "colour": "neg"}],
        "verdict": "one term pulls each way"}),

    # === Batch Normalisation (Ioffe & Szegedy, 2015) ======================
    "internal_shift": ("density_pair", {
        "a_label": "layer input, early", "b_label": "layer input, later",
        "from_gap": 0.05, "to_gap": 0.38,
        "verdict": "the distribution each layer sees keeps moving"}),

    # === Dropout (Srivastava et al., 2014) ================================
    "dropout_mask": ("matrix_grid", {
        "rows": 8, "cols": 8, "row_label": "units", "col_label": "batch",
        "cmap_colour": "in",
        "verdict": "a different sub-network on every forward pass"}),

    # === CLIP (Radford et al., 2021) ======================================
    "contrastive_matrix": ("matrix_grid", {
        "rows": 8, "cols": 8, "row_label": "image", "col_label": "text",
        "cmap_colour": "in",
        "values": [[1.0 if i == j else 0.14 for j in range(8)] for i in range(8)],
        "highlight": {"block": [0, 0, 7, 7]},
        "verdict": "pull the diagonal together, push everything else apart"}),

    # === Vision Transformer (Dosovitskiy et al., 2021) ====================
    "patchify": ("grid_collapse", {
        "n": 12, "s": 4, "grid_label": "image pixels",
        "region_label": r"$16\times16$ patches",
        "out_label": "tokens", "note": "each patch becomes one token",
        "verdict": "an image is a sequence"}),

    # === Chinchilla (Hoffmann et al., 2022) ===============================
    "compute_frontier": ("curve_family", {
        "series": [{"name": "1B", "points": [[1e19, 3.1], [1e20, 2.85], [1e21, 2.78]]},
                   {"name": "10B", "points": [[1e20, 2.72], [1e21, 2.50], [1e22, 2.44]]},
                   {"name": "70B", "points": [[1e21, 2.46], [1e22, 2.24], [1e23, 2.18]]}],
        "xlabel": "training FLOPs", "ylabel": "loss",
        "hi_label": "worse", "lo_label": "better",
        "frontier": "min", "frontier_label": "compute optimal",
        "verdict": "scale parameters and data together"}),
    "param_data_ladder": ("scale_ladder", {
        "rungs": [{"label": "1B", "size": 0.4}, {"label": "10B", "size": 1.0},
                  {"label": "70B", "size": 2.0},
                  {"label": "280B", "size": 3.0, "note": "over-parameterised"}],
        "verdict": "bigger is not automatically better per FLOP"}),

    # === LoRA (Hu et al., 2022) ===========================================
    "lora_branch": ("flow_split", {
        "source": "x", "sink": "h",
        "branches": [{"label": "frozen W", "weight": 0.9, "chosen": True},
                     {"label": "BA (rank r)", "weight": 0.3, "chosen": True}],
        "verdict": "train a low-rank update, keep the base weights fixed"}),

    # === Mixture of Experts (Shazeer et al., 2017) ========================
    "moe_router": ("flow_split", {
        "source": "token", "sink": "output",
        "branches": [{"label": "expert 1", "weight": 0.15},
                     {"label": "expert 2", "weight": 0.9, "chosen": True},
                     {"label": "expert 3", "weight": 0.2},
                     {"label": "expert 4", "weight": 0.1}],
        "verdict": "capacity grows, compute per token does not"}),

    # === DDPM (Ho et al., 2020) ===========================================
    "diffusion_pair": ("before_after", {
        "left": {"label": "noise", "n": 16, "colour": "dim"},
        "right": {"label": "sample", "n": 16, "colour": "in"},
        "arrow_label": "T denoising steps",
        "verdict": "learn to reverse a fixed corruption process"}),
    "noise_schedule": ("growth_curve", {
        "kind": "linear", "xlabel": "timestep t", "ylabel": "noise level",
        "curve_label": r"$\beta_t$", "reference": False,
        "verdict": "the forward process is fixed, not learned"}),

    # === InstructGPT / RLHF (Ouyang et al., 2022) =========================
    "rlhf_stages": ("timeline", {
        "events": [{"label": "SFT", "note": "demonstrations"},
                   {"label": "reward model", "note": "ranked comparisons"},
                   {"label": "PPO", "note": "optimise against the RM"}],
        "axis_label": "three stages",
        "verdict": "the reward model is a learned stand-in for human preference"}),

    # === Chain-of-Thought (Wei et al., 2022) ==============================
    "cot_tokens": ("sequence_tokens", {
        "tokens": ["Q", "step 1", "step 2", "step 3", "therefore", "A"],
        "note": "the model writes its working before the answer",
        "verdict": "test-time compute the model chooses to spend"}),

    # === Knowledge Distillation (Hinton et al., 2015) =====================
    "distillation": ("flow_split", {
        "source": "input", "sink": "student loss",
        "branches": [{"label": "teacher (frozen)", "weight": 0.8, "chosen": True},
                     {"label": "student", "weight": 0.8, "chosen": True}],
        "verdict": "soft targets carry more signal than hard labels"}),
    "soft_vs_hard": ("distribution_pair", {
        "top_title": "hard label", "bottom_title": "teacher's soft distribution"}),

    # === SimCLR (Chen et al., 2020) =======================================
    "two_views": ("before_after", {
        "left": {"label": "view 1", "n": 12, "colour": "in"},
        "right": {"label": "view 2", "n": 12, "colour": "out"},
        "arrow_label": "same image, two augmentations",
        "verdict": "agreement between views is the whole signal"}),

    # === Speculative decoding (Leviathan et al., 2023) ====================
    "draft_verify": ("pipeline", {
        "stages": [["prompt", "in"], ["draft model\nk tokens", "out"],
                   ["target model\nverify in parallel", "hot"],
                   ["accept prefix", "in"], ["resample", "neg"]],
        "verdict": "one expensive forward pass validates several cheap ones"}),

    # === FlashAttention (Dao et al., 2022) ================================
    "attention_tiling": ("matrix_grid", {
        "rows": 12, "cols": 12, "row_label": "queries", "col_label": "keys",
        "cmap_colour": "in", "highlight": {"block": [4, 4, 7, 7]},
        "verdict": "never materialise the full matrix; work tile by tile"}),

    # === SynOOD (Li et al., ICCV 2025) ====================================
    "boundary_problem": ("boundary_band", {
        "a_label": "InD images", "b_label": "easy OOD",
        "band_label": "the InD/OOD boundary",
        "synth_label": "synthesised near-boundary samples",
        "n_a": 34, "n_b": 30, "n_synth": 14,
        "verdict": "hard OOD samples live here, and CLIP misreads them"}),
    "clip_misalignment": ("dual_space_align", {
        "top_label": "text feature space", "bottom_label": "image feature space",
        "links": [[0.10, 0.16, 0.10], [0.30, 0.62, 0.30], [0.50, 0.20, 0.50],
                  [0.70, 0.88, 0.70], [0.90, 0.44, 0.90]],
        "before_note": "images are packed densely, labels are not — so hard "
                       "OOD samples snap to InD labels",
        "after_note": "after fine-tuning, each boundary sample reaches its "
                      "own negative label",
        "verdict": "the fix is alignment, not a better score function"}),
    "synood_generation": ("feedback_loop", {
        "stages": [["InD image", "in"], ["MLLM\ncontext labels", "out"],
                   ["in-painting\ndiffusion", "hot"], ["synthetic\nimage", "in"],
                   ["OOD score\n(Energy)", "neg"]],
        "feedback_from": 4, "feedback_to": 2,
        "feedback_label": r"$\nabla_\epsilon\mathcal{L}^{O}$  updates the noise",
        "iterations_label": "3 iterations",
        "verdict": "the OOD score is the loss, and the noise is what it trains"}),
    "energy_loss": ("annotated_equation", {
        "latex": r"$\mathcal{L}^{O}=m_{\mathrm{out}}-\tau\log\sum_{i=1}^{C}"
                 r"e^{g_i(x^{\mathrm{syn}})/\tau}$",
        "terms": [{"text": "OOD threshold", "at": 0.30, "note": "how far to push",
                   "colour": "out"},
                  {"text": "energy of the sample", "at": 0.74,
                   "note": "from a ResNet50", "colour": "neg"}],
        "verdict": "descend this and the image drifts across the boundary"}),
    "synood_training": ("module_states", {
        "modules": [{"label": "CLIP image\nencoder F", "state": "frozen"},
                    {"label": "projection δ", "state": "train",
                     "note": r"CLIP loss $\mathcal{L}^{P}$"},
                    {"label": "CLIP text\nencoder H", "state": "frozen"},
                    {"label": "negative label\nfeatures", "state": "train",
                     "note": r"$\approx$ 5.5k of 11k"}],
        "verdict": "under 1% extra parameters, under 2 ms per image"}),
    "synthetic_examples": ("pair_strip", {
        "columns": ["hourglass", "lifeboat", "tench", "drake", "harmonica",
                    "cheeseburger", "lipstick"],
        "row_a": "InD", "row_b": "synthesised OOD",
        "colour_a": "in", "colour_b": "hot",
        "verdict": "same scene and style, different subject"}),
    "synood_ablation": ("ablation_grid", {
        "components": ["projection", "FT label", "neg image", "grad image"],
        "metrics": ["AUROC ↑", "FPR95 ↓"],
        "rows": [{"marks": [0, 0, 0, 0], "values": ["94.21", "25.40"]},
                 {"marks": [1, 0, 1, 0], "values": ["94.85", "22.59"]},
                 {"marks": [1, 0, 0, 1], "values": ["96.21", "17.29"]},
                 {"marks": [1, 1, 1, 0], "values": ["95.01", "21.93"]},
                 {"marks": [1, 1, 0, 1], "values": ["97.01", "14.27"],
                  "best": True}],
        "verdict": "the iterative generation is what carries the gain"}),
    "synthetic_amount": ("peak_curve", {
        "xs": ["1k", "5k", "10k", "20k", "30k", "50k", "75k", "100k"],
        "ys": [95.7, 96.0, 96.3, 96.6, 96.85, 97.01, 96.6, 96.3],
        "peak": 5, "xlabel": "synthetic OOD samples", "ylabel": "AUROC",
        "peak_label": "50k", "overshoot_label": "too many labels become learnable",
        "verdict": "more data helps until the negative-label balance tips"}),

    # === CMA: Cross-Modal Alignment (Kim & Hwang, CVPR 2025) ==============
    "cma_tradeoff": ("tradeoff_scatter", {
        "xlabel": "ImageNet-1k ID accuracy", "ylabel": "1 − FPR95 on 4 OoD sets",
        "xlim": [65, 85], "ylim": [50, 82],
        "corner_note": "better is up and to the right",
        "groups": [
            {"name": "zero-shot", "colour": "out",
             "points": [[66.6, 72.0, "NegLabel"], [66.6, 56.4, "MCM"]]},
            {"name": "prompt learning", "colour": "dim",
             "points": [[71.95, 73.7, "CoOp"], [71.72, 62.2, "LoCoOp"]]},
            {"name": "single-modal fine-tuning", "colour": "neg",
             "points": [[79.22, 53.9, "LP"], [81.44, 60.5, "FFT"],
                        [81.48, 55.1, "LP-FT"]]},
            {"name": "multi-modal fine-tuning", "colour": "in",
             "points": [[82.58, 73.6, "FLYP"], [82.67, 75.6, "m²-mix"],
                        [82.64, 80.1, "CMA", True]]}],
        "verdict": "zero-shot detects well but classifies poorly; fine-tuning does the reverse"}),
    "cma_hypersphere": ("hypersphere_shift", {
        "sphere_label": "CLIP's hyperspherical embedding space",
        "groups": [{"name": "ID images", "colour": "in", "from": 205, "to": 168,
                    "n": 16, "spread": 26},
                   {"name": "ID texts", "colour": "hot", "from": 120, "to": 158,
                    "n": 14, "spread": 24},
                   {"name": "OoD texts", "colour": "neg", "from": 20, "to": 14,
                    "n": 12, "spread": 22, "marker": "s"}],
        "bars": {"title": "similarity to the dog image",
                 "labels": ["cat", "<OoD label>", "dog"],
                 "before": [0.55, 0.62, 0.72], "after": [0.50, 0.30, 0.95]},
        "before_note": "images and texts sit in separate clusters — the modality gap",
        "after_note": "CMA pulls ID images and ID texts together, leaving OoD texts behind",
        "verdict": "the gap barely hurts classification, but it blurs ID versus OoD text"}),
    "clip_loss_terms": ("annotated_equation", {
        "latex": r"$\mathcal{L}^{k}_{\mathrm{image}}=-(i_k\cdot t_k)/\tau"
                 r"+\log\sum_{j=1}^{B}\exp((i_k\cdot t_j)/\tau)$",
        "terms": [{"text": "pull the positive text in", "at": 0.30,
                   "note": "alignment", "colour": "in"},
                  {"text": "push every text away", "at": 0.76,
                   "note": "including the positive one", "colour": "neg"}],
        "verdict": "the second term is not uniformity — it repels texts from images"}),
    "cma_objective": ("annotated_equation", {
        "latex": r"$\mathcal{L}_{\mathrm{CMA}}=\mathcal{L}_{\mathrm{CLIP}}"
                 r"+\frac{\lambda}{2B}\sum_{k=1}^{B}"
                 r"(\mathcal{L}^{k}_{\mathrm{image_{CMA}}}"
                 r"+\mathcal{L}^{k}_{\mathrm{text_{CMA}}})$",
        "terms": [{"text": "the usual contrastive loss", "at": 0.26,
                   "note": "keeps ID accuracy", "colour": "in"},
                  {"text": "raise similarity to every ID text", "at": 0.74,
                   "note": r"strength $\lambda$", "colour": "hot"}],
        "verdict": "equivalently: down-weight the repulsion to " + r"$(1-\lambda)$"}),
    "cma_ebm": ("annotated_equation", {
        "latex": r"$\max_\theta\ \mathbb{E}_p[\log q_\theta(t|i)]"
                 r"\ +\ \mathbb{E}_p[\log q_\theta(i)]$",
        "terms": [{"text": "discriminative", "at": 0.36,
                   "note": "what contrastive loss already does", "colour": "dim"},
                  {"text": "generative", "at": 0.80,
                   "note": "what the CMA term adds", "colour": "hot"}],
        "verdict": "on a hypersphere the partition function drops out"}),
    "cma_metrics": ("metric_table", {
        "columns": ["Uni-All", "Uni-I", "Uni-T", "Uni-CM", "Uni-CMM",
                    "Align-ID", "Align-OoD"],
        "rows": [{"label": "Zero-shot", "values": ["1.594", "0.874", "1.320",
                                                   "2.346", "2.141", "0.023", "0.035"]},
                 {"label": "CoOp", "values": ["1.702", "0.874", "1.701", "2.343",
                                              "2.073", "0.032", "0.069"]},
                 {"label": "FLYP", "values": ["1.821", "1.395", "1.215", "2.437",
                                              "2.124", "0.052", "0.068"]},
                 {"label": "m²-mix", "values": ["1.208", "0.947", "1.441", "1.584",
                                                "1.014", "0.104", "0.130"]},
                 {"label": "CMA", "values": ["0.862", "0.719", "0.993", "1.323",
                                             "0.725", "0.114", "0.138"],
                  "best": True}],
        "verdict": "smallest cross-modal distance, largest separation from OoD text"}),
    "cma_embeddings": ("embedding_grid", {
        "panels": [
            {"label": "Zero-shot", "groups": [
                {"colour": "in", "angle": 210, "spread": 30, "n": 18},
                {"colour": "hot", "angle": 110, "spread": 26, "n": 14},
                {"colour": "neg", "angle": 20, "spread": 26, "n": 12}]},
            {"label": "FLYP", "groups": [
                {"colour": "in", "angle": 205, "spread": 34, "n": 18},
                {"colour": "hot", "angle": 115, "spread": 30, "n": 14},
                {"colour": "neg", "angle": 25, "spread": 28, "n": 12}]},
            {"label": "m²-mix", "groups": [
                {"colour": "in", "angle": 190, "spread": 40, "n": 18},
                {"colour": "hot", "angle": 150, "spread": 34, "n": 14},
                {"colour": "neg", "angle": 30, "spread": 30, "n": 12}]},
            {"label": "CMA", "groups": [
                {"colour": "in", "angle": 175, "spread": 46, "n": 18},
                {"colour": "hot", "angle": 168, "spread": 40, "n": 14},
                {"colour": "neg", "angle": 20, "spread": 24, "n": 12}]}],
        "legend": [["ID image", "in"], ["ID text", "hot"], ["OoD text", "neg"]],
        "verdict": "only CMA merges the ID clusters while keeping OoD text apart"}),

    # === GOOD: Guided OOD Sampling (Gao et al., NeurIPS 2025) =============
    "dreamood_problem": ("panels", {
        "fmt": "{:.0%}",
        "panels": [{"title": "Misalignment", "beat": 0,
                    "bars": [["latent nudge", 0.15, "in"],
                             ["image change", 0.95, "neg"]],
                    "verdict": "semantics drift unpredictably"},
                   {"title": "Limited diversity", "beat": 1,
                    "bars": [["semantic", 0.90, "in"],
                             ["covariate", 0.12, "neg"]],
                    "verdict": "only one axis of shift"}],
        "verdict": "both come from perturbing text embeddings instead of pixels"}),
    "good_framework": ("pipeline", {
        "stages": [["ID image\n+ class prompt", "in"], ["Stable\nDiffusion", "mid"],
                   ["guided\nsampling", "hot"], ["synthetic\noutliers", "out"],
                   ["outlier exposure\ntraining", "in"],
                   ["unified OOD\nscore at test", "hot"]],
        "per_beat": 2,
        "verdict": "one stage, not four — no latent space to align first",
        "note": "the classifier is off-the-shelf and frozen throughout"}),
    "good_trajectory": ("trajectory_guidance", {
        "levels": 5,
        "level_labels": [r"$\mathcal{M}_T$", r"$\mathcal{M}_t$", r"$\mathcal{M}_0$"],
        "start_label": r"$x_T\sim\mathcal{N}(0,I)$", "end_label": r"$x_0$",
        "path_label": "DDIM denoising step",
        "deflections": [{"at": 1, "label": r"$\Delta_t$  variance guidance",
                         "angle": 26, "colour": "hot"},
                        {"at": 3, "label": r"$\Delta_0$  mean guidance",
                         "angle": 62, "colour": "out"}],
        "verdict": "the classifier bends the trajectory at every step"}),
    "good_energy": ("energy_walk", {
        "xlabel": "pixel space", "ylabel": r"model density $p_\phi(x)$",
        "peak_label": "ID lives here", "tail_label": "low-likelihood region",
        "marker_label": r"$x_{0|t}$",
        "arrow_label": r"climb $+\nabla_x E_\phi(x)$",
        "verdict": "free energy is a negative log-density, so its gradient points out"}),
    "good_knn": ("knn_sparsity", {
        "k": 5, "dense_label": "ID feature cluster",
        "sparse_label": "feature-sparse region",
        "marker_label": r"$z(x)$",
        "radius_label": r"guidance climbs $\nabla_x D_k(x)$ — the k-NN radius grows",
        "verdict": "rare features, not just low pixel likelihood"}),
    "good_energy_eq": ("annotated_equation", {
        "latex": r"$E_\phi(x)=-\log\sum_{k=1}^{C}e^{f_k(x;\phi)}$",
        "terms": [{"text": "the classifier's own logits", "at": 0.72,
                   "note": "no extra training", "colour": "in"},
                  {"text": "free energy", "at": 0.22,
                   "note": "negative log-density up to Z", "colour": "hot"}],
        "verdict": "any discriminative classifier is secretly an energy model"}),
    "good_step": ("annotated_equation", {
        "latex": r"$x_{t-1}=\mathrm{Sample}(x_t,x_{0|t},t)"
                 r"+\Delta_t/\sqrt{\alpha_t}+\sqrt{\bar{\alpha}_{t-1}}\,\Delta_0$",
        "terms": [{"text": "plain DDIM", "at": 0.28, "note": "unchanged",
                   "colour": "dim"},
                  {"text": "guidance on the noisy sample", "at": 0.62,
                   "note": r"$\rho_t\nabla_{x_t}$", "colour": "hot"},
                  {"text": "guidance on the clean estimate", "at": 0.88,
                   "note": r"$\mu_t\nabla_{x_{0|t}}$", "colour": "out"}],
        "verdict": "two gradients, both from the same frozen classifier"}),
    "good_stepsize": ("param_grid", {
        "rows": ["0.1", "0.5", "1.0", "2.0", "5.0"],
        "cols": ["0.1", "0.5", "1.0", "2.0", "5.0"],
        "row_label": r"mean guidance $\bar{\mu}$",
        "col_label": r"variance guidance $\bar{\rho}$",
        "shift": 0.5,
        "note": "energy of 500 generated samples per cell",
        "verdict": "no single cell is right — they sample the whole grid"}),
    "good_examples": ("pair_strip", {
        "columns": ["hen", "tiger", "zebra", "otter", "violin", "swing", "pig"],
        "row_a": r"GOOD$_{img}$", "row_b": r"GOOD$_{feat}$",
        "colour_a": "hot", "colour_b": "in",
        "verdict": "the two guidances fail the classifier in different ways"}),
    "good_unified": ("ratio_slider", {
        "a_label": r"image-level  $\hat{E}(x;f)$",
        "b_label": r"feature-level  $D(x;f)$",
        "a_short": "1−w", "b_short": "w",
        "steps": [[90, 10], [70, 30], [45, 55], [25, 75], [10, 90]],
        "notes": [r"$w=1-\exp(-a\cdot \mathrm{KL}(q_{test}\|q_{ID}))$",
                  "large feature-space shift ⇒ trust the k-NN distance more"]}),
    "good_ablation": ("ablation_grid", {
        "components": ["GOOD_img", "GOOD_feat", "balanced sampling",
                       "unified score"],
        "metrics": ["FPR95 ↓", "AUROC ↑", "ID Acc ↑"],
        "rows": [{"marks": [0, 0, 0, 0], "values": ["48.68", "89.46", "87.64"]},
                 {"marks": [1, 0, 1, 0], "values": ["32.23", "92.29", "86.98"]},
                 {"marks": [0, 1, 1, 0], "values": ["32.93", "91.98", "87.24"]},
                 {"marks": [1, 1, 1, 0], "values": ["30.82", "92.81", "87.16"]},
                 {"marks": [1, 1, 1, 1], "values": ["17.70", "96.28", "87.16"],
                  "best": True}],
        "verdict": "the unified score is the single biggest jump"}),
    "good_stages": ("ablation_grid", {
        "components": ["manual\ndata", "align\nlatents", "sample\nembeds",
                       "generate", "train"],
        "metrics": ["stages"],
        "rows": [{"marks": [0, 1, 1, 1, 1], "values": ["4"]},
                 {"marks": [0, 1, 1, 1, 1], "values": ["4"]},
                 {"marks": [0, 0, 0, 1, 1], "values": ["2"], "best": True}],
        "verdict": "Dream-OOD reports over 16 h on ImageNet-100; GOOD skips two stages"}),

    # === LeJEPA (Balestriero & LeCun, arXiv:2511.08544) ====================
    "jepa_blueprint": ("siamese_predict", {
        "view_a": "view 1", "view_b": "view 2",
        "encoder": r"$f_\theta$", "shared_label": "shared weights",
        "loss_label": "predict one\nfrom the other",
        "collapse_label": "...or collapse",
        "verdict": "the prediction task alone is satisfied by a constant"}),
    "collapse_modes": ("distribution_shapes", {
        "shapes": [{"label": "what we want", "kind": "ball", "good": True,
                    "colour": "in"},
                   {"label": "complete collapse", "kind": "point"},
                   {"label": "dimensional collapse", "kind": "line"},
                   {"label": "anisotropic", "kind": "aniso"}],
        "verdict": "three ways to win the prediction task without learning anything"}),
    "jepa_heuristics": ("strike_list", {
        "title": "what current JEPAs need to avoid collapse",
        "items": ["stop-gradient", "teacher-student with EMA schedule",
                  "asymmetric view generation", "whitening / normalisation layers",
                  "negative samples", "hyper-parameter schedulers"],
        "kept_title": "what LeJEPA needs",
        "kept": [r"one trade-off $\lambda$"],
        "verdict": "each is under-specified; combined, they are a game of whack-a-mole"}),
    "lejepa_variance": ("boundary_spread", {
        "n_lines": 10,
        "left": {"label": "isotropic embeddings", "spread": 0.06,
                 "scale_x": 1.0, "scale_y": 1.0,
                 "note": r"Var$(\hat\beta)=0.0056$"},
        "right": {"label": "anisotropic embeddings", "spread": 0.30,
                  "scale_x": 1.6, "scale_y": 0.45,
                  "note": r"Var$(\hat\beta)=0.0801$ — condition number 20"},
        "verdict": "same features, same energy — but a 14x worse estimator"}),
    "lejepa_probe_bias": ("annotated_equation", {
        "latex": r"$\hat\beta=\arg\min_\beta\|\mathbf{y}-\mathbf{Z}\beta\|_2^2"
                 r"+\lambda\|\beta\|_2^2$",
        "terms": [{"text": "unknown downstream task", "at": 0.30,
                   "note": "we must be good for any y", "colour": "out"},
                  {"text": "the embeddings we control", "at": 0.56,
                   "note": "only their distribution", "colour": "in"},
                  {"text": "ridge penalty", "at": 0.86,
                   "note": "anisotropy biases through it", "colour": "neg"}],
        "verdict": "Lemmas 1 and 2: anisotropy amplifies both bias and variance"}),
    "lejepa_optimal": ("annotated_equation", {
        "latex": r"$\mathrm{ISB}_{k\text{-NN}}=\frac{r_0^4}{(K+2)^2}"
                 r"\tau_g^2\,J(p)+\mathcal{O}(r_0^4)$",
        "terms": [{"text": "Fisher-information-like functional of the density",
                   "at": 0.74, "note": "minimised by the isotropic Gaussian",
                   "colour": "hot"}],
        "verdict": "Theorem 1: unique minimiser under a covariance constraint"}),
    "sigreg_slicing": ("sliced_projection", {
        "cloud": "x", "n_dirs": 5, "bad_dir": 2,
        "cloud_label": r"embeddings $p_z$",
        "target_label": r"target $\mathcal{N}(0,I)$",
        "note": "marginals are standard Gaussian, yet the joint is degenerate",
        "verdict": "one direction is enough to expose it"}),
    "cramer_wold": ("annotated_equation", {
        "latex": r"$\langle\mathbf{u},X\rangle\ \sim\ \langle\mathbf{u},Y\rangle"
                 r"\quad\forall\,\mathbf{u}\in\mathbb{S}^{d-1}"
                 r"\quad\Longleftrightarrow\quad X\sim Y$",
        "terms": [{"text": "every 1-D projection agrees", "at": 0.24,
                   "note": "cheap univariate tests", "colour": "in"},
                  {"text": "the full joint agrees", "at": 0.86,
                   "note": "what we actually want", "colour": "hot"}],
        "verdict": "Lemma 3: slicing loses nothing, in principle"}),
    "sigreg_tests": ("ablation_grid", {
        "components": ["differentiable", "bounded\ngradient", "no sorting",
                       "identifiable", "linear\ncost"],
        "metrics": ["verdict"],
        "rows": [{"marks": [1, 0, 1, 0, 1], "values": ["moments"]},
                 {"marks": [0, 1, 0, 1, 0], "values": ["CDF"]},
                 {"marks": [1, 1, 1, 1, 1], "values": ["Epps-Pulley"],
                  "best": True}],
        "verdict": "only the characteristic-function test satisfies all five"}),
    "epps_pulley": ("annotated_equation", {
        "latex": r"$EP=N\int_{-\infty}^{\infty}\left|\hat\phi_X(t)-\phi(t)\right|^2"
                 r"w(t)\,dt$",
        "terms": [{"text": "empirical characteristic function", "at": 0.30,
                   "note": r"$\frac{1}{n}\sum e^{itX_j}$ — an average, so all-reduce works",
                   "colour": "in"},
                  {"text": "the target's CF", "at": 0.58,
                   "note": "closed form for a Gaussian", "colour": "hot"},
                  {"text": "Gaussian window", "at": 0.86,
                   "note": "keeps the integral finite", "colour": "out"}],
        "verdict": "bounded loss, bounded gradient, bounded curvature (Theorem 4)"}),
    "sigreg_directions": ("curve_family", {
        "xlabel": "number of directions  M  (log scale)",
        "ylabel": "expected directional statistic",
        "hi_label": "worse", "lo_label": "better", "logx": True,
        "series": [{"name": "fixed", "colour": "dim",
                    "points": [[64, 1350], [128, 1180], [256, 980],
                               [512, 790], [1024, 610], [2048, 430]]},
                   {"name": "resampled", "colour": "hot",
                    "points": [[64, 190], [128, 160], [256, 135],
                               [512, 115], [1024, 95], [2048, 80]]}],
        "note": "resampling every minibatch beats a large fixed set",
        "verdict": "SGD itself defeats the curse of dimensionality"}),
    "lejepa_loss": ("annotated_equation", {
        "latex": r"$\mathcal{L}_{\mathrm{LeJEPA}}=\lambda\cdot\mathrm{SIGReg}"
                 r"+(1-\lambda)\cdot\mathcal{L}_{\mathrm{pred}}$",
        "terms": [{"text": "isotropic Gaussian constraint", "at": 0.44,
                   "note": "replaces every heuristic", "colour": "hot"},
                  {"text": "views predict their mean", "at": 0.82,
                   "note": r"$\|\mu_n-z_{n,v}\|^2$", "colour": "in"}],
        "verdict": "one hyperparameter, about 50 lines of PyTorch"}),
    "lejepa_architectures": ("tradeoff_scatter", {
        "xlabel": "parameters (millions)", "ylabel": "ImageNet-10 top-1 (%)",
        "xlim": [0, 20], "ylim": [90.5, 95.8],
        "corner_note": "50 timm models, LeJEPA out of the box",
        "groups": [
            {"name": "convnext", "colour": "in",
             "points": [[3.5, 92.4, ""], [5.2, 92.9, ""], [15.6, 93.6, ""]]},
            {"name": "resnet", "colour": "out",
             "points": [[5.4, 95.0, ""], [11.2, 94.5, ""], [16.0, 94.0, ""]]},
            {"name": "maxvit", "colour": "hot",
             "points": [[7.4, 94.9, ""], [15.5, 93.4, ""]]},
            {"name": "levit / efficientnet", "colour": "dim",
             "points": [[9.2, 92.6, ""], [12.4, 91.7, ""], [18.9, 93.0, ""]]}],
        "verdict": "91.5% to 95% across 8 families — no per-architecture tuning"}),
    "lejepa_loss_probe": ("tradeoff_scatter", {
        "xlabel": "LeJEPA training loss (lower is better)",
        "ylabel": "downstream linear-probe accuracy (%)",
        "xlim": [0, 10], "ylim": [15, 82],
        "corner_note": "each point is one hyperparameter setting",
        "groups": [{"name": "ViT-base, ImageNet-1k", "colour": "hot",
                    "points": [[1.0, 79.2, ""], [1.8, 72.9, ""], [3.0, 64.1, ""],
                               [4.6, 52.4, ""], [6.6, 38.5, ""], [8.8, 18.9, ""]]}],
        "verdict": "Spearman 85%, and 99% after rescaling by lambda"}),
    "lejepa_indomain": ("curve_family", {
        "xlabel": "labelled samples per class (log scale)",
        "ylabel": "Galaxy10 top-1 accuracy (%)",
        "hi_label": "better", "lo_label": "worse", "logx": True,
        "series": [{"name": "LeJEPA", "colour": "hot",
                    "points": [[1, 29.4], [5, 45.0], [10, 55.0], [100, 68.0],
                               [1000, 78.0], [11000, 82.7]]},
                   {"name": "DINOv3", "colour": "dim",
                    "points": [[1, 24.7], [5, 38.0], [10, 47.0], [100, 62.0],
                               [1000, 74.0], [11000, 81.6]]},
                   {"name": "DINOv2", "colour": "out",
                    "points": [[1, 21.1], [5, 34.0], [10, 43.0], [100, 58.0],
                               [1000, 71.0], [11000, 78.3]]}],
        "note": "11k training images, 10 galaxy morphologies",
        "verdict": "in-domain pretraining beats frontier transfer at every budget"}),

    # === Dreamer 4 (Hafner, Yan & Lillicrap, arXiv:2509.24527) ============
    "d4_why_worldmodel": ("pipeline", {
        "stages": [["offline\nvideo + actions", "in"], ["world\nmodel", "hot"],
                   ["imagined\nrollouts", "out"], ["RL on the\npolicy", "in"],
                   ["deploy", "mid"]],
        "per_beat": 2,
        "verdict": "no environment interaction anywhere in the loop",
        "note": "which is the point: online rollouts are unsafe for robots"}),
    "d4_gap": ("quadrant_map", {
        "xlabel": "breadth of the data distribution it can fit",
        "ylabel": "accuracy of object interactions",
        "lo_x": "one narrow domain", "hi_x": "diverse real video",
        "lo_y": "hallucinated", "hi_y": "precise physics",
        "goal_label": "what imagination training needs",
        "items": [{"label": "Dreamer 3", "x": 0.14, "y": 0.82,
                   "note": "fast, but narrow"},
                  {"label": "Genie 3 / video models", "x": 0.84, "y": 0.24,
                   "note": "broad, but imprecise and slow"},
                  {"label": "Dreamer 4", "x": 0.74, "y": 0.80, "best": True}],
        "verdict": "qualitative positioning — the axes are arguments, not measurements"}),
    "d4_architecture": ("factorized_attention", {
        "times": 7, "spaces": 4, "n_layers": 12, "every": 4,
        "space_label": "space attention\n(every layer)",
        "time_label": "time attention\n(every 4th layer)",
        "layer_label": "layer stack",
        "note": "block-causal in time, plus grouped-query attention to shrink the KV cache",
        "verdict": "dense attention over all video tokens would be unaffordable"}),
    "d4_tokenizer": ("bottleneck_codec", {
        "grid": 6, "drop": 0.45, "n_latents": 5,
        "in_label": "image patches", "out_label": "reconstruction",
        "latent_label": "learned\nlatent tokens",
        "bottleneck_label": "low-dim\n+ tanh",
        "note": r"$\mathcal{L}=\mathcal{L}_{MSE}+0.2\,\mathcal{L}_{LPIPS}$"
                "   ·   causal in time, so frames decode one by one",
        "verdict": "masked autoencoding is what improves the spatial consistency downstream"}),
    "d4_flow": ("interpolation_path", {
        "start_label": r"noise  $x_0\sim\mathcal{N}(0,I)$",
        "end_label": r"data  $x_1$",
        "marks": [0.0, 0.25, 0.5, 0.75, 1.0],
        "tau_label": r"signal level  $\tau$",
        "velocity_label": r"the network predicts $v=x_1-x_0$",
        "note": r"$x_\tau=(1-\tau)x_0+\tau x_1$",
        "verdict": "flow matching: a straight path, and a velocity field along it"}),
    "d4_shortcut": ("step_schedule", {
        "fine_steps": 16, "coarse_steps": 4,
        "start_label": "noise", "end_label": "frame",
        "fine_label": "diffusion (64+ steps)", "coarse_label": "shortcut (K=4)",
        "distill_label": "bootstrap: two half-steps become one",
        "verdict": "conditioning on the step size buys real-time inference"}),
    "d4_forcing": ("noise_ladder", {
        "frames": [{"label": "t-4", "tau": 0.9}, {"label": "t-3", "tau": 0.9},
                   {"label": "t-2", "tau": 0.9}, {"label": "t-1", "tau": 0.9},
                   {"label": "t", "tau": 0.35}, {"label": "t+1", "tau": 0.0}],
        "tau_label": r"signal $\tau$",
        "note": r"at inference the history is lightly corrupted to $\tau_{ctx}=0.1$,"
                " so the model tolerates its own imperfect generations",
        "verdict": "every timestep is both a denoising task and context for later ones"}),
    "d4_xpred": ("annotated_equation", {
        "latex": r"$\mathcal{L}(\theta)=\|\hat{z}_1-z_1\|_2^2"
                 r"\quad\text{vs}\quad\|\hat{v}-v\|_2^2$",
        "terms": [{"text": "x-prediction", "at": 0.26,
                   "note": "predict the clean frame", "colour": "hot"},
                  {"text": "v-prediction", "at": 0.76,
                   "note": "high-frequency outputs that accumulate error",
                   "colour": "neg"}],
        "verdict": "x-prediction is what makes arbitrarily long rollouts stable"}),
    "d4_rampweight": ("energy_walk", {
        "xlabel": r"signal level $\tau$", "ylabel": "learning signal",
        "peak_label": "most learning signal here",
        "tail_label": r"at $\tau\approx0$: predict the mean",
        "marker_label": r"$w(\tau)$",
        "arrow_label": "",
        "verdict": "a one-line loss weight, no schedule"}),
    "d4_agent_tokens": ("modality_attention", {
        "modalities": ["image tokens", "action tokens", "register tokens",
                       "agent tokens"],
        "matrix": [[1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 0],
                   [1, 1, 1, 1]],
        "highlight_row": 3,
        "verdict": "agent tokens see everything; nothing sees them back"}),
    "d4_pmpo": ("threshold_line", {
        "threshold": 0.5, "n": 34,
        "rule": r"sign of $A_t=R^\lambda_t-v_t$",
        "lo_label": r"$\mathcal{D}^-$  push down",
        "hi_label": r"$\mathcal{D}^+$  push up",
        "radius_label": "",
        "verdict": "PMPO uses only the sign, so no return or advantage normalisation"}),
    "d4_imagination": ("arrow_cycle", {
        "stages": [["context from data", "in"], ["policy picks action", "hot"],
                   ["world model predicts", "out"],
                   ["reward + value heads", "neg"], ["update policy", "hot"]],
        "centre_label": "one imagined step",
        "verdict": "the transformer stays frozen; only the policy and value heads move"}),
    "d4_phases": ("timeline", {
        "events": [{"label": "Pretrain", "note": "tokenizer, then world model"},
                   {"label": "Finetune", "note": "insert task tokens, BC + reward"},
                   {"label": "Imagine", "note": "RL on rollouts"}],
        "axis_label": "three phases, one transformer",
        "verdict": "each phase reuses the same weights and the same losses"}),
    "d4_milestones": ("milestone_chain", {
        "unit": "%",
        "stages": [{"label": "logs", "value": 99}, {"label": "planks", "value": 98},
                   {"label": "sticks", "value": 97},
                   {"label": "cobble", "value": 96},
                   {"label": "stone pick", "value": 93},
                   {"label": "iron ore", "value": 65},
                   {"label": "iron pick", "value": 29},
                   {"label": "diamond", "value": 0.7, "colour": "hot"}],
        "compare": [{"name": "VPT", "values": [58, 67, 53, 40, 23, 11, 0.6, 0]},
                    {"name": "VLA", "values": [90, 84, 86, 54, 46, 26, 11, 0]}],
        "verdict": "first agent to reach diamonds from offline data alone"}),
    "d4_worldmodels": ("metric_table", {
        "columns": ["params", "resolution", "context", "FPS", "tasks solved"],
        "rows": [{"label": "MineWorld", "values": ["1.2B", "384x224", "0.8 s", "2", "—"]},
                 {"label": "Lucid-v1", "values": ["1.1B", "640x360", "1.0 s", "44", "0 / 16"]},
                 {"label": "Oasis (small)", "values": ["500M", "640x360", "1.6 s", "20", "0 / 16"]},
                 {"label": "Oasis (large)", "values": ["—", "360x360", "1.6 s", "~5", "5 / 16"]},
                 {"label": "Dreamer 4", "values": ["2B", "640x360", "9.6 s", "21", "14 / 16"],
                  "best": True}],
        "verdict": "6x the context of prior models, still real-time on one H100"}),
    "d4_actions": ("curve_family", {
        "series": [{"name": "SSIM", "colour": "hot",
                    "points": [[1, 0], [10, 75], [100, 100], [1000, 100], [2541, 100]]},
                   {"name": "PSNR", "colour": "in",
                    "points": [[1, 0], [10, 53], [100, 85], [1000, 95], [2541, 100]]}],
        "xlabel": "hours of video paired with actions (log scale)",
        "ylabel": "% of the all-actions model", "logx": True,
        "hi_label": "better", "lo_label": "worse",
        "note": "2541 hours of video total",
        "verdict": "100 hours of actions recovers almost all of the conditioning"}),
    "d4_ablation": ("metric_table", {
        "columns": ["train step (s)", "inference FPS", "FVD"],
        "rows": [{"label": "Diffusion forcing transformer", "values": ["9.8", "9.1", "306"]},
                 {"label": "+ fewer steps (K=4)", "values": ["9.8", "9.1", "875"]},
                 {"label": "+ shortcut model", "values": ["9.8", "9.1", "329"]},
                 {"label": "+ x-prediction and x-loss", "values": ["9.8", "9.1", "151"]},
                 {"label": "+ ramp weight", "values": ["9.8", "9.1", "102"]},
                 {"label": "+ architecture (all)", "values": ["0.8", "21.4", "91"]}],
        "verdict": "K=4 alone wrecks quality; the shortcut objective is what rescues it"}),

    # === Attention Is All You Need — animated figures ======================
    "attn_rnn_vs_attn": ("sequential_vs_parallel", {
        "tokens": ["The", "animal", "didn't", "cross", "the", "street"],
        "top_label": "Recurrence", "top_note": "O(n) sequential steps",
        "bottom_label": "Self-attention", "bottom_note": "O(1) sequential steps",
        "rate": 0.55,
        "verdict": "the recurrent chain is the thing that cannot be parallelised"}),
    "attn_lookup": ("attention_lookup", {
        "tokens": ["The", "animal", "didn't", "cross", "the", "street", "it"],
        "query": 6, "scores": [0.4, 3.2, 0.3, 0.9, 0.4, 1.6, 0.5],
        "key_label": "keys and values (every position)",
        "query_label": r'query: "it"',
        "output_label": "output = weighted sum of values",
        "note": "the weights are computed, not learned — they change with every input",
        "verdict": '"it" resolves to "animal" because that key scores highest'}),
    "attn_scaling": ("saturation_curve", {
        "curve": "logistic", "xlim": 9.0,
        "xlabel": "gap between the top logit and the rest",
        "ylabel": "softmax weight on the top key",
        "flat_label": "flat — no gradient",
        "slope_label": "gradient",
        "points": [{"label": "unscaled", "x": 6.4, "colour": "neg",
                    "note": r"$d_k=64$, so scores scale like $\sqrt{64}=8$"},
                   {"label": r"scaled by $1/\sqrt{d_k}$", "x": 0.8,
                    "colour": "hot", "note": "variance back to 1"}],
        "note": "the tangent is the gradient the softmax passes backwards",
        "verdict": "saturation is not just a peaked distribution — it is a dead gradient"}),
    "attn_heads": ("head_patterns", {
        "tokens": ["The", "animal", "didn't", "cross", "the", "street",
                   "because", "it", "was", "tired"],
        "heads": [{"name": "head 1", "weights": [0.1, 0.9, 0.1, 0.2, 0.1, 0.2, 0.1, 0.3, 0.1, 0.2]},
                  {"name": "head 2", "weights": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.9, 0.2, 0.1, 0.6]},
                  {"name": "head 3", "weights": [0.6, 0.2, 0.1, 0.7, 0.5, 0.3, 0.1, 0.2, 0.4, 0.1]},
                  {"name": "head 4", "weights": [0.1, 0.2, 0.8, 0.3, 0.1, 0.2, 0.2, 0.2, 0.7, 0.2]}],
        "verdict": "one attention is one averaging; h heads let it average several ways at once"}),
    "attn_complexity": ("metric_table", {
        "columns": ["complexity per layer", "sequential ops", "max path length"],
        "rows": [{"label": "Self-attention",
                  "values": [r"$O(n^2 d)$", r"$O(1)$", r"$O(1)$"], "best": True},
                 {"label": "Recurrent",
                  "values": [r"$O(n d^2)$", r"$O(n)$", r"$O(n)$"]},
                 {"label": "Convolutional",
                  "values": [r"$O(k n d^2)$", r"$O(1)$", r"$O(\log_k n)$"]},
                 {"label": "Self-attn (restricted)",
                  "values": [r"$O(r n d)$", r"$O(1)$", r"$O(n/r)$"]}],
        "verdict": r"self-attention is cheaper per layer whenever $n < d$"}),
    "attn_results": ("metric_table", {
        "columns": ["EN-DE BLEU", "EN-FR BLEU", "training FLOPs"],
        "rows": [{"label": "GNMT + RL", "values": ["24.6", "39.92", "1.4e20"]},
                 {"label": "ConvS2S", "values": ["25.16", "40.46", "1.5e20"]},
                 {"label": "MoE", "values": ["26.03", "40.56", "1.2e20"]},
                 {"label": "Transformer (base)", "values": ["27.3", "38.1", "3.3e18"]},
                 {"label": "Transformer (big)", "values": ["28.4", "41.8", "2.3e19"],
                  "best": True}],
        "verdict": "new state of the art at a fraction of the training cost"}),

    # === Intuitive Physics from Interaction (Schulze Buschoff et al., 2026) =
    "ip_hypothesis": ("panels", {
        "fmt": "{:.0f}",
        "panels": [{"title": "Passive observation", "beat": 0,
                    "bars": [["observes", 90, "in"],
                             ["intervenes", 10, "neg"]],
                    "verdict": "supervised fine-tuning"},
                   {"title": "Active interaction", "beat": 1,
                    "bars": [["observes", 90, "in"],
                             ["intervenes", 90, "hot"]],
                    "verdict": "reinforcement learning"}],
        "verdict": "the cognitive-science claim the paper sets out to test"}),

    "ip_principles": ("physics_principles", {
        "principles": [{"name": "support", "kind": "support",
                        "note": "unsupported things fall"},
                       {"name": "solidity", "kind": "solidity",
                        "note": "solids do not interpenetrate"},
                       {"name": "permanence", "kind": "permanence",
                        "note": "hidden things still exist"},
                       {"name": "continuity", "kind": "continuity",
                        "note": "objects move on connected paths"}],
        "verdict": "the implicit expectations infants already have — and VLMs largely do not"}),
    "ip_shortcut": ("shortcut_failure", {
        "left": {"label": "training distribution", "blocks": 3, "offset": 0.35,
                 "truth": "stable"},
        "right": {"label": "shifted distribution", "blocks": 4, "offset": 0.35,
                  "truth": "unstable"},
        "cue_label": "the cue the model reads:",
        "shortcut_says": "small offset  ->  say stable",
        "verdict": "the same surface cue, two different answers — a shortcut, not a rule"}),
    "ip_environment": ("object_stack", {
        "blocks": 4, "offset": 0.9, "displaced": "top",
        "block_label": "displaced block",
        "centre_label": "centre of the tower",
        "offset_label": "x-offset",
        "action_label": "the action: move it back",
        "note": "256x256 RGB from a fixed camera in ThreeDWorld, 2-4 cubes,"
                " everything else held constant",
        "verdict": "one number — the offset — decides both stability and the correct action"}),
    "ip_design": ("design_matrix", {
        "rows": ["top block", "side block"],
        "cols": ["binary stability", "x-only", "x-y"],
        "cells": [["is this tower stable?", "move it by how much?", "—"],
                  ["—", "move it by how much?", "move it sideways and up"]],
        "icons": [[{"blocks": 3, "offset": 0.9, "displaced": "top"},
                   {"blocks": 3, "offset": 0.9, "displaced": "top"},
                   None],
                  [None,
                   {"blocks": 3, "offset": 0.5, "displaced": "side"},
                   {"blocks": 3, "offset": 0.5, "displaced": "side"}]],
        "row_label": "dataset", "col_label": "action type",
        "note": "10,000 training images and 10,000 held-out per combination",
        "verdict": "four tasks that share visual statistics and the same physics"}),
    "ip_reward": ("reward_profile", {
        "xlabel": "distance from the optimal position",
        "ylabel": "reward", "ymin": -6.0, "ymax": 21.0,
        "optimum_label": "optimal position",
        "regimes": [{"label": "unparseable answer", "from": -3, "to": -1.9,
                     "level": -5, "colour": "neg"},
                    {"label": "below the floor", "from": 1.9, "to": 3,
                     "level": -4, "colour": "neg"},
                    {"label": "moves but unstable", "from": -1.9, "to": 1.9,
                     "level": 2, "base": -2, "gauss": True, "colour": "dim",
                     "label_y": -1.4},
                    {"label": "stable and taller", "from": -1.9, "to": 1.9,
                     "level": 20, "gauss": True, "colour": "hot",
                     "label_y": 20}],
        "note": "a Gaussian in distance, with a step change when the tower survives",
        "verdict": "the shaping is what makes a single scalar teach a continuous action"}),
    "ip_grpo": ("group_advantage", {
        "rewards": [19.2, 2.1, 18.6, -5, 17.9, 0.4, 19.6, -2.2,
                    18.1, 15.4, -5, 19.9, 3.3, 18.8, -1.1, 19.4],
        "prompt_label": "image +\nprompt",
        "mean_label": "group mean = the baseline",
        "adv_label": "advantage = (reward - mean) / std   ->   push up or down",
        "verdict": "no value network: the other 15 samples are the baseline"}),
    "ip_within_task": ("task_gallery", {
        "fmt": "{:.2f}",
        "cards": [{"label": "binary stability\ntop block", "action": "binary",
                   "blocks": 3, "offset": 0.9, "displaced": "top",
                   "note": "GRPO 0.94  ·  SFT 0.97", "value": 0.94},
                  {"label": "x-only\ntop block", "action": "x",
                   "blocks": 3, "offset": 0.9, "displaced": "top",
                   "note": "GRPO 20.00  ·  SFT 20.00", "value": 20.0,
                   "best": True},
                  {"label": "x-only\nside block", "action": "x",
                   "blocks": 3, "offset": 0.6, "displaced": "side",
                   "note": "GRPO 20.00  ·  SFT 20.00", "value": 20.0},
                  {"label": "x-y\nside block", "action": "xy",
                   "blocks": 3, "offset": 0.6, "displaced": "side",
                   "note": "GRPO 17.31  ·  SFT 19.86", "value": 17.31}],
        "verdict": "every model reaches ceiling on its own task — interaction buys nothing here"}),
    "ip_within_task_bars": ("bars", {
        "title": "within-task score as a fraction of that task's ceiling"
                 "   (binary ceiling 1.0, reward tasks 20)",
        "items": [["GRPO  binary stability  (0.94)", 0.943, "hot"],
                  ["SFT   binary stability  (0.97)", 0.969, "dim"],
                  ["GRPO  x-only top  (20.00)", 1.0, "hot"],
                  ["SFT   x-only top  (20.00)", 1.0, "dim"],
                  ["GRPO  x-y side  (17.31)", 0.866, "hot"],
                  ["SFT   x-y side  (19.86)", 0.993, "dim"]],
        "max": 1.05, "fmt": "{:.2f}",
        "verdict": "hypothesis 1 fails: no within-task advantage from interaction"}),
    "ip_generalization": ("value_matrix", {
        "rows": ["binary stability top", "x-only top", "x-only side", "x-y side"],
        "cols": ["binary stab. top", "x-only top", "x-only side", "x-y side"],
        "values": [[0.943, 0.788, 0.503, -0.264],
                   [10.626, 19.999, 0.042, 0.750],
                   [-0.373, -0.152, 19.998, 5.396],
                   [-0.723, -0.535, 3.738, 17.313]],
        "row_label": "evaluated on", "col_label": "trained on (GRPO)",
        "fmt": "{:.2f}",
        "diagonal_note": "the diagonal is near ceiling; almost everything else"
                         " sits at or below the task baseline",
        "verdict": "hypothesis 2 fails too — and the same matrix for SFT looks the same"}),
    "ip_real": ("task_gallery", {
        "fmt": "{:.2f}",
        "cards": [{"label": "trained on\nbinary stability", "action": "binary",
                   "blocks": 3, "offset": 0.8, "displaced": "top",
                   "note": "GRPO 0.60  ·  SFT 0.59", "value": 0.60,
                   "best": True},
                  {"label": "trained on\nx-only top", "action": "x",
                   "blocks": 3, "offset": 0.9, "displaced": "top",
                   "note": "GRPO 0.57  ·  SFT 0.53", "value": 0.57},
                  {"label": "trained on\nx-only side", "action": "x",
                   "blocks": 3, "offset": 0.6, "displaced": "side",
                   "note": "GRPO 0.52  ·  SFT 0.55", "value": 0.52},
                  {"label": "trained on\nx-y side", "action": "xy",
                   "blocks": 3, "offset": 0.6, "displaced": "side",
                   "note": "GRPO 0.31  ·  SFT 0.57", "value": 0.31}],
        "verdict": "chance is 0.50 — only the matching task transfers, and all sit below humans"}),
    "ip_real_bars": ("bars", {
        "title": "accuracy on 100 real wooden block towers (Lerer et al., 2016)",
        "items": [["GRPO, trained on binary stability", 0.60, "hot"],
                  ["SFT,  trained on binary stability", 0.59, "dim"],
                  ["GRPO, trained on x-only top", 0.57, "hot"],
                  ["GRPO, trained on x-only side", 0.52, "dim"],
                  ["GRPO, trained on x-y side", 0.31, "neg"],
                  ["chance", 0.50, "dim"]],
        "fmt": "{:.2f}",
        "verdict": "transfer only from the matching task, and all below the human average"}),
    "ip_probe": ("probe_layers", {
        "series": [{"name": "linear probe for tower stability", "colour": "hot",
                    "values": [0.72, 0.86, 0.93, 0.95, 0.96, 0.96, 0.97, 0.96,
                               0.97, 0.96]},
                   {"name": "linear probe for the x-offset", "colour": "in",
                    "values": [0.66, 0.80, 0.88, 0.91, 0.92, 0.93, 0.93, 0.92,
                               0.93, 0.92]}],
        "behaviour": 0.55, "behaviour_label": "what the base model actually answers",
        "chance": 0.5, "ylabel": "decoding accuracy", "xlabel": "layer",
        "gap_label": "competence  vs  performance",
        "verdict": "the information is there from the first layers — and unchanged by either method"}),
    "ip_multistep": ("frame_sequence", {
        "rows": [{"label": "single-image", "note": "one state, one action",
                  "frames": [{"offset": 0.7, "displaced": "side"}],
                  "actions": ["act"]},
                 {"label": "triplet", "note": "sees the consequences",
                  "frames": [{"offset": 0.9, "displaced": "side"},
                             {"offset": 0.5, "displaced": "side"},
                             {"offset": 0.2, "displaced": "side"}],
                  "actions": ["act", "act", "act"]}],
        "verdict": "same physics, same reward, same visuals — and neither transfers to the other"}),
    "ip_multistep_matrix": ("value_matrix", {
        "rows": ["single-image", "triplet (multi-step)"],
        "cols": ["single-image", "triplet (multi-step)"],
        "values": [[17.3, 1.2], [0.9, 16.8]],
        "row_label": "evaluated on", "col_label": "trained on",
        "fmt": "{:.1f}",
        "diagonal_note": "same visual statistics, same reward function,"
                         " same physics — and still no transfer",
        "verdict": "even seeing the consequence of an action does not help"}),

    # === SynOOD vs GOOD — comparison =======================================
    "vs_shared_idea": ("venn2", {
        "left": {"label": "SynOOD", "items": ["edit a real ID image",
                                              "MLLM picks the context",
                                              "in-paint, 3 iterations",
                                              "fine-tune CLIP"]},
        "right": {"label": "GOOD", "items": ["generate from a prompt",
                                             "classifier gradients",
                                             "guide every DDIM step",
                                             "outlier exposure"]},
        "shared": {"label": "both", "items": ["diffusion model",
                                              "an OOD score as the loss",
                                              "synthetic outliers",
                                              "no manual outlier data"]},
        "verdict": "same recipe at the top level, opposite choices underneath"}),
    "vs_pipelines": ("compare_pipelines", {
        "rows": [{"label": "SynOOD", "note": "ICCV 2025",
                  "stages": [["real ID image", "in"], ["MLLM names the context", "diff"],
                             ["in-paint at strength 0.6", "diff"],
                             ["energy score", "same"],
                             ["gradient on the noise", "diff"],
                             ["3 outer iterations", "diff"]]},
                 {"label": "GOOD", "note": "NeurIPS 2025",
                  "stages": [["class prompt", "in"], ["Stable Diffusion", "diff"],
                             ["energy + kNN", "diff"],
                             ["gradient in each step", "diff"],
                             ["25 step-size settings", "diff"],
                             ["pool the spectrum", "diff"]]}],
        "note": "both close a loop between a generator and an OOD scorer —"
                " but at different points in the sampling process",
        "verdict": "SynOOD edits an image from outside; GOOD steers the sampler from inside"}),
    "vs_where_samples_land": ("axis_positions", {
        "xlabel": "how far the synthetic samples sit from the ID distribution",
        "lo_label": "indistinguishable", "hi_label": "trivially separable",
        "bands": [{"label": "useless — no signal", "from": 0.0, "to": 0.16,
                   "colour": "neg"},
                  {"label": "informative", "from": 0.16, "to": 0.66,
                   "colour": "hot"},
                  {"label": "too easy — loose boundary", "from": 0.66, "to": 1.0,
                   "colour": "neg"}],
        "items": [{"label": "SynOOD", "at": 0.20, "to": 0.42, "colour": "in",
                   "note": "narrow by construction: editing can only move so far"},
                  {"label": "GOOD", "at": 0.14, "to": 0.86, "colour": "hot",
                   "note": "deliberately spans the range, then pools all 25 settings"}],
        "verdict": "one targets the boundary; the other samples a spectrum and lets training sort it out"}),
    "vs_what_gets_trained": ("compare_pipelines", {
        "rows": [{"label": "SynOOD", "note": "the detector is CLIP",
                  "stages": [["CLIP image encoder", "same"],
                             ["projection layer", "diff"],
                             ["CLIP text encoder", "same"],
                             ["5.5k of 11k neg labels", "diff"],
                             ["NegLabel score", "same"]]},
                 {"label": "GOOD", "note": "the detector is a classifier",
                  "stages": [["ResNet / ViT classifier", "diff"],
                             ["outlier-exposure head", "diff"],
                             ["free energy", "same"],
                             ["kNN feature distance", "diff"],
                             ["KL-weighted score", "diff"]]}],
        "note": "SynOOD adapts a vision-language detector; GOOD trains an ordinary classifier",
        "verdict": "different detectors, so the synthetic data is doing different jobs"}),
    "vs_settings": ("metric_table", {
        "columns": ["ID dataset", "detector", "baseline it builds on",
                    "FPR95", "AUROC"],
        "rows": [{"label": "SynOOD",
                  "values": ["ImageNet-1k", "CLIP ViT-B/16", "NegLabel",
                             "14.27", "97.01"]},
                 {"label": "GOOD",
                  "values": ["ImageNet-100", "ResNet / ViT", "Energy + OE",
                             "17.68", "96.30"]}],
        "verdict": "the two right-hand columns are NOT comparable — read the two left ones first"}),
    "vs_gains": ("compare_pipelines", {
        "rows": [{"label": "SynOOD", "note": "vs NegLabel, ImageNet-1k",
                  "stages": [["25.40 -> 14.27 FPR95", "in"],
                             ["gain is Places and Texture", "diff"],
                             ["iNaturalist, SUN barely move", "neg"],
                             ["near-OOD still 52.86", "neg"]]},
                 {"label": "GOOD", "note": "vs NCIS, ImageNet-100",
                  "stages": [["33.89 -> 17.68 FPR95", "in"],
                             ["mostly the unified score", "diff"],
                             ["Textures worse than NPOS", "neg"],
                             ["limited by the generator", "neg"]]}],
        "note": "each headline hides a different asterisk",
        "verdict": "both improve on their own baseline; neither result transfers to the other's setting"}),
    "vs_ablation_lesson": ("bars", {
        "title": "FPR95 attributable to each component, in each paper's own setting",
        "items": [["SynOOD: direct generation from negative labels", 22.59, "dim"],
                  ["SynOOD: + iterative gradient generation", 17.29, "hot"],
                  ["SynOOD: + label fine-tuning", 14.27, "hot"],
                  ["GOOD: both guidances, energy score only", 30.82, "dim"],
                  ["GOOD: + unified OOD score", 17.70, "hot"]],
        "fmt": "{:.2f}", "max": 34.0,
        "verdict": "SynOOD's gain is the data; GOOD's is largely the test-time score"}),

    # === LaCoT: Latent Chain-of-Thought (Sun et al., 2025) =================
    "lacot_latent": ("annotated_equation", {
        "latex": r"$P(Z|X,Y)=P(XZY)\ /\ \sum_{Z'}P(XZ'Y)$",
        "terms": [{"text": "the rationale is a latent variable", "at": 0.20,
                   "note": "not a fixed string to imitate", "colour": "hot"},
                  {"text": "intractable normaliser", "at": 0.72,
                   "note": "sum over every possible rationale", "colour": "neg"}],
        "verdict": "reasoning as posterior inference, not next-token prediction"}),
    "lacot_modes": ("mode_coverage", {
        "xlabel": "space of possible reasoning chains  Z",
        "target_label": r"target posterior  $P(Z|X,Y)$",
        "modes": [{"at": 0.18, "height": 0.75, "label": "algebraic route"},
                  {"at": 0.50, "height": 1.00, "label": "geometric route"},
                  {"at": 0.84, "height": 0.65, "label": "counting route"}],
        "collapsed": {"label": "PPO / GRPO: one mode", "at": 0.50, "n": 14},
        "covering": {"label": "AVI: proportional to reward", "n": 26},
        "verdict": "a KL penalty to the SFT policy is exactly what prevents finding the others"}),
    "lacot_why_not": ("compare_pipelines", {
        "rows": [{"label": "SFT", "note": "teacher forcing",
                  "stages": [["copy a reference chain", "neg"],
                             ["max log-likelihood", "same"],
                             ["parrots the trace", "neg"]]},
                 {"label": "PPO / GRPO", "note": "scalar reward",
                  "stages": [["sample the policy", "same"],
                             ["score with a critic", "neg"],
                             ["KL penalty to SFT", "neg"],
                             ["collapses to one mode", "neg"]]},
                 {"label": "LaCoT", "note": "amortized variational inference",
                  "stages": [["sample latent rationales", "in"],
                             ["reward = likelihood", "diff"],
                             ["reference-guided filter", "diff"],
                             ["matches the posterior", "in"]]}],
        "note": "no critic model anywhere in the third row — the reward is a likelihood",
        "verdict": "reward hacking is impossible when the reward is the model's own likelihood"}),
    "lacot_gflownet": ("trajectory_flow", {
        "states": [r"$z_1$", r"$z_2$", r"$z_i$", r"$z_j$", r"$z_n$", r"$\top$"],
        "flows": [r"$F(z_1)$", "", r"$F(z_i)$", r"$F(z_j)$", "", ""],
        "probs": [r"$q_\theta$", r"$q_\theta$", r"$q_\theta$", r"$q_\theta$",
                  r"$q_\theta$"],
        "balance": [2, 3], "balance_label": "sub-trajectory balance must hold here",
        "terminal_label": "terminal <eos>",
        "note": "in a causal LM each state has one parent, so the backward policy is 1",
        "verdict": "train the flow to be consistent and the policy samples in proportion to reward"}),
    "lacot_reward": ("annotated_equation", {
        "latex": r"$R(z_{1:t}\top)=\log P(X\,z_{1:t}\top\,Y)$",
        "terms": [{"text": "the reward is a likelihood", "at": 0.32,
                   "note": "no learned critic, nothing to hack", "colour": "hot"},
                  {"text": "the answer is in the conditioning", "at": 0.80,
                   "note": "how much this prefix helps reach Y", "colour": "in"}],
        "verdict": "a rationale is good exactly when it makes the right answer likely"}),
    "lacot_interp": ("sparse_interpolation", {
        "n": 33, "every": 8,
        "xlabel": "token position within the rationale (~1k tokens)",
        "ylabel": r"state reward  $R(z_{1:t}\top)$",
        "exact_label": r"computed every $\lambda=8$ steps",
        "interp_label": "linearly interpolated between",
        "saving_label": "8x fewer forward passes",
        "verdict": "local smoothness of the log-likelihood is what makes this safe"}),
    "lacot_reference": ("threshold_line", {
        "threshold": 0.62, "n": 24,
        "rule": r"keep only if $R(Z_i)>\delta_s R(Z_{ref})$",
        "lo_label": "discarded before the gradient",
        "hi_label": "better than reference — kept",
        "radius_label": r"$\delta_s$ anneals: explore freely for 50 steps, then tighten",
        "verdict": "filtering, not clipping — exploration stays free, variance still drops"}),
    "lacot_forgetting": ("curve_family", {
        "series": [{"name": "unconstrained", "colour": "neg",
                    "points": [[1, 0.30], [20, 0.52], [40, 0.48], [60, 0.30],
                               [80, 0.18], [100, 0.12]]},
                   {"name": "reference-guided", "colour": "hot",
                    "points": [[1, 0.30], [20, 0.55], [40, 0.66], [60, 0.72],
                               [80, 0.76], [100, 0.78]]}],
        "xlabel": "training step", "ylabel": "rationale reward",
        "hi_label": "better", "lo_label": "worse", "logx": False,
        "note": "unconstrained exploration drifts to high-likelihood, low-reward gibberish",
        "verdict": "catastrophic forgetting, fixed without a KL penalty"}),
    "lacot_bin": ("candidate_ranking", {
        "query_label": "image +\nquestion",
        "score_label": r"length-normalised $\pi_\Phi(Z_iY_i|X)\,/\,|Z_iY_i|$",
        "candidates": [{"label": r"$Z_1 Y_1$", "score": 0.31},
                       {"label": r"$Z_2 Y_2$", "score": 0.47, "best": True},
                       {"label": r"$Z_3 Y_3$", "score": 0.22},
                       {"label": r"$Z_4 Y_4$", "score": 0.38},
                       {"label": r"$Z_5 Y_5$", "score": 0.19}],
        "note": "Best-of-N needs a separate reward model here; BiN needs only the model itself",
        "verdict": "rank by marginal likelihood, treating the rationale as an integration variable"}),
    "lacot_roles": ("compare_pipelines", {
        "rows": [{"label": "pre-trained LVLM", "note": "no reasoning",
                  "stages": [["User: question", "same"],
                             ["Assistant: answer", "in"]]},
                 {"label": "fine-tuned reasoner", "note": "reasoning always",
                  "stages": [["User: question", "same"],
                             ["<think> ... </think>", "neg"],
                             ["<conclusion> answer", "in"]]},
                 {"label": "LaCoT", "note": "reasoning optional",
                  "stages": [["User: question", "same"],
                             ["Analyzer: Z   (optional)", "diff"],
                             ["Assistant: answer", "in"]]}],
        "note": "a new role token lets the model decide whether a rationale is needed at all",
        "verdict": "the rationale becomes a separable latent, not a mandatory prefix"}),
    "lacot_results": ("metric_table", {
        "columns": ["MathVista", "MathVerse-VO", "MMMU", "MMVet"],
        "rows": [{"label": "GPT-4o", "values": ["60.0", "40.6", "70.7", "69.1"]},
                 {"label": "Qwen2.5-VL-7B", "values": ["63.7", "37.8", "50.0", "70.5"]},
                 {"label": "R1-Onevision (GRPO)", "values": ["64.1", "43.3", "47.9", "71.1"]},
                 {"label": "LaCoT-Qwen-7B", "values": ["68.4", "48.8", "54.9", "74.2"],
                  "best": True}],
        "verdict": "best open-source LVLM here, within ~3 points of GPT-4o at 7B"}),
    "lacot_ablation": ("bars", {
        "title": "Qwen2.5-VL-7B on MathVista — training algorithm only",
        "items": [["GRPO", 62.6, "neg"], ["SFT", 62.7, "dim"],
                  ["zero-shot (no CoT)", 63.7, "dim"],
                  ["RGFN (ours)", 68.4, "hot"]],
        "fmt": "{:.1f}", "max": 72.0,
        "verdict": "both SFT and GRPO land at or below prompting the base model"}),
    "lacot_diversity": ("tradeoff_scatter", {
        "xlabel": "rationale diversity", "ylabel": "max log-likelihood of the rationale",
        "xlim": [0.41, 0.57], "ylim": [-0.52, -0.12],
        "corner_note": "better is up and to the right",
        "groups": [{"name": "SFT", "colour": "dim",
                    "points": [[0.437, -0.44, "3B"], [0.487, -0.30, "7B"],
                               [0.513, -0.33, "7B T=0.7"]]},
                   {"name": "LaCoT", "colour": "hot",
                    "points": [[0.463, -0.20, "3B", True],
                               [0.523, -0.27, "7B", True],
                               [0.545, -0.35, "7B T=0.7", True]]}],
        "verdict": "higher-likelihood rationales and more of them — the usual trade-off does not bind"}),
    "lacot_bin_scaling": ("curve_family", {
        "series": [{"name": "T=0.7", "colour": "hot",
                    "points": [[1, 33.7], [5, 37.2], [10, 37.8]]},
                   {"name": "T=0.5", "colour": "in",
                    "points": [[1, 32.7], [5, 36.0], [10, 36.0]]},
                   {"name": "greedy", "colour": "dim",
                    "points": [[1, 31.5], [5, 31.5], [10, 31.5]]}],
        "xlabel": "number of sampled rationales  N", "ylabel": "MathVerse-VO accuracy (%)",
        "hi_label": "better", "lo_label": "worse", "logx": False,
        "note": r"Monte-Carlo error of the estimator falls as $O(1/\sqrt{N})$",
        "verdict": "accuracy rises with N and with temperature — diversity is the resource"}),

    # === 10-799 Lectures 2 & 4: Diffusion and Score-based Models ===========
    "gen_taxonomy": ("hierarchy", {
        "root": "generative models",
        "levels": [["likelihood-based", "likelihood-free"],
                   ["autoregressive", "VAE", "normalizing flow", "EBM", "GAN"]],
        "verdict": "sampling from p(x) is hard; sampling from a Gaussian is easy"}),
    "gen_problems": ("strike_list", {
        "title": "why each family falls short",
        "items": ["autoregressive: one pixel at a time — 8.3M passes for a 4K image",
                  "VAE: blurry samples, because different x map to the same z region",
                  "GAN: unstable training and mode collapse",
                  "normalizing flow: architecture must be invertible",
                  "EBM: the partition function is intractable"],
        "kept_title": "what we want",
        "kept": ["sample all at once", "stable training", "no partition function"],
        "verdict": "every route to p(x) is blocked somewhere"}),
    "diff_chain": ("diffusion_chain", {
        "n": 5, "data_label": r"data  $x_0$", "noise_label": r"noise  $x_T$",
        "t_labels": ["t=0", "t=1", "t=2", "t=3", "t=4=T"],
        "forward_label": "forward process — add noise (fixed, no learning)",
        "reverse_label": "reverse process — denoise (this is what we learn)",
        "forward_eq": r"$q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)$",
        "reverse_eq": r"$p_\theta(x_{t-1}|x_t)=\mathcal{N}(\mu_\theta(x_t,t),\ \Sigma_\theta(x_t,t))$",
        "verdict": "turn noise into data, one small step at a time"}),
    "diff_forward": ("annotated_equation", {
        "latex": r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,"
                 r"\qquad \bar\alpha_t=\prod_{s=1}^{t}\alpha_s$",
        "terms": [{"text": "jump straight to any t", "at": 0.26,
                   "note": "no need to simulate the chain", "colour": "hot"},
                  {"text": "one Gaussian sample", "at": 0.66,
                   "note": r"$\epsilon\sim\mathcal{N}(0,I)$", "colour": "in"}],
        "verdict": "the forward process has a closed form — that is what makes training cheap"}),
    "diff_elbo": ("annotated_equation", {
        "latex": r"$\log p_\theta(x_0)\ \geq\ \mathbb{E}_q\!\left["
                 r"\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$",
        "terms": [{"text": "KL matching at every step", "at": 0.46,
                   "note": "the term that actually trains", "colour": "hot"},
                  {"text": "prior matching", "at": 0.70,
                   "note": "constant — both ends are fixed Gaussians",
                   "colour": "dim"},
                  {"text": "reconstruction", "at": 0.92,
                   "note": "the last denoising step", "colour": "in"}],
        "verdict": "same Jensen's-inequality move as the VAE, applied T times"}),
    "diff_simplify": ("swap", {
        "context": "the full variational objective",
        "before": "match the posterior mean at every step",
        "after": r"just predict the noise:  $\|\epsilon-\epsilon_\theta(x_t,t)\|^2$",
        "before_colour": "dim", "after_colour": "hot",
        "note": "fix the forward process, fix the reverse variance, reparameterise the mean",
        "verdict": "three simplifications turn a page of KL terms into one L2 loss"}),
    "ddpm_algorithms": ("compare_pipelines", {
        "rows": [{"label": "Training", "note": "one step",
                  "stages": [[r"sample $x_0$", "in"], [r"sample $t$", "same"],
                             [r"sample $\epsilon$", "same"],
                             [r"form $x_t$ directly", "diff"],
                             [r"step on $\|\epsilon-\epsilon_\theta\|^2$", "diff"]]},
                 {"label": "Sampling", "note": "T steps",
                  "stages": [[r"$x_T\sim\mathcal{N}(0,I)$", "out"],
                             [r"predict $\epsilon_\theta$", "diff"],
                             ["subtract it", "diff"], ["add fresh noise", "same"],
                             [r"repeat to $t=0$", "in"]]}],
        "note": "no chain simulation during training — that is the whole trick",
        "verdict": "training touches one timestep at a time; only sampling is sequential"}),
    "diff_is_vae": ("compare_pipelines", {
        "rows": [{"label": "VAE", "note": "",
                  "stages": [["x", "in"], ["encoder (learned)", "neg"],
                             ["z", "out"], ["decoder (learned)", "diff"],
                             ["x", "in"]]},
                 {"label": "Diffusion", "note": "",
                  "stages": [["x", "in"], ["noising (fixed)", "same"],
                             [r"$x_T$", "out"], ["denoiser (learned)", "diff"],
                             ["x", "in"]]}],
        "note": "the encoder is not learned, the latent is the same size as the data,"
                " and there are T of them",
        "verdict": "a diffusion model is a VAE with a hand-designed hierarchical encoder"}),
    "unet": ("unet_shape", {
        "depth": 4, "in_label": r"$x_t$", "out_label": r"$\epsilon_\theta$",
        "bottleneck_label": "bottleneck", "skip_label": "skip connections",
        "note": "coarse features by downsampling, fine features by upsampling,"
                " and the output has the same shape as the input",
        "verdict": "U-Net dominated diffusion architectures for years, and this is why"}),
    "score_idea": ("annotated_equation", {
        "latex": r"$s_\theta(x)\ \approx\ \nabla_x \log p_{\mathrm{data}}(x)$",
        "terms": [{"text": "the score", "at": 0.62,
                   "note": "gradient of log-density w.r.t. the input", "colour": "hot"},
                  {"text": "a network we can just train", "at": 0.20,
                   "note": "outputs a vector, not a probability", "colour": "in"}],
        "verdict": "no partition function: the gradient of log Z is zero"}),
    "score_field": ("score_field", {
        "field_label": r"$\nabla_x \log p(x)$  — always points uphill",
        "note": "knowing the direction of increasing density is enough to sample;"
                " we never need the normalising constant",
        "verdict": "if you only want to sample, you never needed the density itself"}),
    "score_matching": ("annotated_equation", {
        "latex": r"$L(\theta)=\mathbb{E}_{p_{\mathrm{data}}}\!\left[\tfrac{1}{2}"
                 r"\|s_\theta(x)\|^2+\mathrm{tr}(J_{s_\theta}(x))\right]+C$",
        "terms": [{"text": "no ground-truth score anywhere", "at": 0.34,
                   "note": "integration by parts removed it", "colour": "hot"},
                  {"text": "trace of a Jacobian", "at": 0.76,
                   "note": "one backward pass per dimension — far too slow",
                   "colour": "neg"}],
        "verdict": "the trick works, and then the cost of the trace kills it"}),
    "denoising_sm": ("annotated_equation", {
        "latex": r"$\nabla_{\tilde x}\log q_\sigma(\tilde x|x)"
                 r"=\tfrac{1}{\sigma^2}(x-\tilde x)$",
        "terms": [{"text": "noise the data yourself", "at": 0.40,
                   "note": "then the score is known in closed form", "colour": "hot"},
                  {"text": "it points back at the clean sample", "at": 0.82,
                   "note": "so score matching becomes denoising", "colour": "in"}],
        "verdict": "no Jacobian, no ground-truth score — and it is the DDPM loss in disguise"}),
    "langevin": ("score_field", {
        "walk": True, "walk_steps": 44, "start_x": -1.05, "start_y": -0.7,
        "field_label": r"$\nabla_x\log p(x)$",
        "walk_label": r"$x_{t+1}=x_t+\eta\,s_\theta(x_t)+\sqrt{2\eta}\,z_t$",
        "note": "gradient ascent in data space, plus noise so it explores"
                " rather than collapsing onto a mode",
        "verdict": "Langevin dynamics: follow the score, add noise, repeat"}),
    "score_pitfalls": ("score_field", {
        "bad_region": [-1.05, -0.62, 0.42],
        "bad_label": "score estimate is unreliable here",
        "field_label": r"$s_\theta(x)$ learned from data",
        "note": "and this is exactly where Langevin has to start",
        "verdict": "three pitfalls: low-density regions, the manifold hypothesis,"
                   " and high vs higher density"}),
    "noise_levels": ("noise_scales", {
        "sigmas": [0.02, 0.10, 0.32],
        "labels": [r"$\sigma$ too small", r"$\sigma$ just right", r"$\sigma$ too large"],
        "verdicts": ["low-density gaps remain", "fills the gaps, keeps the modes",
                     "the data distribution is destroyed"],
        "xlabel": "data space",
        "verdict": "so do not pick one — use many, and anneal from large to small"}),
    "ncsn": ("annotated_equation", {
        "latex": r"$\mathbb{E}_{\sigma_t}\mathbb{E}_{x}\mathbb{E}_{\tilde x}"
                 r"\left[\left\|s_\theta(\tilde x_{\sigma_t},\sigma_t)"
                 r"+\tfrac{x-\tilde x_{\sigma_t}}{\sigma_t^2}\right\|^2\right]$",
        "terms": [{"text": "one network, conditioned on the noise level", "at": 0.44,
                   "note": "not one network per sigma", "colour": "hot"},
                  {"text": "the closed-form score of the noise", "at": 0.80,
                   "note": "denoising score matching, at every level", "colour": "in"}],
        "verdict": "NCSN: multi-level denoising score matching, sampled by annealed Langevin"}),
    "ddpm_vs_ncsn": ("compare_pipelines", {
        "rows": [{"label": "DDPM", "note": "variance preserving",
                  "stages": [[r"$\lambda_t=\sqrt{\alpha_t}$", "diff"],
                             [r"$\sigma_t=\sqrt{1-\alpha_t}$", "diff"],
                             [r"predict $\epsilon_\theta$", "in"],
                             ["ancestral sampling", "same"]]},
                 {"label": "NCSN", "note": "variance exploding",
                  "stages": [[r"$\lambda_t=1$", "diff"], [r"$\sigma_t=\sigma_t$", "diff"],
                             [r"predict $s_\theta$", "in"],
                             ["annealed Langevin", "same"]]}],
        "note": r"both are $x_t=\lambda_t x_0+\sigma_t\epsilon$ — only the schedule differs,"
                r" and $s_\theta=-\epsilon_\theta/\sigma_t$",
        "verdict": "two sides of the same coin, arrived at from opposite directions"}),
    "sde_limit": ("annotated_equation", {
        "latex": r"$dx=f(x,t)\,dt+g(t)\,dw$",
        "terms": [{"text": "drift", "at": 0.34,
                   "note": r"DDPM: $-\tfrac{1}{2}\beta(t)x$   ·   NCSN: $0$",
                   "colour": "in"},
                  {"text": "diffusion", "at": 0.72,
                   "note": r"DDPM: $\sqrt{\beta(t)}$   ·   NCSN: $\sqrt{d\sigma^2/dt}$",
                   "colour": "hot"}],
        "verdict": "take the number of noise levels to infinity and both become one SDE"}),
    "sde_reverse": ("annotated_equation", {
        "latex": r"$dx=\left[f(x,t)-g(t)^2\nabla_x\log p_t(x)\right]dt"
                 r"+g(t)\,d\bar w$",
        "terms": [{"text": "the reverse-time SDE", "at": 0.22,
                   "note": "Anderson, 1982", "colour": "in"},
                  {"text": "the only unknown is the score", "at": 0.62,
                   "note": "train it by score matching, then use any ODE/SDE solver",
                   "colour": "hot"}],
        "verdict": "every sampler — DDPM, DDIM, Euler, Heun — is a solver for this equation"}),

    # === Respecting Modality Gap (Hu et al., 2026) ========================
    "open_world": ("stream_router", {
        "box_title": "CLIP", "box_sub1": "frozen", "box_sub2": "no retraining",
        "bin_top": "accept as ID", "bin_bottom": "flag as OOD"}),
    "shared_space": ("two_clusters", {
        "space_label": "CLIP's shared unit sphere", "a_label": "text embeddings",
        "b_label": "image embeddings", "gap_label": "modality\ngap"}),
    "subspace_gap": ("vector_decompose", {
        "plane_label": r"$\mathcal{S}$", "plane_note": "span of the image features",
        "in_label": r"$\mathbf{w}^*$", "out_label": r"$\mathbf{r}_y$",
        "out_note": "component\noutside " + r"$\mathcal{S}$",
        "proj_label": r"$\mathrm{Proj}_{\mathcal{S}}(\mathbf{r}_y)$",
        "gap_label": "the gap"}),
    "prototype_walk": ("cluster_walk", {
        "space_label": "image feature space",
        "route_label": r"route by $S_{\mathrm{NegLabel}}$",
        "skip_label": "uncertain — skip",
        "legend_a": "text-embedding start (Eq. 15)",
        "legend_b": "learned visual prototype",
        "counter_label": "test samples seen"}),
    "regret_curve": ("growth_curve", {
        "xlabel": "test samples seen  " + r"$N$", "ylabel": "cumulative regret",
        "ref_label": "linear — no learning",
        "curve_label": r"$\mathcal{O}(\sqrt{N})$",
        "note": "average loss per sample " + r"$\to 0$"}),
    "similarity_bars": ("bars", {
        "title": r"image $\mathbf{z}$  vs  text $\mathbf{r}_y$",
        "verdict": r"$S_{\mathrm{MCM}}=\max_y\ \mathrm{softmax}$"}),
    "mass_split": ("stacked_bar", {
        "total_label": "all probability mass",
        "a_label": "{pct}%  on the K ID names",
        "b_label": "the rest leaks onto 10,000 negative names",
        "verdict": "ID mass / total mass  =  the score"}),
    "tug_of_war": ("attractor", {
        "marker_label": r"$\mathbf{w}^*$",
        "pull_label": "pulled in by its own images  " + r"$(1-\pi)$",
        "push_label": "pushed away by everyone else's  " + r"$\pi$"}),
    "error_bar": ("split_bar", {
        "title": r"$\|\mathbf{r}_y-\mathbf{w}^*\|_2^2$",
        "a_label": "alignable — the in-span part",
        "b_label": "irreducible — the orthogonal part",
        "verdict": "the bound vanishes only if this is zero"}),
    "soft_target": ("distribution_pair", {
        "top_title": "hard label — unavailable at test time",
        "bottom_title": r"CLIP's own prediction $P(\hat{y}|\mathbf{x})$"}),
    "sphere_step": ("manifold_step", {
        "set_label": r"$\mathbb{S}^{d-1}$", "start_label": r"$\mathbf{w}^{(i-1)}$",
        "step_label": r"$-\eta_i\nabla$", "end_label": r"$\mathbf{w}^{(i)}$",
        "verdict": r"$\ell_2$ renormalise = projection onto the sphere"}),
    "score_swap": ("swap", {
        "context": "NegLabel", "before": r"$\mathbf{r}_y$  —  text embedding",
        "after": r"$\mathbf{w}_y^{(i)}$  —  learned visual",
        "note": "everything else is identical"}),

    # === Inference-Optimal VLMs (Li et al., 2024) =========================
    "token_stack": ("token_compare", {
        "source": "image", "encoder": "CLIP ViT-L", "encoder_note": "fixed cost",
        "grid": 24, "grid_label": "576 visual tokens", "few": 10,
        "few_label": "~50 text tokens",
        "verdict": "the image is over 90% of what the LLM processes"}),
    "flops_budget": ("area_compare", {
        "title": r"$\mathrm{FLOPs}\ =\ \mathcal{O}(N\times T)$",
        "verdict": "same compute — which one is better?"}),
    "scaling_curves": ("curve_family", {
        "series": [
            {"name": "0.5B", "points": [[0.5, 0.6197, 1], [2, 0.6090, 4], [8, 0.5986, 16], [18, 0.5925, 36], [32, 0.5883, 64], [72, 0.5824, 144], [288, 0.5725, 576]]},
            {"name": "1.8B", "points": [[1.8, 0.5709, 1], [7.2, 0.5612, 4], [28.8, 0.5517, 16], [64.8, 0.5463, 36], [115.2, 0.5424, 64], [259.2, 0.5371, 144], [1036.8, 0.5281, 576]]},
            {"name": "4B", "points": [[4, 0.5428, 1], [16, 0.5337, 4], [64, 0.5248, 16], [144, 0.5197, 36], [256, 0.5161, 64], [576, 0.5110, 144], [2304, 0.5026, 576]]},
            {"name": "7B", "points": [[7, 0.5242, 1], [28, 0.5154, 4], [112, 0.5069, 16], [252, 0.5020, 36], [448, 0.4985, 64], [1008, 0.4937, 144], [4032, 0.4856, 576]]}],
        "xlabel": r"inference FLOPs  $\mathcal{O}(N(Q{+}V))$",
        "ylabel": "downstream error", "hi_label": "worse", "lo_label": "better",
        "frontier": "min", "frontier_label": "Pareto optimal",
        "note": "biggest LLM, fewest tokens"}),
    "q_shift": ("ratio_slider", {
        "a_label": "text tokens  " + r"$Q$",
        "b_label": "optimal visual tokens  " + r"$V^*$",
        "a_short": "Q", "b_short": "V*",
        "steps": [[0, 1], [10, 4], [25, 9], [50, 16], [100, 36]],
        "notes": ["the text is a fixed cost the LLM already pays",
                  "so a few more visual tokens are nearly free"]}),
    "ocr_flip": ("panels", {"verdict": "the recipe flips with the task"}),
    "bench_delta": ("diverging_bars", {
        "caption": "7B LLM w/ 36 tokens  vs  0.5B w/ 576",
        "group_a": "visual reasoning — better",
        "group_b": "text recognition — worse"}),
    "grid_compress": ("grid_collapse", {
        "grid_label": r"$\sqrt{n}\times\sqrt{n}$ patches",
        "region_label": r"depth-wise conv, kernel = stride = $s$",
        "out_label": "9 tokens",
        "note": "cross-attention runs inside each region only"}),
    "quecc_flow": ("pipeline", {
        "verdict": "the query decides which patches survive",
        "note": "and it can be cached when the prompt is fixed"}),
}


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

PRIMITIVES = {
    "annotated_equation":    annotated_equation,
    "area_compare":          area_compare,
    "attractor":             attractor,
    "bars":                  bars,
    "before_after":          before_after,
    "cluster_walk":          cluster_walk,
    "curve_family":          curve_family,
    "dataflow":              dataflow,
    "density_pair":          density_pair,
    "distribution_pair":     distribution_pair,
    "diverging_bars":        diverging_bars,
    "flow_split":            flow_split,
    "grid_collapse":         grid_collapse,
    "growth_curve":          growth_curve,
    "hierarchy":             hierarchy,
    "manifold_step":         manifold_step,
    "matrix_grid":           matrix_grid,
    "panels":                panels,
    "parallel_heads":        parallel_heads,
    "pipeline":              pipeline,
    "ratio_slider":          ratio_slider,
    "scale_ladder":          scale_ladder,
    "scatter_boundary":      scatter_boundary,
    "sequence_tokens":       sequence_tokens,
    "split_bar":             split_bar,
    "stacked_bar":           stacked_bar,
    "stacked_blocks":        stacked_blocks,
    "stream_router":         stream_router,
    "swap":                  swap,
    "threshold_line":        threshold_line,
    "timeline":              timeline,
    "token_compare":         token_compare,
    "two_clusters":          two_clusters,
    "vector_decompose":      vector_decompose,
    "feedback_loop":         feedback_loop,
    "boundary_band":         boundary_band,
    "dual_space_align":      dual_space_align,
    "module_states":         module_states,
    "pair_strip":            pair_strip,
    "ablation_grid":         ablation_grid,
    "peak_curve":            peak_curve,
    "tradeoff_scatter":      tradeoff_scatter,
    "hypersphere_shift":     hypersphere_shift,
    "metric_table":          metric_table,
    "embedding_grid":        embedding_grid,
    "trajectory_guidance":   trajectory_guidance,
    "energy_walk":           energy_walk,
    "knn_sparsity":          knn_sparsity,
    "param_grid":            param_grid,
    "sliced_projection":     sliced_projection,
    "distribution_shapes":   distribution_shapes,
    "siamese_predict":       siamese_predict,
    "strike_list":           strike_list,
    "boundary_spread":       boundary_spread,
    "step_schedule":         step_schedule,
    "noise_ladder":          noise_ladder,
    "milestone_chain":       milestone_chain,
    "quadrant_map":          quadrant_map,
    "bottleneck_codec":      bottleneck_codec,
    "factorized_attention":  factorized_attention,
    "modality_attention":    modality_attention,
    "interpolation_path":    interpolation_path,
    "attention_lookup":      attention_lookup,
    "head_patterns":         head_patterns,
    "sequential_vs_parallel": sequential_vs_parallel,
    "score_sharpening":      score_sharpening,
    "saturation_curve":      saturation_curve,
    "residual_stack":        residual_stack,
    "matmul_chain":          matmul_chain,
    "design_matrix":         design_matrix,
    "object_stack":          object_stack,
    "reward_profile":        reward_profile,
    "group_advantage":       group_advantage,
    "value_matrix":          value_matrix,
    "probe_layers":          probe_layers,
    "physics_principles":    physics_principles,
    "shortcut_failure":      shortcut_failure,
    "task_gallery":          task_gallery,
    "frame_sequence":        frame_sequence,
    "compare_pipelines":     compare_pipelines,
    "venn2":                 venn2,
    "axis_positions":        axis_positions,
    "mode_coverage":         mode_coverage,
    "trajectory_flow":       trajectory_flow,
    "sparse_interpolation":  sparse_interpolation,
    "candidate_ranking":     candidate_ranking,
    "diffusion_chain":       diffusion_chain,
    "score_field":           score_field,
    "unet_shape":            unet_shape,
    "noise_scales":          noise_scales,
}


def _bind(fn, preset):
    """A preset is a primitive with opts pre-filled. The storyboard's own opts
    still win, so a preset is a starting point rather than a lock."""
    def draw(R, ax, u, d, o):
        return fn(R, ax, u, d, {**preset, **(o or {})})
    draw.__doc__ = fn.__doc__
    draw._preset_of = getattr(fn, "__name__", "?")
    return draw


def _resolve():
    """Merge the long-tail module, then bind every preset to its primitive."""
    import sys as _sys
    partial = ("illustrations_extra" in _sys.modules
               and not hasattr(_sys.modules["illustrations_extra"], "PRIMITIVES"))
    if not partial:                        # skip while it is mid-import
        try:
            import illustrations_extra as _x
            PRIMITIVES.update(getattr(_x, "PRIMITIVES", {}))
            PRESETS.update(getattr(_x, "PRESETS", {}))
        except Exception as e:             # the long tail is optional
            print(f"[illustrations] illustrations_extra unavailable: {e}")
    figs = dict(PRIMITIVES)
    for name, (prim, opts) in PRESETS.items():
        if prim in PRIMITIVES:
            figs[name] = _bind(PRIMITIVES[prim], opts)
        elif not partial:
            print(f"[illustrations] preset {name!r} wants unknown "
                  f"primitive {prim!r}")
    return figs


FIGURES = _resolve()


def catalogue() -> str:
    out = ["PRIMITIVES  — generic; drive these from the storyboard's opts"]
    for n in sorted(PRIMITIVES):
        doc = (PRIMITIVES[n].__doc__ or "").strip().split("\n")[0]
        out.append(f"  {n:<22} {doc[:72]}")
    out.append("")
    out.append("PRESETS  — a primitive with one paper's values filled in")
    for n in sorted(PRESETS):
        out.append(f"  {n:<22} -> {PRESETS[n][0]}")
    out.append("")
    out.append(f"{len(PRIMITIVES)} primitives, {len(PRESETS)} presets, "
               f"{len(FIGURES)} names")
    return "\n".join(out)