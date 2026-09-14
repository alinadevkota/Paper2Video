"""
illustrations_extra.py — the long tail of the figure library.

Same contract as illustrations.py; these are simply the shapes that come up
less often. Merged into the registry automatically when this file is present,
and safe to delete if you never need them.

Put a new primitive here when it feels specialised, and move it into
illustrations.py once you find yourself reaching for it repeatedly.
"""

import numpy as np
from matplotlib.patches import (Circle, FancyArrowPatch, FancyBboxPatch,
                                Polygon, Rectangle, Wedge)   # noqa: F401

from illustrations import (W, H, ALPHA, BETA, K, D0, NS, VS, _arrow, _box,
                           _col, _ease, _err, _fade, _fl, _pal, _proj3,
                           _tail, _tex)                      # noqa: F401


def arrow_cycle(R, ax, u, d, o):
    """A closed loop of stages — training loops, EM, any repeat-until process.

    opts: stages [label | [label, colour]], centre_label, verdict, spin
    """
    C = R.c
    x, y, w, h = _box(o)
    cx, cy = x + w / 2, y + h / 2 + 0.15
    rad = min(w * 0.26, h * 0.36)
    stages = o.get("stages", [])
    n = max(len(stages), 1)

    for k, st in enumerate(stages):
        lab = st[0] if isinstance(st, (list, tuple)) else st
        col = _col(R, st[1] if isinstance(st, (list, tuple)) and len(st) > 1
                   else "hot")
        a = _fade(u, R.rt(k, 0.5 + k * 0.8, 0.6), 0.5)
        if a <= 0.01:
            continue
        th = np.pi / 2 - 2 * np.pi * k / n
        px, py = cx + rad * np.cos(th), cy + rad * np.sin(th)
        R.box(ax, px - 1.15, py - 0.42, 2.30, 0.84, a=a, fc=col, r=0.12, z=7)
        R.txt(ax, px, py, str(lab), 15, "#FFFFFF", a, ha="center", z=9)
        th2 = np.pi / 2 - 2 * np.pi * (k + 1) / n
        qx, qy = cx + rad * np.cos(th2), cy + rad * np.sin(th2)
        mid = ((px + qx) / 2 + (cx - (px + qx) / 2) * -0.18,
               (py + qy) / 2 + (cy - (py + qy) / 2) * -0.18)
        _arrow(ax, (px + (qx - px) * 0.28, py + (qy - py) * 0.28),
               (px + (qx - px) * 0.74, py + (qy - py) * 0.74),
               C["accent"], 2.8, a, 18)

    if o.get("centre_label"):
        R.txt(ax, cx, cy, o["centre_label"], 20, C["muted"],
              _fade(u, R.rt(0, 0.4, 0.6), 0.6), ha="center")
    _tail(R, ax, o, u, n, x, y, w)


def budget_ladder(R, ax, u, d, o):
    """Hold the compute budget fixed; walk the candidates; the error drops."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    combos = o.get("combos", [["0.5B", 576], ["1.8B", 144], ["4B", 64],
                              ["7B", 36], ["7B", 1]])
    errs = [_err(float(n[:-1]), v) for n, v in combos]

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    R.txt(ax, x + w / 2, y + h - 0.2, o.get("title", ""), 20, C["ink"],
          a0, ha="center", weight="bold")

    step = max((d - 2.0) / max(len(combos), 1), 0.8)
    lo, hi = min(errs) - 0.02, max(errs) + 0.02
    for k, ((nl, v), e) in enumerate(zip(combos, errs)):
        t0 = R.rt(min(k + 1, 3), 1.2 + k * step, step)
        a = _fade(u, t0, 0.5)
        if a <= 0.01:
            continue
        yy = y + h - 1.1 - k * ((h - 2.2) / max(len(combos) - 1, 1))
        last = k == len(combos) - 1
        col = C["accent"] if last else C["dim"]
        bwid = (w - 5.2) * (hi - e) / (hi - lo)
        g = _ease((u - t0) / 0.7)
        ax.add_patch(Rectangle((x + 3.6, yy - 0.22), max(bwid, 0.1) * g, 0.44,
                               fc=col, ec="none", alpha=min(a, 1.0), zorder=6))
        R.txt(ax, x + 3.35, yy, f"{nl}  ·  {v} tok", 17,
              C["ink"] if last else C["muted"], a, ha="right",
              weight="bold" if last else "normal")
        R.txt(ax, x + 3.6 + max(bwid, 0.1) * g + 0.2, yy, f"{e:.3f}", 17,
              C["ink"] if last else C["muted"], a * g,
              weight="bold" if last else "normal")

    a3 = _fade(u, R.rt(3, 1.2 + len(combos) * step, 0.9), 0.7)
    R.txt(ax, x + w / 2, y + 0.28,
          o.get("verdict", ""), 20, C["accent"], a3,
          ha="center", weight="bold")


def connectivity_compare(R, ax, u, d, o):
    """How information reaches one position under different layer types, drawn
    as connection patterns over a row of positions, with the path length
    called out. The argument every architecture paper makes.

    opts: variants [{label, kind: chain|all|local|dilated, note, k}],
          n_positions, target, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    vs = o.get("variants", [])
    n = int(o.get("n_positions", 8))
    tgt = int(o.get("target", n - 1))
    rows = max(len(vs), 1)
    lane = (h - 1.2) / rows
    px0 = x + 3.4
    step = (w - 4.4) / max(n - 1, 1)

    for vi, v in enumerate(vs):
        a = _fade(u, R.rt(vi, 0.5 + vi * 1.3, 0.7), 0.6)
        if a <= 0.01:
            continue
        yy = y + h - 0.8 - vi * lane
        kind = v.get("kind", "chain")
        kk = int(v.get("k", 2))
        for i in range(n):
            hot = i == tgt
            ax.plot([px0 + i * step], [yy], "o", ms=13,
                    color=C["accent"] if hot else C["dim"],
                    alpha=min(a, 1.0), zorder=7)
        if kind == "chain":
            for i in range(n - 1):
                ax.plot([px0 + i * step, px0 + (i + 1) * step], [yy, yy], lw=2.2,
                        color=C["muted"], alpha=min(a, 1.0) * 0.8, zorder=5)
        elif kind == "all":
            for i in range(n):
                if i == tgt:
                    continue
                mid = (px0 + i * step + px0 + tgt * step) / 2
                arc = abs(i - tgt) * 0.055 + 0.16
                t = np.linspace(0, 1, 30)
                bx_ = (1 - t) * (px0 + i * step) + t * (px0 + tgt * step)
                by_ = yy + np.sin(np.pi * t) * arc
                ax.plot(bx_, by_, lw=1.8, color=C["accent"],
                        alpha=min(a, 1.0) * 0.55, zorder=5)
        elif kind in ("local", "dilated"):
            gap = kk if kind == "local" else kk * 2
            for i in range(n):
                for j in (i - gap, i + gap):
                    if 0 <= j < n and j > i:
                        t = np.linspace(0, 1, 20)
                        ax.plot((1 - t) * (px0 + i * step) + t * (px0 + j * step),
                                yy + np.sin(np.pi * t) * 0.22, lw=1.8,
                                color=C["muted"], alpha=min(a, 1.0) * 0.7,
                                zorder=5)
        R.txt(ax, x + 3.15, yy, str(v.get("label", "")), 18, C["ink"], a,
              ha="right", weight="bold")
        if v.get("note"):
            R.txt(ax, x + 3.15, yy - 0.42, v["note"], 15, C["muted"], a,
                  ha="right")
    _tail(R, ax, o, u, rows, x, y, w)


def exponent_bars(R, ax, u, d, o):
    """alpha vs beta: how fast error moves with parameters versus with tokens."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    al = float(o.get("alpha", ALPHA))
    be = float(o.get("beta", BETA))
    mx = max(al, be) * 1.18
    bw = w - 3.4

    for i, (lab, val, col, beat) in enumerate(
            [(o.get("a_label", r"$\alpha$"), al, C["accent"], 0),
             (o.get("b_label", r"$\beta$"), be, P["vis"], 1)]):
        a = _fade(u, R.rt(beat, 0.5 + i * 1.3, 0.7), 0.6)
        if a <= 0.01:
            continue
        yy = y + h * (0.68 - i * 0.30)
        g = _ease((u - R.rt(beat, 0.5 + i * 1.3, 0.7)) / 0.9)
        ax.add_patch(Rectangle((x + 3.0, yy - 0.32), bw * val / mx * g, 0.64,
                               fc=col, ec="none", alpha=min(a, 1.0), zorder=6))
        R.txt(ax, x + 2.75, yy, lab, 19, C["ink"], a, ha="right")
        R.txt(ax, x + 3.0 + bw * val / mx * g + 0.22, yy, f"{val:.3f}", 20,
              col, a * g, weight="bold")

    a2 = _fade(u, R.rt(2, 3.2, 0.8), 0.7)
    ratio = al / be if be else 0
    R.txt(ax, x + w / 2, y + h * 0.18,
          o.get("verdict", f"error moves {ratio:.0f}x faster with parameters"),
          22, C["accent"], a2, ha="center", weight="bold")


def graph_nodes(R, ax, u, d, o):
    """A small graph whose edges light up in waves — message passing, attention
    between entities, information flow.

    opts: nodes [{label, x, y, colour}], edges [[i, j]], waves, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    nodes = o.get("nodes", [])
    edges = o.get("edges", [])
    if not nodes:
        return
    cx, cy = x + w / 2, y + h / 2 + 0.2
    sc = min(w * 0.16, h * 0.30)

    def P(nd, k):
        if "x" in nd:
            return cx + float(nd["x"]) * sc, cy + float(nd["y"]) * sc
        th = np.pi / 2 - 2 * np.pi * k / len(nodes)
        return cx + sc * 1.5 * np.cos(th), cy + sc * 1.5 * np.sin(th)

    pos = [P(nd, k) for k, nd in enumerate(nodes)]
    t1 = R.rt(1, 1.4, 0.8)
    waves = int(o.get("waves", 3))
    for k, (i, j) in enumerate(edges):
        step = k % max(waves, 1)
        a = _fade(u, t1 + step * 0.9, 0.5)
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], lw=2.4,
                color=C["accent"], alpha=min(a, 1.0) * 0.75, zorder=5)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for k, nd in enumerate(nodes):
        px, py = pos[k]
        ax.add_patch(Circle((px, py), 0.46, fc=_col(R, nd.get("colour", "in")),
                            ec="none", alpha=min(a0, 1.0), zorder=7))
        R.txt(ax, px, py, str(nd.get("label", "")), 14, "#FFFFFF", a0,
              ha="center", z=9)
    _tail(R, ax, o, u, 3, x, y, w)


def nested_boxes(R, ax, u, d, o):
    """Containment — subsets, capability hierarchies, scope of a claim.

    opts: layers [{label, colour}] outermost first, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    layers = o.get("layers", [])
    n = max(len(layers), 1)
    cx, cy = x + w / 2, y + h / 2 + 0.2
    for k, L in enumerate(layers):
        a = _fade(u, R.rt(k, 0.5 + k * 1.0, 0.6), 0.6)
        if a <= 0.01:
            continue
        f = 1.0 - k / (n + 0.6)
        bw, bh = (w - 2.0) * f, (h - 1.8) * f
        R.box(ax, cx - bw / 2, cy - bh / 2, bw, bh, a=a * 0.55,
              fc=_col(R, L.get("colour", "in")), r=0.16, z=4 + k)
        R.txt(ax, cx, cy + bh / 2 - 0.42, str(L.get("label", "")), 18,
              C["ink"], a, ha="center", z=9)
    _tail(R, ax, o, u, n, x, y, w)


def queue_buffer(R, ax, u, d, o):
    """Items entering a fixed-size buffer and older ones being evicted —
    caches, replay buffers, sliding windows, KV eviction.

    opts: slots, arrivals, kept_label, evicted_label, rate, verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    slots = int(o.get("slots", 6))
    cy = y + h * 0.58
    sw = min((w - 2.0) / slots, 1.9)
    bx = x + (w - sw * slots) / 2

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for i in range(slots):
        R.box(ax, bx + i * sw, cy - 0.55, sw * 0.9, 1.1, a=a0 * 0.4,
              fc=C["panel"], r=0.10, z=4)

    t1 = R.rt(1, 1.6, 0.8)
    rate = float(o.get("rate", 1.1))
    arrived = int(max(0.0, u - t1) / max(rate, 0.05))
    for i in range(min(arrived, slots + 3)):
        pos = arrived - 1 - i
        if pos >= slots:
            a = max(0.0, 1.0 - (pos - slots + 1) * 0.5)
            col = C["neg"]
        else:
            a, col = 1.0, _col(R, "in")
        px = bx + max(pos, 0) * sw + (0 if pos < slots else (pos - slots) * 0.4)
        if a <= 0.02:
            continue
        R.box(ax, px, cy - 0.5, sw * 0.9, 1.0, a=a, fc=col, r=0.10, z=6)
        R.txt(ax, px + sw * 0.45, cy, str(arrived - i), 15, "#FFFFFF", a,
              ha="center", z=9)

    a2 = _fade(u, R.rt(2, 3.0, 0.8), 0.6)
    R.txt(ax, bx, cy + 1.0, o.get("kept_label", ""), 18, _col(R, "in"), a2)
    R.txt(ax, bx + slots * sw + 0.2, cy - 1.15, o.get("evicted_label", ""), 17,
          C["neg"], a2)
    _tail(R, ax, o, u, 3, x, y, w)


def radial_compare(R, ax, u, d, o):
    """A radar chart over several axes — multi-benchmark comparisons where a
    bar chart would need too many rows.

    opts: axes [str], series [{name, values (0..1), colour}], verdict
    """
    C = R.c
    x, y, w, h = _box(o)
    cx, cy = x + w / 2, y + h / 2 + 0.25
    rad = min(w * 0.20, h * 0.36)
    axes_ = o.get("axes", [])
    n = max(len(axes_), 3)

    a0 = _fade(u, R.rt(0, 0.4, 0.6), 0.6)
    for ring in (0.33, 0.66, 1.0):
        th = np.linspace(0, 2 * np.pi, 80)
        ax.plot(cx + rad * ring * np.cos(th), cy + rad * ring * np.sin(th),
                lw=1.2, color=C["line"], alpha=min(a0, 1.0) * 0.8, zorder=4)
    for k, lab in enumerate(axes_):
        th = np.pi / 2 - 2 * np.pi * k / n
        ax.plot([cx, cx + rad * np.cos(th)], [cy, cy + rad * np.sin(th)],
                lw=1.2, color=C["line"], alpha=min(a0, 1.0) * 0.8, zorder=4)
        R.txt(ax, cx + rad * 1.20 * np.cos(th), cy + rad * 1.20 * np.sin(th),
              str(lab), 15, C["muted"], a0, ha="center")

    for si, s in enumerate(o.get("series", [])):
        a = _fade(u, R.rt(si + 1, 1.2 + si * 1.1, 0.7), 0.6)
        if a <= 0.01:
            continue
        g = _ease((u - R.rt(si + 1, 1.2 + si * 1.1, 0.7)) / 0.9)
        col = _col(R, s.get("colour", "hot" if si else "dim"))
        vs = list(s.get("values", []))[:n]
        pts = []
        for k, v in enumerate(vs):
            th = np.pi / 2 - 2 * np.pi * k / n
            rr = rad * float(v) * g
            pts.append((cx + rr * np.cos(th), cy + rr * np.sin(th)))
        if len(pts) > 2:
            ax.add_patch(Polygon(pts, closed=True, fc=col, ec=col, lw=2.4,
                                 alpha=min(a, 1.0) * 0.30, zorder=6))
            ax.plot([p[0] for p in pts] + [pts[0][0]],
                    [p[1] for p in pts] + [pts[0][1]], lw=2.4, color=col,
                    alpha=min(a, 1.0), zorder=7)
        R.txt(ax, x + 0.3, y + h - 0.4 - si * 0.5, s.get("name", ""), 17, col,
              a, weight="bold")
    _tail(R, ax, o, u, len(o.get("series", [])) + 1, x, y, w)


def two_banks(R, ax, u, d, o):
    """Eq. 12 / 13: two prototype banks, two counters, two step sizes."""
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o, (8.30, 2.30, 6.90, 5.10))
    half = w / 2 - 0.25

    for side, (lab, col, cnt, eta, n) in enumerate([
            ("K ID prototypes", C["pos"], r"$c^+_i$", r"$\rho/\sqrt{c^+_i}$", 6),
            ("L OOD pseudo-prototypes", C["neg"], r"$c^-_i$",
             r"$\rho/\sqrt{c^-_i}$", 8)]):
        a = _fade(u, R.rt(side, 0.5 + side * 1.4, 0.7), 0.6)
        if a <= 0.01:
            continue
        bx = x + side * (half + 0.5)
        R.box(ax, bx, y + 1.55, half, h - 2.1, a=a * 0.5, fc=col, r=0.14, z=3)
        R.txt(ax, bx + half / 2, y + h - 0.85, lab, 17, col, a, ha="center",
              weight="bold")
        rng = np.random.default_rng(20 + side)
        for k in range(n):
            px = bx + 0.45 + (k % 3) * (half - 0.9) / 2
            py = y + h - 1.75 - (k // 3) * 0.75
            g = _fade(u, R.rt(side, 0.5 + side * 1.4, 0.7) + k * 0.06, 0.4)
            ax.plot([px], [py], "*", ms=17, color=col, alpha=min(g, 1.0) * a,
                    zorder=7)
        ac = _fade(u, R.rt(2 + side, 3.0 + side * 0.9, 0.8), 0.6)
        R.txt(ax, bx + half / 2, y + 1.15, "counter " + cnt, 16, C["muted"], ac,
              ha="center")
        R.txt(ax, bx + half / 2, y + 0.62, "step " + eta, 17, C["ink"], ac,
              ha="center")

    a4 = _fade(u, R.rt(3, 4.4, 0.9), 0.7)
    R.txt(ax, x + w / 2, y + 0.12,
          o.get("verdict", ""), 17, C["accent"],
          a4, ha="center", weight="bold")


def wave_stack(R, ax, u, d, o):
    """Sinusoids at geometrically spaced frequencies, drawn one per beat —
    positional encodings, Fourier features, multi-scale bases.

    opts: n_waves, base_freq, ratio, dim_labels, note, verdict
    """
    C, P = R.c, _pal(R)
    x, y, w, h = _box(o)
    n = int(o.get("n_waves", 5))
    base = float(o.get("base_freq", 1.0))
    ratio = float(o.get("ratio", 2.1))
    lane = (h - 1.6) / max(n, 1)
    xs = np.linspace(0, 1, 300)
    labels = o.get("dim_labels", [])

    for k in range(n):
        a = _fade(u, R.rt(k, 0.5 + k * 0.7, 0.55), 0.5)
        if a <= 0.01:
            continue
        g = _ease((u - R.rt(k, 0.5 + k * 0.7, 0.55)) / 0.9)
        m = max(3, int(len(xs) * g))
        yy = y + h - 0.9 - k * lane
        f = base * (ratio ** k)
        col = P["vis"] if k % 2 == 0 else P["txt"]
        ax.plot(x + 1.6 + xs[:m] * (w - 2.3),
                yy + np.sin(2 * np.pi * f * xs[:m]) * lane * 0.34,
                lw=2.4, color=col, alpha=min(a, 1.0), zorder=6)
        lab = labels[k] if k < len(labels) else f"dim {2 * k}"
        R.txt(ax, x + 1.45, yy, str(lab), 15, C["muted"], a, ha="right")

    a2 = _fade(u, R.rt(n, 0.5 + n * 0.7, 0.8), 0.7)
    R.txt(ax, x + 0.3, y + 0.62, o.get("note", ""), 17, C["muted"], a2)
    _tail(R, ax, o, u, n + 1, x, y, w)



PRIMITIVES = {
    "arrow_cycle":           arrow_cycle,
    "budget_ladder":         budget_ladder,
    "connectivity_compare":  connectivity_compare,
    "exponent_bars":         exponent_bars,
    "graph_nodes":           graph_nodes,
    "nested_boxes":          nested_boxes,
    "queue_buffer":          queue_buffer,
    "radial_compare":        radial_compare,
    "two_banks":             two_banks,
    "wave_stack":            wave_stack,
}

PRESETS = {
    # === Attention Is All You Need ========================================
    "positional_encoding": ("wave_stack", {
        "n_waves": 5, "base_freq": 1.0, "ratio": 2.2,
        "dim_labels": ["dim 0", "dim 2", "dim 4", "dim 6", "dim 8"],
        "note": r"$PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$",
        "verdict": "geometric frequencies, so relative offsets are linear"}),
    "path_length": ("connectivity_compare", {
        "n_positions": 8, "target": 7,
        "variants": [{"label": "Recurrent", "kind": "chain",
                      "note": r"path length $\mathcal{O}(n)$"},
                     {"label": "Convolutional", "kind": "local", "k": 2,
                      "note": r"path length $\mathcal{O}(\log_k n)$"},
                     {"label": "Self-Attention", "kind": "all",
                      "note": r"path length $\mathcal{O}(1)$"}],
        "verdict": "any position reaches any other in one step"}),

    # === Graph Neural Networks (Kipf & Welling, 2017) =====================
    "message_passing": ("graph_nodes", {
        "nodes": [{"label": "a"}, {"label": "b"}, {"label": "c"},
                  {"label": "d"}, {"label": "e"}],
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 2]],
        "waves": 3,
        "verdict": "each layer moves information one hop further"}),

    # === Transformer-XL / KV cache work ===================================
    "kv_eviction": ("queue_buffer", {
        "slots": 8, "kept_label": "KV cache", "evicted_label": "evicted",
        "verdict": "a bounded cache means a bounded context"}),

    # === Inference-Optimal VLMs ===========================================
    "alpha_beta": ("exponent_bars", {
        "a_label": r"$\alpha$   LLM parameters",
        "b_label": r"$\beta$   visual tokens"}),
    "pareto_pick": ("budget_ladder", {
        "title": "fixed inference budget",
        "verdict": "trade tokens for parameters — error falls"}),

    # === Respecting Modality Gap ==========================================
    "two_banks_ood": ("two_banks", {
        "verdict": "separate counters — each decays at its own rate"}),

    # === Scope / capability arguments (generic) ===========================
    "scope_nesting": ("nested_boxes", {
        "layers": [{"label": "all inputs"},
                   {"label": "in-distribution", "colour": "hot"},
                   {"label": "seen in training", "colour": "out"}],
        "verdict": "each contains the next"}),
    "training_loop": ("arrow_cycle", {
        "stages": [["sample", "in"], ["forward", "hot"], ["loss", "neg"],
                   ["update", "out"]],
        "centre_label": "one step", "verdict": "until convergence"}),
    "benchmark_radar": ("radial_compare", {
        "axes": ["GQA", "MMB", "MME", "SQA", "POPE", "VQAv2"]}),
}