#!/usr/bin/env python3
"""Build the page's static assets from the videos in videos/.

    python make_assets.py            posters, figure tiles and captions (fast)
    python make_assets.py --scenes   also count scenes per video (slow: decodes everything)

Writes
    posters/<name>.jpg     one full frame per video, used as the <video> poster
    figures/<name>-<mm-ss>.jpg   frames cropped to the drawing, for the figure wall
    captions/<name>.vtt    the mov_text subtitle track of each video, as WebVTT

Needs only OpenCV (cv2) and NumPy. No ffmpeg.
"""
import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEOS = Path("videos")
BG = (247, 250, 251)                     # frame background, BGR
TILE = (640, 400)

# Frame used as the poster, as a fraction of the running time.
POSTER = {
    "transformer": .15, "diffusion": .15, "lejepa": .25, "dreamer4": .25,
    "good": .55, "vlm_scaling": .35, "lacot": .55, "modality_gap": .25,
    "intuitive_physics": .45,
}
# Frames that go on the figure wall.
WALL = {
    "transformer": [.05, .15, .25, .45, .55, .65],
    "diffusion": [.15],
    "lejepa": [.15, .25, .55, .92],
    "dreamer4": [.05, .15, .25],
    "good": [.45, .55],
    "vlm_scaling": [.35, .65],
    "lacot": [.55],
    "modality_gap": [.25, .65],
    "intuitive_physics": [.05, .15, .45, .55, .65],
}


# ---------------------------------------------------------------- mp4 parsing
def boxes(f, start, end):
    pos = start
    while pos < end - 8:
        f.seek(pos)
        head = f.read(8)
        if len(head) < 8:
            break
        size, typ = struct.unpack(">I4s", head)
        hs = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hs = 16
        if size == 0:
            size = end - pos
        yield typ, pos + hs, pos + size
        pos += size


def find(f, s, e, path):
    for t, s2, e2 in boxes(f, s, e):
        if t == path[0]:
            if len(path) == 1:
                yield s2, e2
            else:
                yield from find(f, s2, e2, path[1:])


def subtitle_cues(path):
    """(start, end, text) for every non-empty sample of the first subtitle track."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        for ts, te in find(f, 0, size, [b"moov", b"trak"]):
            hd = list(find(f, ts, te, [b"mdia", b"hdlr"]))
            f.seek(hd[0][0] + 8)
            if f.read(4) != b"sbtl":
                continue
            mh = list(find(f, ts, te, [b"mdia", b"mdhd"]))[0]
            f.seek(mh[0])
            version = f.read(1)[0]
            f.seek(mh[0] + 4 + (16 if version == 1 else 8))
            tscale = struct.unpack(">I", f.read(4))[0]
            stbl = list(find(f, ts, te, [b"mdia", b"minf", b"stbl"]))[0]

            def box(name):
                r = list(find(f, stbl[0], stbl[1], [name]))
                return r[0] if r else None

            b = box(b"stsz"); f.seek(b[0] + 4)
            fixed, n = struct.unpack(">II", f.read(8))
            sizes = [fixed] * n if fixed else list(struct.unpack(f">{n}I", f.read(4 * n)))
            b = box(b"stco")
            if b:
                f.seek(b[0] + 4); n = struct.unpack(">I", f.read(4))[0]
                offs = list(struct.unpack(f">{n}I", f.read(4 * n)))
            else:
                b = box(b"co64"); f.seek(b[0] + 4); n = struct.unpack(">I", f.read(4))[0]
                offs = list(struct.unpack(f">{n}Q", f.read(8 * n)))
            b = box(b"stsc"); f.seek(b[0] + 4); n = struct.unpack(">I", f.read(4))[0]
            stsc = [struct.unpack(">III", f.read(12)) for _ in range(n)]
            b = box(b"stts"); f.seek(b[0] + 4); n = struct.unpack(">I", f.read(4))[0]
            stts = [struct.unpack(">II", f.read(8)) for _ in range(n)]

            sample_offs, si = [], 0
            for ci, (first, per_chunk, _) in enumerate(stsc):
                last = stsc[ci + 1][0] - 1 if ci + 1 < len(stsc) else len(offs)
                for c in range(first - 1, last):
                    o = offs[c]
                    for _ in range(per_chunk):
                        if si < len(sizes):
                            sample_offs.append(o); o += sizes[si]; si += 1
            times, t = [], 0
            for count, dur in stts:
                for _ in range(count):
                    times.append((t / tscale, (t + dur) / tscale)); t += dur
            cues = []
            for i, o in enumerate(sample_offs):
                f.seek(o)
                n = struct.unpack(">H", f.read(2))[0]
                text = f.read(n).decode("utf-8", "replace").strip()
                if text:
                    cues.append((times[i][0], times[i][1], text))
            return cues
    return []


def vtt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# --------------------------------------------------------------------- frames
def frame_at(cap, frac):
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
    ok, fr = cap.read()
    return fr if ok else None


def tile(frame):
    """Crop the drawing out of a frame (below the heading, above the narration
    caption), trim to content, and letterbox onto a uniform tile."""
    h, w = frame.shape[:2]
    body = frame[int(h * .12):int(h * .86), :]
    diff = np.abs(body.astype(int) - np.array(BG)).sum(axis=2)
    ys, xs = np.where(diff > 36)
    if len(ys) < 50:
        return None
    pad = 24
    y0, y1 = max(0, ys.min() - pad), min(body.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(body.shape[1], xs.max() + pad)
    crop = body[y0:y1, x0:x1]
    s = min(TILE[0] / crop.shape[1], TILE[1] / crop.shape[0], 1.0)
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)
    canvas = np.full((TILE[1], TILE[0], 3), BG, np.uint8)
    oy, ox = (TILE[1] - crop.shape[0]) // 2, (TILE[0] - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas


def stamp(seconds):
    return f"{int(seconds // 60):02d}-{int(seconds % 60):02d}"


def count_scenes(path):
    """Scenes start with a new heading: watch the heading strip every 2 s."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * 2)
    prev, scenes, i = None, 1, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            strip = cv2.cvtColor(fr[20:70, 60:1600], cv2.COLOR_BGR2GRAY)
            sig = (cv2.resize(strip, (160, 5)) < 128)
            if prev is not None and (sig != prev).sum() > 40:
                scenes += 1
            prev = sig
        i += 1
    return scenes


def main():
    for d in ("posters", "figures", "captions"):
        Path(d).mkdir(exist_ok=True)
    wall, meta = [], {}
    for video in sorted(VIDEOS.glob("*.mp4")):
        name = video.stem
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        meta[name] = {"duration": round(dur)}

        fr = frame_at(cap, POSTER.get(name, .2))
        if fr is not None:
            cv2.imwrite(f"posters/{name}.jpg", cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 82])

        for frac in WALL.get(name, []):
            fr = frame_at(cap, frac)
            t = tile(fr) if fr is not None else None
            if t is None:
                print(f"  {name} @ {frac}: nothing to crop"); continue
            secs = dur * frac
            out = f"figures/{name}-{stamp(secs)}.jpg"
            cv2.imwrite(out, t, [cv2.IMWRITE_JPEG_QUALITY, 80])
            wall.append({"src": out, "video": name, "t": round(secs)})

        cues = subtitle_cues(video)
        with open(f"captions/{name}.vtt", "w") as w:
            w.write("WEBVTT\n\n")
            for s, e, text in cues:
                w.write(f"{vtt_time(s)} --> {vtt_time(e)}\n{text}\n\n")
        meta[name]["cues"] = len(cues)
        print(f"{name:18s} {int(dur // 60)}:{int(dur % 60):02d}  {len(cues)} cues  "
              f"{len(WALL.get(name, []))} tiles")

    if "--scenes" in sys.argv:
        for video in sorted(VIDEOS.glob("*.mp4")):
            n = count_scenes(video)
            meta[video.stem]["scenes"] = n
            print(f"{video.stem:18s} {n} scenes", flush=True)

    json.dump({"videos": meta, "wall": wall}, open("assets.json", "w"), indent=1)
    print(f"{len(wall)} tiles; wrote assets.json (paste the wall list into index.html)")


if __name__ == "__main__":
    main()
