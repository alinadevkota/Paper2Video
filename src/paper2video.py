#!/usr/bin/env python3
"""
paper2video.py — turn a paper (PDF) into a narrated explainer video.

    python paper2video.py paper.pdf -o out.mp4        # end to end
    python paper2video.py paper.pdf --plan-only       # write storyboard.json, stop
    python paper2video.py storyboard.json -o out.mp4  # render an edited storyboard

Three stages, and the middle one is the only one that needs judgement:

    1. extract   PDF -> text + page renders + embedded figures        (poppler)
    2. plan      text -> storyboard.json                              (LLM, or offline outline)
    3. speak     narration -> one audio clip per scene                (TTS)
    4. render    storyboard -> MP4 with audio + subtitles             (matplotlib -> ffmpeg)

The output is a finished video: narration on the audio track, captions drawn
into the frames, and a selectable subtitle track muxed in. Scene lengths are
set by how long the narration actually takes to say, so nothing drifts.

The storyboard is a plain JSON file on purpose. Auto-planning a paper is
approximate; editing JSON is not. Run --plan-only, fix the storyboard, then
render. Re-rendering is cheap.

Requirements
    pip install matplotlib numpy pillow
    apt-get install poppler-utils ffmpeg
    export ANTHROPIC_API_KEY=...        # optional; without it, --planner outline

Voice, best first — the script picks whichever is installed:
    pip install kokoro soundfile     # local, natural            --tts kokoro
    pip install edge-tts             # online, no key, natural   --tts edge
    piper + a .onnx voice            # local, natural            --tts piper --voice v.onnx
    macOS built-in `say`                                         --tts say
    apt-get install espeak-ng        # always works, robotic     --tts espeak

Run `python paper2video.py --schema` to print the storyboard format.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from illustrations import FIGURES as _FIGURES
except Exception:                                   # library optional
    _FIGURES = {}

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

# --------------------------------------------------------------------------- #
# theme
# --------------------------------------------------------------------------- #

W, H = 16.0, 9.0          # scene units; 16:9 with equal aspect

THEMES = {
    "paper": dict(bg="#FBFAF7", ink="#12161F", muted="#6E7684", line="#D8D4CB",
                  panel="#F2EFE8", accent="#C98A00", accent_soft="#F6EAD0",
                  accent_ink="#7A5A00", pos="#2563EB", neg="#D2401E",
                  dim="#B9B4A9"),
    "slate": dict(bg="#11151C", ink="#F2F4F8", muted="#98A2B3", line="#2A3140",
                  panel="#1A202B", accent="#F0B429", accent_soft="#2A2416",
                  accent_ink="#F0B429", pos="#5B9CF8", neg="#F4795B",
                  dim="#3C4454"),
}

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["mathtext.fontset"] = "dejavusans"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def fade(u: float, start: float, dur: float = 0.55) -> float:
    if dur <= 0:
        return 1.0 if u >= start else 0.0
    return ease((u - start) / dur)


def hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def have(prog: str) -> bool:
    return shutil.which(prog) is not None


def wrap(text: str, width: int) -> list[str]:
    out: list[str] = []
    for para in str(text).split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return out


# --------------------------------------------------------------------------- #
# generated video clips (B-roll)
# --------------------------------------------------------------------------- #

CLIP_W, CLIP_H = 1280, 720          # clips are decoded at this size, then scaled


class ClipReader:
    """Streams a video file frame-by-frame through an ffmpeg pipe.

    Rendering is sequential, so we pull one frame per output frame instead of
    holding the whole clip in RAM. Short clips hold on their last frame.
    """

    def __init__(self, path: Path, fps: int, loop: bool = False):
        self.path, self.fps, self.loop = Path(path), fps, loop
        self.n = CLIP_W * CLIP_H * 3
        self.last = None
        self.proc = None
        self._open()

    def _open(self):
        self.close()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-i", str(self.path),
             "-vf", f"scale={CLIP_W}:{CLIP_H}:force_original_aspect_ratio=decrease,"
                    f"pad={CLIP_W}:{CLIP_H}:-1:-1:color=black,fps={self.fps}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def frame(self):
        if self.proc is None:
            return self.last
        buf = self.proc.stdout.read(self.n)
        if len(buf) < self.n:
            if self.loop:
                self._open()
                buf = self.proc.stdout.read(self.n)
            if len(buf) < self.n:
                self.close()
                return self.last
        self.last = np.frombuffer(buf, np.uint8).reshape(CLIP_H, CLIP_W, 3)
        return self.last

    def close(self):
        if self.proc:
            try:
                self.proc.stdout.close(); self.proc.kill(); self.proc.wait()
            except Exception:
                pass
            self.proc = None


def clip_duration(path: Path) -> float:
    return audio_len(path) or 0.0


def ensure_clips(board: dict, clips_dir: Path, args) -> None:
    """Materialise every `clip` scene that has a prompt but no file yet.

    Generation is delegated to --video-cmd, a shell template with {prompt},
    {out}, {seconds}, {width}, {height}. That keeps this script model-agnostic:
    point it at ComfyUI, a diffusers script, or a hosted API. Results are
    cached by prompt hash, because these are slow and expensive to make.
    """
    import hashlib
    todo = [sc for sc in board.get("scenes", [])
            if sc.get("type") == "clip" and not sc.get("video") and sc.get("prompt")]
    if not todo:
        return
    if not args.video_cmd:
        print(f"      {len(todo)} clip scene(s) need generating but --video-cmd "
              f"is unset; they will render as a caption card")
        return

    clips_dir.mkdir(parents=True, exist_ok=True)
    for sc in todo:
        secs = float(sc.get("clip_seconds", args.video_seconds))
        key = hashlib.sha1(
            f"{sc['prompt']}|{secs}|{args.video_cmd}".encode()).hexdigest()[:16]
        dst = clips_dir / f"clip_{key}.mp4"
        if dst.exists():
            sc["video"] = str(dst)
            print(f"      cached: {dst.name}")
            continue
        cmd = args.video_cmd.format(prompt=sc["prompt"], out=str(dst),
                                    seconds=secs, width=CLIP_W, height=CLIP_H)
        print(f"      generating {dst.name}: {sc['prompt'][:60]}...")
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            if dst.exists() and dst.stat().st_size > 1000:
                sc["video"] = str(dst)
            else:
                print("      generator produced nothing usable")
        except subprocess.CalledProcessError as e:
            print(f"      generator failed: "
                  f"{(e.stderr or b'')[-300:].decode(errors='ignore')}")


# --------------------------------------------------------------------------- #
# text to speech
# --------------------------------------------------------------------------- #

LEAD, TAIL = 0.45, 0.90          # silence before/after narration inside a scene


def narration_lines(sc: dict) -> list[str]:
    """`narration` is a string, or a list of strings for a multi-beat scene.

    Each beat gets its own audio clip, its own subtitle cue, and reveals the
    next element on screen. This is what keeps a 40-second scene alive.
    """
    n = sc.get("narration", "")
    if isinstance(n, str):
        n = [n]
    return [" ".join(str(x).split()) for x in n if str(x).strip()]


def importable(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def tts_available(voice: str | None) -> list[str]:
    """Every engine installed, best quality first."""
    out = []
    if importable("kokoro") and importable("soundfile"):
        out.append("kokoro")
    if importable("edge_tts"):
        out.append("edge")
    if have("piper") and voice:
        out.append("piper")
    if have("say"):
        out.append("say")
    if have("espeak-ng") or have("espeak"):
        out.append("espeak")
    return out


def tts_detect(voice: str | None) -> str | None:
    got = tts_available(voice)
    return got[0] if got else None


DEFAULT_VOICES = {
    "kokoro": "af_heart",
    "edge": "en-US-AndrewNeural",
    "say": "Samantha",
    "espeak": "en-us+f3",
}

_KOKORO = {}


def _to_wav(src: Path, dst: Path):
    """Normalise anything to 24 kHz mono wav so mixing is predictable."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-ar", "24000", "-ac", "1", str(dst)], check=True)
    if src != dst:
        src.unlink(missing_ok=True)


def synth(text: str, dst: Path, backend: str, voice: str | None, rate: float):
    """Render one narration line to dst (.wav). Raises on failure."""
    voice = voice or DEFAULT_VOICES.get(backend)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if backend == "kokoro":
        import soundfile as sf
        from kokoro import KPipeline
        if "p" not in _KOKORO:
            _KOKORO["p"] = KPipeline(lang_code="a")
        chunks = [a for _, _, a in _KOKORO["p"](text, voice=voice, speed=rate)]
        if not chunks:
            raise RuntimeError("kokoro produced no audio")
        sf.write(str(dst), np.concatenate([np.asarray(c) for c in chunks]), 24000)
        return

    if backend == "edge":
        import asyncio
        import edge_tts
        pct = int(round((rate - 1.0) * 100))
        tmp = dst.with_suffix(".mp3")
        asyncio.run(edge_tts.Communicate(
            text, voice, rate=f"{pct:+d}%").save(str(tmp)))
        _to_wav(tmp, dst)
        return

    if backend == "piper":
        tmp = dst.with_suffix(".piper.wav")
        subprocess.run(["piper", "-m", str(voice), "-f", str(tmp)],
                       input=text, text=True, check=True, capture_output=True)
        _to_wav(tmp, dst)
        return

    if backend == "say":
        tmp = dst.with_suffix(".aiff")
        cmd = ["say", "-r", str(int(180 * rate)), "-o", str(tmp)]
        if voice:
            cmd += ["-v", voice]
        subprocess.run(cmd + [text], check=True, capture_output=True)
        _to_wav(tmp, dst)
        return

    if backend == "espeak":
        exe = "espeak-ng" if have("espeak-ng") else "espeak"
        tmp = dst.with_suffix(".espeak.wav")
        subprocess.run([exe, "-v", voice or "en-us", "-s", str(int(170 * rate)),
                        "-p", "38", "-w", str(tmp), text],
                       check=True, capture_output=True)
        _to_wav(tmp, dst)
        return

    raise RuntimeError(f"unknown tts backend: {backend}")


def build_voice(scenes, vdir: Path, args) -> list[Path | None]:
    """One audio file per scene, cached against the narration text."""
    vdir.mkdir(parents=True, exist_ok=True)
    man_path = vdir / "manifest.json"
    manifest = {}
    if man_path.exists():
        try:
            manifest = json.loads(man_path.read_text())
        except Exception:
            manifest = {}

    backend = args.tts
    if backend == "auto":
        backend = tts_detect(args.voice)
        if backend is None:
            print("      no TTS engine found — video will be silent "
                  "(see --help for install options)")
            return [[] for _ in scenes]
    if backend == "none":
        return [[] for _ in scenes]

    import hashlib
    # if the preferred engine dies mid-run (network blip on edge, a bad model
    # path on piper) drop to the next one installed rather than going silent
    chain = [backend] + [b for b in tts_available(args.voice) if b != backend]
    print(f"      voice: {backend}"
          f"{' / ' + (args.voice or DEFAULT_VOICES.get(backend) or '')}"
          + (f"  (fallback: {', '.join(chain[1:])})" if len(chain) > 1 else ""))

    out: list[list[Path | None]] = []
    made = reused = 0
    for i, sc in enumerate(scenes):
        clips: list[Path | None] = []
        for b, text in enumerate(narration_lines(sc)):
            wav = vdir / f"scene_{i:02d}_{b:02d}.wav"
            tag = f"{i}:{b}"
            key = hashlib.sha1(
                f"{chain[0]}|{args.voice}|{args.tts_rate}|{text}".encode()
            ).hexdigest()
            if wav.exists() and manifest.get(tag) == key:
                clips.append(wav); reused += 1; continue
            last = None
            for n, eng in enumerate(chain):
                try:
                    # edge-tts is a free public endpoint and intermittently
                    # returns an empty response; retry before giving up on it
                    for attempt in range(args.tts_retries):
                        try:
                            synth(text, wav, eng,
                                  args.voice if eng == backend else None,
                                  args.tts_rate)
                            break
                        except Exception as e:
                            last = e
                            wav.unlink(missing_ok=True)
                            if attempt == args.tts_retries - 1:
                                raise
                            time.sleep(1.5 * (attempt + 1))
                            print(f"      {tag}: {eng} attempt "
                                  f"{attempt + 1} failed, retrying")
                    if n:
                        print(f"      {tag}: {chain[0]} failed, used {eng}")
                        chain = [eng] + [x for x in chain if x != eng]
                    manifest[tag] = hashlib.sha1(
                        f"{chain[0]}|{args.voice}|{args.tts_rate}|{text}"
                        .encode()).hexdigest()
                    clips.append(wav); made += 1
                    break
                except Exception as e:
                    last = e
            else:
                print(f"      {tag}: no engine could speak it ({last}); silent")
                clips.append(None)
        out.append(clips)
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"      {made} clips synthesised, {reused} reused -> {vdir}")
    return out


# --------------------------------------------------------------------------- #
# 1. extraction
# --------------------------------------------------------------------------- #

@dataclass
class Extracted:
    text: str
    pages: list[Path] = field(default_factory=list)     # full-page renders
    figures: list[Path] = field(default_factory=list)   # embedded rasters
    n_pages: int = 0


def extract(pdf: Path, assets: Path, dpi: int = 130, max_pages: int = 40) -> Extracted:
    """Pull out text, page renders and embedded images using poppler."""
    assets.mkdir(parents=True, exist_ok=True)
    if not have("pdftotext"):
        raise SystemExit("poppler-utils not found (need pdftotext/pdftoppm). "
                         "apt-get install poppler-utils")

    n_pages = 0
    try:
        info = run(["pdfinfo", str(pdf)]).stdout
        m = re.search(r"^Pages:\s+(\d+)", info, re.M)
        n_pages = int(m.group(1)) if m else 0
    except Exception:
        pass

    txt = assets / "paper.txt"
    run(["pdftotext", "-layout", str(pdf), str(txt)])
    text = txt.read_text(errors="ignore")

    # page renders: needed for vector figures, which pdfimages cannot see
    last = min(n_pages or max_pages, max_pages)
    run(["pdftoppm", "-png", "-r", str(dpi), "-f", "1", "-l", str(last),
         str(pdf), str(assets / "page")])
    pages = sorted(assets.glob("page-*.png"))

    figures: list[Path] = []
    if have("pdfimages"):
        try:
            run(["pdfimages", "-png", "-f", "1", "-l", str(last),
                 str(pdf), str(assets / "fig")])
            # drop masks / decorative slivers
            figures = [p for p in sorted(assets.glob("fig-*.png"))
                       if p.stat().st_size > 25_000]
        except subprocess.CalledProcessError:
            pass

    return Extracted(text=text, pages=pages, figures=figures, n_pages=n_pages or last)


# --------------------------------------------------------------------------- #
# 2. planning
# --------------------------------------------------------------------------- #

SCHEMA_DOC = """
storyboard.json
---------------
{
  "title":  "short title for the deck",
  "theme":  "paper" | "slate",
  "scenes": [ <scene>, ... ]
}

Every scene accepts:
  "narration": str     spoken/subtitle line; also sets the default duration
  "seconds":   float   explicit duration, overrides the narration estimate

Scene types
-----------
{"type":"title",     "title":str, "subtitle":str, "footer":str}
{"type":"bullets",   "heading":str, "items":[{"head":str,"body":str}, ...]}   # <=5
{"type":"equation",  "heading":str, "label":str, "latex":str, "note":str,
                     "figure":str, "opts":{...}}   # figure -> side-by-side layout
{"type":"figure",    "heading":str, "image":str, "caption":str,
                     "crop":[x0,y0,x1,y1], "zoom":bool}
{"type":"split",     "heading":str, "lines":[str,...], "image":str,
                     "crop":[...], "caption":str}
{"type":"bars",      "heading":str, "subhead":str, "note":str,
                     "series":[{"label":str,"value":float,"highlight":bool,
                                "annot":str}, ...],
                     "lower_is_better":bool}
{"type":"clip",      "heading":str, "caption":str,
                     "prompt":str,        # generate via --video-cmd
                     "video":str,         # or point at an existing file
                     "clip_seconds":float, "fit":"cover"|"contain"}
{"type":"illustration",
                     "figure":str,        # a primitive or preset name
                     "heading":str, "caption":str, "opts":{...}}
{"type":"banner",    "text":str, "heading":str}
{"type":"takeaways", "heading":str, "items":[{"head":str,"body":str}, ...],
                     "caveat":str, "footer":str}

"image" is a path (relative to the storyboard file or the assets dir), or
"page:N" to use the render of page N. "crop" is [x0,y0,x1,y1] in 0-1
fractions of that image — the way to grab a vector figure off a page.
`illustration` draws an animated diagram whose reveals follow the narration
beats. Use it whenever a concept is spatial or procedural — a decomposition, a
migration, a routing decision.

"figure" names either a PRIMITIVE (generic; you supply the content through
"opts") or a PRESET (a primitive with one paper's values already filled in;
your "opts" still override). Run `--figures` for the live list. An `equation`
scene also accepts "figure", which moves the formula to a left panel and runs
the diagram beside it.

Figures live in illustrations.py, with the long tail in illustrations_extra.py.
Add a preset for a new paper; add a primitive only when the shape is new.

`clip` plays generated or supplied footage full-bleed with text over a scrim.
Use it for intuition only — motion, atmosphere, a physical analogy. Never put
an equation, a number or a claim inside one: video models cannot render text
or data reliably, and the frame is not reproducible.

"latex" is matplotlib mathtext: a LaTeX subset. \\Big, \\dfrac and \\text are
rewritten automatically; anything unparseable falls back to plain text.
"""

PLAN_PROMPT = """You are turning a research paper into a short explainer video.

Return ONLY a JSON object matching this schema. No prose, no markdown fences.

{schema}

Rules:
- 7 to 10 scenes. Open with `title`, close with `takeaways`.
- Follow the paper's own logic: setting -> what everyone does now -> the gap
  this paper identifies -> the method -> the evidence -> what to take away.
- Figure names you may use in "figure" (any other name will not render):
  {figures}
- Use `illustration` for anything spatial or procedural. A vector decomposing
  out of a subspace, prototypes migrating toward clusters, a stream being
  routed — these must be shown, not bulleted. Prefer it over `bullets`
  whenever the content has geometry or a sequence of states.
- Use `equation` for the two or three formulas that actually carry the
  argument. Copy them faithfully from the paper. Do not invent notation.
- Use `bars` for the headline table. Copy the real numbers. Set
  "lower_is_better" correctly (FPR95 lower, AUROC/accuracy higher). Mark the
  paper's own method with "highlight": true.
- `narration` is one or two spoken sentences per scene, plain and concrete.
  Say what the thing is, not that it is important.
- The final `takeaways` scene must include a "caveat" naming a real limitation
  visible in the paper (assumptions, ablations that degrade, unsolved regimes).
  Do not write a promotional summary.
- You may use at most two `clip` scenes, and only where motion genuinely helps
  intuition (an analogy, a physical process, an atmospheric opener). Give them
  a "prompt" describing the footage. Never put equations, numbers or claims in
  a clip — those belong in `equation`, `bars` or `figure`.
- Available figure assets you may reference in "image": {assets}
  Prefer "page:N" with a "crop" when the figure is vector-drawn.

Paper text follows.
---
{paper}
"""


def _extract_json(raw: str) -> dict:
    raw = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw.strip())
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)   # reasoning models
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError("planner did not return JSON:\n" + raw[:500])
    return json.loads(raw[start:end + 1])


def plan_with_llm(text: str, assets_note: str, model: str, max_chars: int,
                  api_base: str | None = None, key_env: str = "ANTHROPIC_API_KEY"
                  ) -> dict:
    """Anthropic by default; any OpenAI-compatible endpoint via --api-base.

    That covers Ollama, LM Studio, llama.cpp, vLLM and friends — i.e. free,
    local planning with no account at all.
    """
    import urllib.request

    try:
        from illustrations import FIGURES as _F
        fig_list = ", ".join(sorted(_F))
    except Exception:
        fig_list = "(illustration library unavailable)"
    prompt = PLAN_PROMPT.format(schema=SCHEMA_DOC, assets=assets_note,
                                figures=fig_list, paper=text[:max_chars])
    key = os.environ.get(key_env)

    if api_base:
        url = api_base.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        body = json.dumps({
            "model": model, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        headers = {"content-type": "application/json",
                   "authorization": f"Bearer {key or 'local'}"}
    else:
        if not key:
            raise RuntimeError(f"{key_env} not set")
        url = "https://api.anthropic.com/v1/messages"
        body = json.dumps({
            "model": model, "max_tokens": 8000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        headers = {"content-type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}

    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.load(r)

    if api_base:
        raw = payload["choices"][0]["message"]["content"]
    else:
        raw = "".join(b.get("text", "") for b in payload.get("content", []))
    return _extract_json(raw)


SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(abstract|introduction|related work|preliminar\w*|"
    r"background|motivation|method\w*|approach|experiments?|results?|"
    r"ablation\w*|discussion|conclusion\w*)\b.*$", re.I | re.M)


def plan_outline(text: str, ex: Extracted, title_hint: str) -> dict:
    """Offline fallback: a structurally sane skeleton meant to be edited."""
    lines = [l.rstrip() for l in text.splitlines()]
    head = [l.strip() for l in lines[:40] if l.strip()]
    title = title_hint
    if head:
        cand = []
        for l in head[:6]:
            if SECTION_RE.match(l) or re.match(
                    r"^(arxiv|proceedings|copyright|\d)", l, re.I):
                if cand:
                    break
                continue
            if len(l) > 12:
                cand.append(l)
            if cand and len(" ".join(cand)) > 45:
                break
        if cand:
            title = " ".join(cand)[:120]

    # section slices
    marks = [(m.start(), m.group(1).lower()) for m in SECTION_RE.finditer(text)]
    marks.append((len(text), "end"))
    sections: dict[str, str] = {}
    for i in range(len(marks) - 1):
        name = marks[i][1]
        body = text[marks[i][0]:marks[i + 1][0]]
        sections.setdefault(name, "")
        if len(body) > len(sections[name]):
            sections[name] = body

    def sentences(key: str, n: int) -> list[str]:
        body = sections.get(key, "")
        body = re.sub(r"\s+", " ", body)
        body = re.sub(r"^\s*\w[\w\s]{0,30}?\b", "", body, count=1)
        out = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if 40 < len(s) < 260]
        return out[:n]

    scenes: list[dict] = [{
        "type": "title", "title": title, "subtitle": "",
        "footer": f"auto-outlined from {ex.n_pages} pages — edit this file",
        "narration": title, "seconds": 6.0,
    }]

    for key, heading in [("abstract", "What the paper claims"),
                         ("introduction", "The setting"),
                         ("method", "The method"),
                         ("experiments", "The evidence")]:
        sents = sentences(key, 3)
        if not sents:
            continue
        scenes.append({
            "type": "bullets", "heading": heading,
            "items": [{"head": "", "body": s} for s in sents],
            "narration": sents[0][:190],
        })

    for i, fig in enumerate(ex.figures[:2]):
        scenes.append({
            "type": "figure", "heading": f"Figure {i + 1}",
            "image": fig.name, "caption": "replace with the real caption",
            "narration": "Walk through what this figure shows.",
        })

    concl = sentences("conclusion", 3) or sentences("abstract", 3)
    scenes.append({
        "type": "takeaways", "heading": "What to take away",
        "items": [{"head": "", "body": s} for s in concl[:3]],
        "caveat": "Replace with a real limitation from the paper.",
        "narration": "The main points, and what the paper does not settle.",
    })
    return {"title": title, "theme": "paper", "scenes": scenes}


# --------------------------------------------------------------------------- #
# 3. rendering
# --------------------------------------------------------------------------- #

TEX_FIXES = [
    (r"\\Big([([{|])", r"\\left\1"), (r"\\Big([)\]}|])", r"\\right\1"),
    (r"\\big([([{|])", r"\\left\1"), (r"\\big([)\]}|])", r"\\right\1"),
    (r"\\Bigg?l", r"\\left"), (r"\\Bigg?r", r"\\right"),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"),
    (r"\\text\b", r"\\mathrm"), (r"\\textbf\b", r"\\mathbf"),
    (r"\\textit\b", r"\\mathit"), (r"\\bm\b", r"\\mathbf"),
    (r"\\operatorname\b", r"\\mathrm"), (r"\\nonumber", ""),
    (r"\\label\{[^}]*\}", ""), (r"\\!", ""), (r"\\;", r"\\,"),
    (r"\\triangleq", r"\\equiv"), (r"\\coloneqq", r"="),
]


_TEX_CACHE: dict = {}


def safe_mathtext(s: str) -> str:
    """Rewrite common unsupported macros and verify the string parses.

    Labels reach matplotlib from dozens of places (captions, notes, axis
    titles, preset strings). A single unsupported macro used to abort the whole
    render, so every string containing '$' passes through here first.
    """
    if s in _TEX_CACHE:
        return _TEX_CACHE[s]
    out = s
    for pat, rep in TEX_FIXES:
        out = re.sub(pat, rep, out)
    try:
        from matplotlib import mathtext
        mathtext.MathTextParser("path").parse(out, 100, None)
    except Exception:
        out = re.sub(r"\\[a-zA-Z]+", "", out).replace("$", "")
    _TEX_CACHE[s] = out
    return out


def tex(s: str) -> str:
    """Best-effort LaTeX -> matplotlib mathtext; verified before use."""
    out = s.strip().strip("$")
    for pat, rep in TEX_FIXES:
        out = re.sub(pat, rep, out)
    expr = f"${out}$"
    try:
        from matplotlib import mathtext
        mathtext.MathTextParser("path").parse(expr, 100, None)
        return expr
    except Exception:
        return re.sub(r"[\\{}$]", "", s)      # legible plain-text fallback


class Renderer:
    def __init__(self, theme: str, base: Path, fps: int, size):
        self.c = THEMES.get(theme, THEMES["paper"])
        self.base = base
        self.fps = fps
        self.size = size
        self._imgs: dict[str, np.ndarray] = {}
        self.beats: list = []
        self.reader = None

    def scrim(self, ax, y0, y1, a_top, a_bot, n=256):
        """Black gradient with a genuine alpha ramp, so text stays readable
        over arbitrary footage without hiding the footage itself."""
        ramp = np.linspace(a_top, a_bot, n).reshape(-1, 1, 1)
        img = np.concatenate([np.zeros((n, 1, 3)), ramp], axis=2)
        ax.imshow(img, extent=[0, W, y0, y1], zorder=3, aspect="auto",
                  interpolation="bilinear")
        ax.set_xlim(0, W); ax.set_ylim(0, H)

    def rt(self, k: int, start: float, step: float) -> float:
        """Reveal time for element k: locked to beat k when the narration has
        beats, otherwise the old evenly-spread schedule."""
        if self.beats:
            i = min(k, len(self.beats) - 1)
            return max(0.0, self.beats[i].start - 0.2)
        return start + k * step

    # ---- primitives ------------------------------------------------------ #
    def axes(self, fig):
        fig.clf()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W); ax.set_ylim(0, H)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Rectangle((0, 0), W, H, fc=self.c["bg"], ec="none", zorder=0))
        return ax

    def txt(self, ax, x, y, s, size=22, color=None, a=1.0, ha="left", va="center",
            weight="normal", style="normal", z=10, rot=0):
        if a <= 0.004 or s in (None, ""):
            return
        if isinstance(s, str) and "$" in s:
            s = safe_mathtext(s)
        ax.text(x, y, s, fontsize=size, color=color or self.c["ink"],
                alpha=min(a, 1.0), ha=ha, va=va, weight=weight, style=style,
                zorder=z, rotation=rot)

    def box(self, ax, x, y, w, h, a=1.0, fc=None, r=0.14, z=1):
        if a <= 0.004:
            return
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc or self.c["panel"], ec="none", alpha=min(a, 1.0), zorder=z))

    def head(self, ax, num, title, a=1.0):
        if a <= 0.004:
            return
        if num:
            self.txt(ax, 0.85, 8.30, num, 17, self.c["accent"], a, weight="bold")
        self.txt(ax, 1.65 if num else 0.85, 8.30, title, 27, self.c["ink"], a,
                 weight="bold")
        ax.plot([0.85, 15.15], [7.92, 7.92], color=self.c["line"], lw=1.4,
                alpha=min(a, 1.0), zorder=3)

    def caption(self, ax, text):
        if not text:
            return
        col = getattr(self, "cap_color", None) or self.c["muted"]
        for i, ln in enumerate(wrap(text, 92)[:2]):
            self.txt(ax, W / 2, 0.92 - i * 0.42, ln, 19, col,
                     1.0, ha="center", z=12)

    # ---- images ---------------------------------------------------------- #
    def image(self, spec, pages: list[Path]):
        if spec in self._imgs:
            return self._imgs[spec]
        from PIL import Image
        if spec.startswith("page:"):
            i = int(spec.split(":", 1)[1]) - 1
            if not (0 <= i < len(pages)):
                return None
            path = pages[i]
        else:
            path = Path(spec)
            for cand in (path, self.base / spec, self.base / "assets" / spec):
                if cand.exists():
                    path = cand
                    break
            else:
                return None
        im = Image.open(path).convert("RGBA")
        flat = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flat.alpha_composite(im)
        arr = np.asarray(flat.convert("RGB"))
        self._imgs[spec] = arr
        return arr

    def draw_image(self, ax, arr, cx, cy, maxw, maxh, a=1.0, crop=None, zoom=0.0):
        if arr is None or a <= 0.004:
            return
        h, w = arr.shape[:2]
        if crop:
            x0, y0, x1, y1 = crop
            arr = arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
            h, w = arr.shape[:2]
        if zoom > 0:                              # slow push-in
            dy, dx = int(h * zoom * 0.5), int(w * zoom * 0.5)
            if dy > 1 and dx > 1:
                arr = arr[dy:h - dy, dx:w - dx]
                h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return
        scale = min(maxw / w, maxh / h)
        dw, dh = w * scale, h * scale
        ax.imshow(arr, extent=[cx - dw / 2, cx + dw / 2, cy - dh / 2, cy + dh / 2],
                  alpha=min(a, 1.0), zorder=6, interpolation="bilinear")
        ax.set_xlim(0, W); ax.set_ylim(0, H)

    # ---- scene types ----------------------------------------------------- #
    def scene(self, ax, sc, u, d, idx, pages):
        kind = sc.get("type", "bullets")
        getattr(self, f"_{kind}", self._bullets)(ax, sc, u, d, idx, pages)

    def _title(self, ax, sc, u, d, idx, pages):
        a = fade(u, 0.3, 0.9)
        lines = wrap(sc.get("title", ""), 44)[:4]
        y = 6.35 + 0.46 * (len(lines) - 1)
        for ln in lines:
            self.txt(ax, W / 2, y, ln, 42, self.c["ink"], a, ha="center", weight="bold")
            y -= 0.92
        b = fade(u, 1.3, 0.8)
        ax.plot([6.0, 10.0], [y - 0.10, y - 0.10], color=self.c["accent"], lw=2.6,
                alpha=min(b, 1.0))
        self.txt(ax, W / 2, y - 0.85, sc.get("subtitle", ""), 24, self.c["muted"],
                 b, ha="center")
        self.txt(ax, W / 2, y - 1.45, sc.get("footer", ""), 19, self.c["muted"],
                 fade(u, 2.2, 0.8), ha="center")

    def _bullets(self, ax, sc, u, d, idx, pages):
        self.head(ax, f"{idx:02d}", sc.get("heading", ""), fade(u, 0.1, 0.5))
        items = sc.get("items", [])[:5]
        step = min(0.85, max(0.45, (d * 0.55) / max(len(items), 1)))
        top, gap = 6.90, min(1.35, 5.4 / max(len(items), 1))
        for k, it in enumerate(items):
            a = fade(u, self.rt(k, 0.5, step), 0.55)
            if a <= 0.004:
                continue
            y = top - k * gap
            ax.plot([1.05], [y], "o", ms=10, color=self.c["accent"], alpha=min(a, 1.0))
            head = it.get("head", "")
            body = it.get("body", "")
            yy = y
            if head:
                self.txt(ax, 1.55, y + 0.22, head, 23, self.c["ink"], a, weight="bold")
                yy = y - 0.30
            for j, ln in enumerate(wrap(body, 84)[:3]):
                self.txt(ax, 1.55, yy - j * 0.44, ln, 19, self.c["muted"], a)

    def _equation(self, ax, sc, u, d, idx, pages):
        """A formula. With "figure" set, the formula moves to a left panel and
        an animated diagram runs beside it — an equation almost always reads
        better with a picture of what it does."""
        self.head(ax, f"{idx:02d}", sc.get("heading", ""), fade(u, 0.1, 0.5))
        a = fade(u, 0.5, 0.6)
        fig = sc.get("figure")
        fn = _FIGURES.get(fig) if fig else None

        if fn is None:
            self.box(ax, 0.85, 3.30, 14.30, 3.60, a * 0.9)
            self.txt(ax, 1.35, 6.45, sc.get("label", ""), 21, self.c["accent"],
                     a, weight="bold")
            self.txt(ax, W / 2, 5.35, tex(sc.get("latex", "")), 28,
                     self.c["ink"], a, ha="center")
            b = fade(u, self.rt(1, 1.6, 1.6), 0.6)
            for j, ln in enumerate(wrap(sc.get("note", ""), 96)[:3]):
                self.txt(ax, 1.35, 4.08 - j * 0.42, ln, 18, self.c["muted"], b)
            return

        self.box(ax, 0.85, 2.20, 6.55, 5.10, a * 0.9)
        self.txt(ax, 1.25, 6.85, sc.get("label", ""), 20, self.c["accent"], a,
                 weight="bold")
        self.txt(ax, 4.12, 5.55, tex(sc.get("latex", "")), 22, self.c["ink"], a,
                 ha="center")
        b = fade(u, self.rt(1, 1.6, 1.6), 0.6)
        for j, ln in enumerate(wrap(sc.get("note", ""), 44)[:5]):
            self.txt(ax, 1.25, 4.05 - j * 0.44, ln, 17, self.c["muted"], b)

        opts = dict(sc.get("opts", {}) or {})
        opts.setdefault("box", (8.15, 2.20, 7.00, 5.10))
        fn(self, ax, u, d, opts)

    def _figure(self, ax, sc, u, d, idx, pages):
        self.head(ax, f"{idx:02d}", sc.get("heading", ""), fade(u, 0.1, 0.5))
        a = fade(u, 0.4, 0.8)
        z = 0.045 * ease(u / max(d, 0.1)) if sc.get("zoom") else 0.0
        self.draw_image(ax, self.image(sc.get("image", ""), pages), W / 2, 4.75,
                        13.4, 5.4, a, sc.get("crop"), z)
        b = fade(u, self.rt(1, 1.2, 1.2), 0.6)
        for j, ln in enumerate(wrap(sc.get("caption", ""), 96)[:2]):
            self.txt(ax, W / 2, 1.85 - j * 0.42, ln, 19, self.c["muted"], b, ha="center")

    def _split(self, ax, sc, u, d, idx, pages):
        self.head(ax, f"{idx:02d}", sc.get("heading", ""), fade(u, 0.1, 0.5))
        lines = sc.get("lines", [])[:6]
        step = min(0.8, max(0.4, (d * 0.5) / max(len(lines), 1)))
        y = 6.75
        for k, ln in enumerate(lines):
            a = fade(u, self.rt(k, 0.5, step), 0.5)
            for j, seg in enumerate(wrap(ln, 42)[:3]):
                self.txt(ax, 0.95, y - j * 0.44, seg, 20,
                         self.c["ink"] if j == 0 else self.c["muted"], a)
                y -= 0.44
            y -= 0.42
        a = fade(u, 0.6, 0.8)
        z = 0.04 * ease(u / max(d, 0.1)) if sc.get("zoom") else 0.0
        self.draw_image(ax, self.image(sc.get("image", ""), pages), 11.45, 4.60,
                        7.0, 5.2, a, sc.get("crop"), z)
        self.txt(ax, 11.45, 1.70, sc.get("caption", ""), 18, self.c["muted"],
                 fade(u, 1.4, 0.6), ha="center")

    def _bars(self, ax, sc, u, d, idx, pages):
        self.head(ax, f"{idx:02d}", sc.get("heading", "Results"), fade(u, 0.1, 0.4))
        a = fade(u, 0.3, 0.5)
        self.txt(ax, 0.85, 7.28, sc.get("subhead", ""), 22, self.c["ink"], a,
                 weight="bold")
        lo = sc.get("lower_is_better", False)
        self.txt(ax, 0.85, 6.82,
                 sc.get("note", "lower is better" if lo else "higher is better"),
                 17, self.c["muted"], a)

        series = sc.get("series", [])[:7]
        if not series:
            return
        vals = [float(s.get("value", 0)) for s in series]
        vmax = max(vals) or 1.0
        dec = min(2, max((len(str(s.get("value", 0)).split(".")[-1])
                          if "." in str(s.get("value", 0)) else 0)
                         for s in series))
        labw = max(len(str(s.get("label", ""))) for s in series)
        x0 = min(5.6, 1.5 + labw * 0.16)
        span = 8.4
        annots = any(s.get("annot") for s in series)
        span = span if annots else span + 1.6
        scale = span / (vmax * 1.06)

        gap = min(0.78, 5.0 / len(series))
        top = 6.28
        step = min(0.5, (d * 0.45) / len(series))
        for k, s in enumerate(series):
            t0 = self.rt(k, 0.9, step)
            ab = fade(u, t0, 0.45)
            if ab <= 0.004:
                continue
            grow = ease((u - t0) / 0.9)
            y = top - k * gap
            hi = bool(s.get("highlight"))
            col = self.c["accent"] if hi else self.c["dim"]
            v = float(s.get("value", 0))
            ax.add_patch(Rectangle((x0, y - 0.21), v * scale * grow, 0.42, fc=col,
                                   ec="none", alpha=min(ab, 1.0), zorder=6))
            self.txt(ax, x0 - 0.28, y, s.get("label", ""), 19,
                     self.c["ink"] if hi else self.c["muted"], ab, ha="right",
                     weight="bold" if hi else "normal")
            self.txt(ax, x0 + v * scale * grow + 0.24, y, f"{v:.{dec}f}", 19,
                     self.c["ink"] if hi else self.c["muted"], ab * grow,
                     weight="bold" if hi else "normal")
            if s.get("annot"):
                self.txt(ax, 15.15, y, s["annot"], 16, self.c["muted"],
                         ab * grow * 0.9, ha="right")

        foot = sc.get("footer", "")
        if foot:
            af = fade(u, self.rt(len(series), 1.2, step), 0.7)
            self.box(ax, 0.85, 1.35, 14.30, 1.30, af, fc=self.c["accent_soft"], r=0.12)
            for j, ln in enumerate(wrap(foot, 88)[:2]):
                self.txt(ax, W / 2, 2.24 - j * 0.48, ln, 20, self.c["accent_ink"],
                         af, ha="center", weight="bold")

    def _clip(self, ax, sc, u, d, idx, pages):
        """Generated (or supplied) video, full-bleed, with text over the top.

        Use this for intuition — motion, atmosphere, a physical analogy. Never
        for equations, numbers or anything a viewer might quote: a video model
        cannot be trusted to render text or data correctly.
        """
        src = sc.get("video")
        arr = self.reader.frame() if self.reader else None
        if arr is not None:
            fit = sc.get("fit", "cover")
            ih, iw = arr.shape[:2]
            if fit == "cover":                       # fill the frame, crop over
                scale = max(W / iw, H / ih)
            else:                                    # letterbox inside the frame
                scale = min(W / iw, H / ih)
            dw, dh = iw * scale, ih * scale
            ax.imshow(arr, extent=[W / 2 - dw / 2, W / 2 + dw / 2,
                                   H / 2 - dh / 2, H / 2 + dh / 2],
                      zorder=2, interpolation="bilinear")
            ax.set_xlim(0, W); ax.set_ylim(0, H)
            # scrim so overlaid text stays legible over any footage
            self.scrim(ax, H - 2.5, H, 0.70, 0.0)
            self.scrim(ax, 0.0, 3.9, 0.0, 0.86)
            over, sub = "#FFFFFF", "#EDEDED"
        else:
            if src:
                self.txt(ax, W / 2, 5.2, "clip missing", 24, self.c["muted"],
                         1.0, ha="center")
            over, sub = self.c["ink"], self.c["muted"]

        a = fade(u, 0.3, 0.7)
        if sc.get("heading"):
            self.txt(ax, 0.9, 8.0, sc["heading"], 30, over, a, weight="bold", z=12)
        lines = wrap(sc.get("caption", ""), 74)[:2]
        for j, ln in enumerate(lines):
            self.txt(ax, 0.9, 2.85 - j * 0.5, ln, 22, sub,
                     fade(u, self.rt(1, 0.9, 0.9), 0.6), z=12)

    def _illustration(self, ax, sc, u, d, idx, pages):
        """An animated diagram, drawn frame by frame, synced to the narration.

        This is the scene type for concepts that need to be *shown* — a vector
        decomposing, prototypes migrating, a stream being routed — rather than
        described in a bullet. See illustrations.py for the figure registry.
        """
        self.head(ax, f"{idx:02d}", sc.get("heading", ""), fade(u, 0.1, 0.5))
        name = sc.get("figure", "")
        fn = _FIGURES.get(name)
        if fn is None:
            # with a hundred-plus figures, listing them all is useless; show
            # the nearest matches instead, which catches typos immediately
            import difflib
            near = difflib.get_close_matches(name, _FIGURES, n=5, cutoff=0.4)
            self.txt(ax, W / 2, 5.2,
                     f"unknown illustration: {name or '(none)'}", 22,
                     self.c["neg"], 1.0, ha="center")
            self.txt(ax, W / 2, 4.6,
                     ("did you mean:  " + ",  ".join(near)) if near
                     else f"{len(_FIGURES)} figures available",
                     18, self.c["ink"], 1.0, ha="center")
            self.txt(ax, W / 2, 4.0, "run  paper2video.py --figures  for the list",
                     16, self.c["muted"], 1.0, ha="center")
            return
        fn(self, ax, u, d, sc.get("opts", {}) or {})
        cap = sc.get("caption", "")
        if cap:
            a = fade(u, self.rt(max(len(self.beats) - 1, 1), d * 0.6, 1.0), 0.7)
            self.box(ax, 0.85, 1.30, 14.30, 0.92, a, fc=self.c["accent_soft"],
                     r=0.12)
            self.txt(ax, W / 2, 1.76, wrap(cap, 92)[0], 20,
                     self.c["accent_ink"], a, ha="center", weight="bold")

    def _banner(self, ax, sc, u, d, idx, pages):
        if sc.get("heading"):
            self.head(ax, "", sc["heading"], fade(u, 0.1, 0.5))
        a = fade(u, 0.4, 0.8)
        lines = wrap(sc.get("text", ""), 52)[:4]
        y = 5.20 + 0.55 * (len(lines) - 1)
        self.box(ax, 1.35, y - 0.55 * len(lines) - 0.35, 13.30,
                 1.10 * len(lines) + 0.70, a, fc=self.c["accent_soft"], r=0.16)
        for ln in lines:
            self.txt(ax, W / 2, y, ln, 30, self.c["accent_ink"], a, ha="center",
                     weight="bold")
            y -= 1.10

    def _takeaways(self, ax, sc, u, d, idx, pages):
        self.head(ax, f"{idx:02d}", sc.get("heading", "What to take away"),
                  fade(u, 0.1, 0.4))
        items = sc.get("items", [])[:3]
        step = min(1.0, max(0.5, (d * 0.45) / max(len(items), 1)))
        for k, it in enumerate(items):
            a = fade(u, self.rt(k, 0.4, step), 0.6)
            if a <= 0.004:
                continue
            y = 6.72 - k * 1.30
            ax.plot([1.05], [y + 0.06], "o", ms=11, color=self.c["accent"],
                    alpha=min(a, 1.0))
            if it.get("head"):
                self.txt(ax, 1.55, y + 0.24, it["head"], 23, self.c["ink"], a,
                         weight="bold")
            for j, ln in enumerate(wrap(it.get("body", ""), 86)[:2]):
                self.txt(ax, 1.55, y - 0.30 - j * 0.44, ln, 19, self.c["muted"], a)

        cav = sc.get("caveat", "")
        if cav:
            a = fade(u, self.rt(len(items), 0.6, step), 0.7)
            self.box(ax, 0.85, 1.35, 14.30, 2.10, a * 0.9, fc=self.c["panel"])
            self.txt(ax, 1.20, 3.02, "Worth questioning", 19, self.c["neg"], a,
                     weight="bold")
            for j, ln in enumerate(wrap(cav, 100)[:3]):
                self.txt(ax, 1.20, 2.45 - j * 0.46, ln, 17, self.c["muted"], a)
        self.txt(ax, W / 2, 0.80, sc.get("footer", ""), 18, self.c["muted"],
                 fade(u, 1.4 + len(items) * step, 0.7), ha="center")


# --------------------------------------------------------------------------- #
# timing, subtitles, audio
# --------------------------------------------------------------------------- #

def duration_of(sc: dict, wpm: float, floor: float, ceil: float) -> float:
    if sc.get("seconds"):
        return float(sc["seconds"])
    words = len(str(sc.get("narration", "")).split())
    body = sum(len(str(v).split()) for k, v in sc.items()
               if k in ("items", "lines", "series", "caveat", "note", "footer"))
    est = words / (wpm / 60.0) + 1.6 + 0.05 * body
    return float(min(max(est, floor), ceil))


@dataclass
class Beat:
    start: float             # offset within the scene where this line begins
    dur: float               # how long it runs
    text: str
    clip: Path | None


@dataclass
class Timing:
    dur: float               # scene length on screen
    beats: list[Beat]        # one per narration line; empty when silent


def plan_timing(scenes, voice, args) -> list[Timing]:
    """Scene length follows the narration, so reveals never drift out of sync."""
    out = []
    for i, sc in enumerate(scenes):
        lines = narration_lines(sc)
        clips = voice[i] if i < len(voice) else []
        beats, t = [], LEAD
        for b, text in enumerate(lines):
            clip = clips[b] if b < len(clips) else None
            al = audio_len(clip) if clip else None
            if al is None:                      # silent: fall back to reading pace
                al = max(1.6, len(text.split()) / (args.wpm / 60.0))
                clip = None
            beats.append(Beat(t, al, text, clip))
            t += al + args.beat_gap
        dur = max(t - args.beat_gap + TAIL,
                  float(sc.get("seconds", 0) or 0),
                  args.min_scene if not beats else 0.0)
        if not beats:
            dur = max(dur, duration_of(sc, args.wpm, args.min_scene,
                                       args.max_scene))
        out.append(Timing(dur, beats))
    return out


def audio_len(path: Path) -> float | None:
    if not have("ffprobe"):
        return None
    try:
        out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                   "-of", "csv=p=0", str(path)]).stdout.strip()
        return float(out)
    except Exception:
        return None


def srt_stamp(t: float) -> str:
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def write_srt(path: Path, scenes, timing: list[Timing]):
    out, t, n = [], 0.0, 1
    for tm in timing:
        for bt in tm.beats:
            out += [str(n),
                    f"{srt_stamp(t + bt.start)} --> {srt_stamp(t + bt.start + bt.dur)}"]
            out += wrap(bt.text, 62)[:2] + [""]
            n += 1
        t += tm.dur
    path.write_text("\n".join(out))


# --------------------------------------------------------------------------- #
# main render loop
# --------------------------------------------------------------------------- #
def render(board, out_video: Path, base: Path, pages, args, timing) -> None:
    """Draw every frame straight into an ffmpeg pipe. Video only; audio comes later."""
    scenes = board["scenes"]
    # The canvas is always 16x9 inches; resolution is dpi. Font sizes are in
    # points, so scaling figsize instead would rescale the layout, not the
    # image — small renders would come out with giant clipped text.
    dpi = args.width / 16.0
    w = int(round(16.0 * dpi)) // 2 * 2
    h = int(round(9.0 * dpi)) // 2 * 2
    if (args.height, args.width) != (h, w):
        print(f"      rendering {w}x{h} (16:9, derived from --width)")
    draw_caps = args.subs in ("render", "both")
    r = Renderer(board.get("theme", "paper"), base, args.fps, (w, h))
    fig = plt.figure(figsize=(16.0, 9.0), dpi=dpi)

    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-vcodec", "rawvideo", "-s", f"{w}x{h}", "-pix_fmt", "rgba",
         "-r", str(args.fps), "-i", "-", "-an", "-vcodec", "libx264",
         "-pix_fmt", "yuv420p", "-crf", str(args.crf), "-preset", args.preset,
         str(out_video)], stdin=subprocess.PIPE)

    total = int(sum(t.dur for t in timing) * args.fps)
    t0, done = time.time(), 0
    for i, (sc, tm) in enumerate(zip(scenes, timing)):
        d = tm.dur
        r.beats = tm.beats
        r.reader = None
        r.cap_color = None
        if sc.get("type") == "clip" and sc.get("video"):
            r.cap_color = "#FFFFFF"
            src = Path(sc["video"])
            if src.exists():
                # loop short footage rather than freezing on a still
                r.reader = ClipReader(src, args.fps,
                                      loop=clip_duration(src) < d - 0.5)
            else:
                print(f"      scene {i}: clip not found: {src}")
        for f in range(int(round(d * args.fps))):
            u = f / args.fps
            ax = r.axes(fig)
            r.scene(ax, sc, u, d, i, pages)
            if draw_caps:
                for bt in tm.beats:
                    if bt.start - 0.25 <= u <= bt.start + bt.dur + 0.30:
                        r.caption(ax, bt.text)
                        break
            veil = 1.0 - min(ease(u / 0.35) if u < 0.35 else 1.0,
                             ease((d - u) / 0.35) if (d - u) < 0.35 else 1.0)
            if veil > 0.004:
                ax.add_patch(Rectangle((0, 0), W, H, fc=r.c["bg"], ec="none",
                                       alpha=min(veil, 1.0), zorder=100))
            fig.canvas.draw()
            proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
            done += 1
            if done % 150 == 0:
                el = time.time() - t0
                frac = done / max(total, 1)
                kind = sc.get("type", "?")
                print(f"  {100 * frac:5.1f}%  scene {i + 1}/{len(scenes)} "
                      f"({kind})  frame {done}/{total}  "
                      f"elapsed {hms(el):>7}  eta {hms(el / max(frac, 1e-6) - el):>7}"
                      f"  {done / max(el, 1e-6):.1f} fps", flush=True)
        if r.reader:
            r.reader.close()
    proc.stdin.close()
    proc.wait()
    plt.close(fig)
    return w, h


def finalize(silent: Path, out: Path, voice, timing, srt: Path | None, args):
    """Mux narration onto the video and, if asked, embed the subtitle track."""
    offs, acc = [], 0.0
    for tm in timing:
        offs.append(acc); acc += tm.dur

    clips = [(offs[i] + bt.start, bt.clip)
             for i, tm in enumerate(timing) for bt in tm.beats if bt.clip]
    soft = srt is not None and args.subs in ("soft", "both") and srt.exists()

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent)]
    for _, p in clips:
        cmd += ["-i", str(p)]
    if soft:
        cmd += ["-i", str(srt)]

    maps = ["-map", "0:v"]
    if clips:
        parts, labels = [], []
        for n, (at, _) in enumerate(clips):
            ms = int(round(at * 1000))
            parts.append(f"[{n + 1}:a]adelay={ms}|{ms}[a{n}]")
            labels.append(f"[a{n}]")
        graph = ";".join(parts) + ";" + "".join(labels) + \
            f"amix=inputs={len(labels)}:dropout_transition=0:normalize=0," \
            f"apad,atrim=0:{acc:.3f}[aout]"
        cmd += ["-filter_complex", graph]
        maps += ["-map", "[aout]"]
    if soft:
        maps += ["-map", f"{len(clips) + 1}:s"]

    cmd += maps + ["-c:v", "copy"]
    if clips:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    if soft:
        cmd += ["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"]
    cmd += ["-movflags", "+faststart", str(out)]

    subprocess.run(cmd, check=True)
    silent.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Turn a paper PDF into an explainer video.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("input", nargs="?", help="paper.pdf or storyboard.json")
    ap.add_argument("-o", "--output", default=None, help="output .mp4")
    ap.add_argument("--storyboard", default=None, help="storyboard path to write/read")
    ap.add_argument("--assets", default=None, help="dir for extracted pages/figures")
    ap.add_argument("--plan-only", action="store_true", help="write storyboard, stop")
    ap.add_argument("--planner", choices=["auto", "llm", "local", "outline"],
                    default="auto")
    ap.add_argument("--model", default=None,
                    help="planner model (default: claude-sonnet-5, "
                         "or qwen3:8b for --planner local)")
    ap.add_argument("--api-base", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint for --planner local")
    ap.add_argument("--api-key-env", default="ANTHROPIC_API_KEY",
                    help="env var holding the key, if the endpoint needs one")
    ap.add_argument("--max-chars", type=int, default=120_000)
    ap.add_argument("--theme", choices=list(THEMES), default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920,
                    help="output width; height is derived to keep 16:9")
    ap.add_argument("--height", type=int, default=1080,
                    help="ignored unless it matches 16:9 for --width")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--wpm", type=float, default=155.0, help="narration pace")
    ap.add_argument("--min-scene", type=float, default=6.0)
    ap.add_argument("--max-scene", type=float, default=22.0)
    ap.add_argument("--tts", default="auto",
                    choices=["auto", "kokoro", "edge", "piper", "say",
                             "espeak", "none"],
                    help="narration engine (auto = best one installed)")
    ap.add_argument("--voice", default=None,
                    help="voice name, or the .onnx path for --tts piper")
    ap.add_argument("--tts-rate", type=float, default=1.0,
                    help="speech rate multiplier")
    ap.add_argument("--video-cmd", default=None,
                    help="shell template that generates a clip; placeholders "
                         "{prompt} {out} {seconds} {width} {height}")
    ap.add_argument("--video-seconds", type=float, default=5.0,
                    help="default length to ask the video model for")
    ap.add_argument("--clips-dir", default=None,
                    help="where generated clips are cached (default: assets/clips)")
    ap.add_argument("--tts-retries", type=int, default=3,
                    help="attempts per clip before falling back to the next engine")
    ap.add_argument("--beat-gap", type=float, default=0.45,
                    help="pause between narration lines inside a scene")
    ap.add_argument("--voice-dir", default=None,
                    help="where scene_NN.wav clips are cached (default: assets/voice)")
    ap.add_argument("--subs", default="both",
                    choices=["both", "render", "soft", "none"],
                    help="both = captions drawn in frames + selectable track")
    ap.add_argument("--no-srt", action="store_true",
                    help="skip the sidecar .srt file")
    ap.add_argument("--schema", action="store_true", help="print storyboard schema")
    ap.add_argument("--figures", action="store_true",
                    help="list available illustration primitives and presets")
    args = ap.parse_args(argv)

    if args.schema:
        print(SCHEMA_DOC)
        return 0
    if args.figures:
        try:
            from illustrations import catalogue
            print(catalogue())
        except Exception as e:
            print(f"illustrations library unavailable: {e}")
        return 0
    if not args.input:
        ap.error("give a .pdf or a storyboard .json (or use --schema)")
    if not have("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH")

    src = Path(args.input).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"no such file: {src}")

    out = Path(args.output).expanduser() if args.output else \
        src.with_suffix("").with_name(src.stem + "_explainer.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    assets = Path(args.assets) if args.assets else out.parent / f"{src.stem}_assets"

    # ---- storyboard: load, or build from the PDF
    if src.suffix.lower() == ".json":
        board = json.loads(src.read_text())
        base = src.parent
        if args.assets and assets.exists():
            base = assets
        pages = sorted(assets.glob("page-*.png")) if assets.exists() else []
        sb_path = src
    else:
        print(f"[1/3] extracting {src.name}")
        ex = extract(src, assets)
        print(f"      {ex.n_pages} pages, {len(ex.figures)} embedded figures, "
              f"{len(ex.text):,} chars of text")

        mode = args.planner
        if mode == "auto":
            mode = "llm" if os.environ.get(args.api_key_env) else "outline"
        local = mode == "local"
        model = args.model or ("qwen3:8b" if local else "claude-sonnet-5")
        note = ", ".join([f"page:{i + 1}" for i in range(min(ex.n_pages, 12))] +
                         [p.name for p in ex.figures[:12]])

        if mode in ("llm", "local"):
            where = args.api_base if local else "the Claude API"
            print(f"[2/3] planning with {model} via {where}")
            try:
                board = plan_with_llm(ex.text, note, model, args.max_chars,
                                      args.api_base if local else None,
                                      args.api_key_env)
            except Exception as e:
                print(f"      planner failed ({e}); falling back to outline")
                board = plan_outline(ex.text, ex, src.stem)
        else:
            print("[2/3] planning offline — free, but crude. Better options:")
            print("      --planner local        (Ollama etc, free)")
            print(f"      export {args.api_key_env}=...   (Claude API, ~$0.09/paper)")
            board = plan_outline(ex.text, ex, src.stem)

        base, pages = assets, ex.pages
        sb_path = Path(args.storyboard) if args.storyboard else \
            out.with_suffix(".storyboard.json")
        sb_path.write_text(json.dumps(board, indent=2))
        print(f"      storyboard -> {sb_path}  ({len(board.get('scenes', []))} scenes)")

    if args.theme:
        board["theme"] = args.theme
    if args.plan_only:
        print("stopping after planning (--plan-only). Edit the storyboard, then:")
        print(f"  python {Path(sys.argv[0]).name} {sb_path} -o {out}")
        return 0

    scenes = board.get("scenes", [])
    if not scenes:
        raise SystemExit("storyboard has no scenes")

    cdir = Path(args.clips_dir) if args.clips_dir else (base / "clips")
    if any(sc.get("type") == "clip" for sc in scenes):
        print("[3/4] video clips")
        ensure_clips(board, cdir, args)

    n_beats = sum(len(narration_lines(sc)) for sc in scenes)
    kinds = {}
    for sc in scenes:
        kinds[sc.get("type", "?")] = kinds.get(sc.get("type", "?"), 0) + 1
    print(f"plan  {len(scenes)} scenes, {n_beats} narration beats")
    print(f"      " + ", ".join(f"{v}x {k}" for k, v in sorted(kinds.items())))

    print("[3/4] narrating")
    vdir = Path(args.voice_dir) if args.voice_dir else (base / "voice")
    voice = build_voice(scenes, vdir, args)
    timing = plan_timing(scenes, voice, args)

    srt = out.with_suffix(".srt")
    write_srt(srt, scenes, timing)

    est = sum(t.dur for t in timing)
    print(f"[4/4] rendering {hms(est)} of video "
          f"({int(est * args.fps)} frames) -> {out}")
    silent = out.with_suffix(".silent.mp4")
    vw, vh = render(board, silent, base, pages, args, timing)
    finalize(silent, out, voice, timing,
             srt if args.subs in ("soft", "both") else None, args)

    tracks = ["video"]
    if any(any(c) for c in voice):
        tracks.append("audio")
    if args.subs in ("soft", "both"):
        tracks.append("subtitles")
    if args.subs in ("render", "both"):
        tracks.append("burned-in captions")
    if args.no_srt:
        srt.unlink(missing_ok=True)
    else:
        print(f"      sidecar subtitles -> {srt}")

    m, sec = divmod(sum(t.dur for t in timing), 60)
    print(f"\ndone  {int(m)}:{sec:04.1f}  {vw}x{vh}@{args.fps}  "
          f"[{', '.join(tracks)}]")
    print(f"      {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    if vdir.exists():
        n = len(list(vdir.glob('scene_*.wav')))
        print(f"      voice cache: {vdir}  ({n} clips, reused on the next run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())