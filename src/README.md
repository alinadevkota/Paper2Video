# src/ — the Paper2Video tool

    paper2video.py          CLI: PDF extraction, planning, narration, timing, rendering
    make_storyboard.py      PDF -> storyboard draft with the Claude API, validated and repaired
    illustrations.py        helpers, the common primitives, the preset dictionary
    illustrations_extra.py  the long tail of primitives (optional; safe to delete)
    storyboard2pptx.py      storyboard -> PPTX (falls back to pictures without pptx_native.py)
    requirements.txt        Python dependencies; system tools listed at the bottom

Not included here: `pptx_native.py` (native PowerPoint emitters), `gen_clip.py` and
`vidgen.py` (generative-footage backends). All three are optional and the code runs
without them.

## Setup

    pip install -r src/requirements.txt
    sudo apt install ffmpeg poppler-utils espeak-ng      # or brew install ...

## Run from the repository root

    python src/paper2video.py --figures                             # list the 100 primitives and 181 presets
    python src/paper2video.py storyboards/transformer.json -o videos/transformer.mp4
    python src/make_storyboard.py paper.pdf -o storyboards/name.json   # Claude drafts, validates, repairs
    python src/make_storyboard.py paper.pdf --dry-run                   # print the prompt, no API call
    python src/paper2video.py paper.pdf --plan-only --storyboard storyboards/name.json   # simpler planner
    python src/storyboard2pptx.py storyboards/transformer.json --per-beat

`make_storyboard.py` sends the paper text plus the live figure catalogue to Claude
(`claude-sonnet-5` by default), validates the reply against the schema and catalogue,
and sends any problems back for up to `--repair` rounds. It uses the `anthropic` SDK
when installed and falls back to plain HTTP otherwise. Write the result into
`storyboards/` with `-o`; the default name is `<pdf stem>_storyboard.json`.

Planning inside `paper2video.py` uses the Claude API when `ANTHROPIC_API_KEY` is set, any OpenAI-compatible
endpoint with `--planner local --api-base ...`, or a regex outline with no key at all.
Keys are read from the environment only; never write one into a file in this repo.

Render caches live in `<name>_assets/` next to the output (extracted pages, `voice/`,
`clips/`). That folder, `.srt` files and `.pptx` output are ignored by git; only finished
videos in `videos/` are committed.
