# Transcript Tool

A python script for bulk transcribing audio files off of my audio recorder.

Recordings are transcribed locally. The current model is [Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). Optionally, [Qwen3 4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) can be enable to automatically name untitled notes. LLMs never clean up the text as the intention of this project is raw transcriptions of real human speech.

## Setup

Install Poetry and ffmpeg, then:

```sh
poetry install
poetry run python src/setup.py
```

In `.env`, set `AUDIO_IN_DIR` (the recorder's folder) and `TEXT_OUT_DIR` (the notes folder). Leave `TITLE_MODEL_PATH` empty to skip titles.

## Run

```sh
poetry run python src/main.py              # transcribe new recordings, then title untitled notes
poetry run python src/main.py --dry-run    # list what it would do
poetry run python src/main.py --force 12   # transcribe recording 12 again, as "T12 (retranscribed).md"
```
