# Transcript Tool

A quick python script made for bulk transcribing audio files off of my audio recorder.

Each run transcribes the recordings (`.mp3`, `.wav` or `.m4a`) that don't have a note yet into Markdown notes named `T<number>.md`, for example `ZOOM0028.MP3` becomes `T28.md`. Renamed notes such as `T28 Some title.md` still count, and notes moved into a subfolder (an archive) don't.

The model is [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) (English), run locally through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) on the GPU (tested with Vulkan on Linux). The text is the model's output unchanged, with a paragraph break wherever the speaker paused for `PARAGRAPH_PAUSE_SECONDS` or longer.

# Getting Started

1. Install Poetry and ffmpeg, for example `sudo pacman -S python-poetry ffmpeg` on Arch or `brew install poetry ffmpeg` on a Mac.

2. Download the model into `models/`:

   ```sh
   mkdir -p models
   curl -L -o models/parakeet-tdt-0.6b-v2-F16.gguf \
       https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F16.gguf
   ```

3. Install the dependencies, then copy `.env.template` to `.env` and fill it in (see [Settings](#settings)):

   ```sh
   poetry install
   cp .env.template .env
   ```

4. Run the script:

   ```sh
   poetry run python src/main.py
   ```

   It copies the recordings to a temporary folder first, so the recorder can be unplugged once it says so. At startup ggml lists the GPU it found; `TRANSCRIBE_BACKEND=cpu` in the environment forces the CPU instead.

## Settings

Settings are read from `.env`:

| Key | Meaning |
| --- | --- |
| `AUDIO_IN_DIR` | Folder holding the recordings, for example the recorder's mounted folder |
| `TEXT_OUT_DIR` | Folder the notes are written to |
| `MODEL_PATH` | The model's GGUF file (a relative path is relative to this repo) |
| `PARAGRAPH_PAUSE_SECONDS` | Pause length that starts a new paragraph (default 3) |

## Options

- `--dry-run` lists what would be transcribed, and writes nothing.
- `--force N [N ...]` transcribes those recording numbers again. The new note is written beside the existing one, as `T<n> (retranscribed).md` (or `(retranscribed 2)` and so on), so no note is ever overwritten.

## Tests

```sh
poetry install --with dev
poetry run pytest
```
