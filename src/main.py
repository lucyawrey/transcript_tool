import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from array import array
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
NUMBER_PATTERN = re.compile(r"^[A-Za-z]+0*(\d+)")
REPO_DIR = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
TARGET_PEAK_DB = -1.0
MAX_PIECE_SECONDS = 20 * 60
UNTITLED_PATTERN = re.compile(r"^T(\d+)\.md$")
FORBIDDEN_IN_NAMES = re.compile(r'[\\/:*?"<>|#^\[\]\x00-\x1f\x7f]')
TITLE_WORDS = 1500
MIN_WORDS_TO_TITLE = 5
MAX_TITLE_CHARS = 60
TITLE_PROMPT = ("You name voice-note transcripts. Reply with a short title of 2 to 6 words that says what the "
                "recording is about. Reply with the title only: no quotes, no dates, no ending punctuation.")


class ConfigError(Exception):
    pass


@dataclass
class Config:
    audio_dir: Path
    notes_dir: Path
    model_path: Path
    pause_seconds: float
    title_model_path: Path | None = None


@dataclass
class Job:
    number: int
    recording: Path
    note: Path


def load_config():
    env = os.environ
    def required_dir(key):
        value = env.get(key)
        if not value:
            raise ConfigError(f"{key} is not set in .env")
        path = Path(value).expanduser()
        if not path.is_dir():
            raise ConfigError(f"{key} is not a folder: {path}")
        return path

    audio_dir = required_dir("AUDIO_IN_DIR")
    notes_dir = required_dir("TEXT_OUT_DIR")
    if not os.access(notes_dir, os.W_OK):
        raise ConfigError(f"TEXT_OUT_DIR is not writable: {notes_dir}")
    model = env.get("MODEL_PATH")
    if not model:
        raise ConfigError("MODEL_PATH is not set in .env")
    try:
        pause = float(env.get("PARAGRAPH_PAUSE_SECONDS") or 3)
    except ValueError:
        raise ConfigError("PARAGRAPH_PAUSE_SECONDS must be a number of seconds")
    if shutil.which("ffmpeg") is None:
        raise ConfigError("ffmpeg is not installed or not on the PATH")
    title_model = env.get("TITLE_MODEL_PATH")
    title_model_path = REPO_DIR / Path(title_model).expanduser() if title_model else None
    if title_model_path and not title_model_path.is_file():
        raise ConfigError(f"TITLE_MODEL_PATH is not a file: {title_model_path}")
    return Config(audio_dir, notes_dir, REPO_DIR / Path(model).expanduser(), pause, title_model_path)


def file_number(name):
    match = NUMBER_PATTERN.match(name)
    return int(match.group(1)) if match else None


def find_recordings(audio_dir):
    recordings, warnings = {}, []
    for path in sorted(audio_dir.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        number = file_number(path.name)
        if number is None:
            warnings.append(f"Skipping {path.name}: no number in its name")
        elif number in recordings:
            warnings.append(f"Skipping {path.name}: {recordings[number].name} already has number {number}")
        else:
            recordings[number] = path
    return recordings, warnings


def existing_note_numbers(notes_dir):
    numbers = set()
    for path in notes_dir.glob("*.md"):
        number = file_number(path.name)
        if number is not None:
            numbers.add(number)
    return numbers


def new_note_path(notes_dir, number, retranscribe):
    if not retranscribe:
        return notes_dir / f"T{number}.md"
    path, copy = notes_dir / f"T{number} (retranscribed).md", 2
    while path.exists():
        path, copy = notes_dir / f"T{number} (retranscribed {copy}).md", copy + 1
    return path


def plan_jobs(recordings, existing, notes_dir, force):
    missing = sorted(set(force) - set(recordings))
    if missing:
        raise ConfigError(f"--force: no recording numbered {', '.join(map(str, missing))}")
    jobs = []
    for number, recording in sorted(recordings.items()):
        if number in force or number not in existing:
            jobs.append(Job(number, recording, new_note_path(notes_dir, number, number in existing)))
    return jobs


def build_paragraphs(words, pause_seconds):
    paragraphs, current, last_end = [], [], None
    for start, end, text in words:
        if current and (start - last_end) / 1000 >= pause_seconds:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text.strip())
        last_end = end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs) + "\n"


def same_words(a, b):
    return a.split() == b.split()


def write_note_atomically(path, text):
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def decode(recording):
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-i", str(recording), "-ac", "1", "-ar", str(SAMPLE_RATE)]
    stats = subprocess.run([*base, "-af", "volumedetect", "-f", "null", "-"],
                           capture_output=True, text=True, check=True).stderr
    peak = re.search(r"max_volume: (-?[\d.]+) dB", stats)
    gain = TARGET_PEAK_DB - float(peak.group(1)) if peak else 0.0
    raw = subprocess.run([*base, "-af", f"volume={gain}dB", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    pcm = array("f")
    pcm.frombytes(raw)
    return pcm


def split_points(pcm):
    max_samples = MAX_PIECE_SECONDS * SAMPLE_RATE
    def cut(start, end):
        if end - start <= max_samples:
            return []
        second = SAMPLE_RATE
        lo, hi = start + (end - start) // 3, start + 2 * (end - start) // 3
        quietest = min(range(lo, hi - second, second // 4),
                       key=lambda i: sum(x * x for x in pcm[i:i + second]))
        point = quietest + second // 2
        return cut(start, point) + [point] + cut(point, end)
    return cut(0, len(pcm))


def pause_cuts(words, pause_seconds):
    return [(a_end + b_start) // 2 * SAMPLE_RATE // 1000
            for (_, a_end, _), (b_start, _, _) in zip(words, words[1:])
            if (b_start - a_end) / 1000 >= pause_seconds]


def size_limited(pcm, bounds):
    limited = []
    for start, end in zip(bounds, bounds[1:]):
        limited += [start, *(start + point for point in split_points(pcm[start:end]))]
    return limited + [bounds[-1]]


def run_pieces(session, pcm, bounds):
    words, texts = [], []
    for start, end in zip(bounds, bounds[1:]):
        result = session.run(pcm[start:end], language="en", timestamps="word")
        offset = start * 1000 // SAMPLE_RATE
        words += [(w.t0_ms + offset, w.t1_ms + offset, w.text) for w in result.words]
        texts.append(result.text)
    return words, " ".join(texts)


def transcribe(session, pcm, pause_seconds):
    first_pass, _ = run_pieces(session, pcm, size_limited(pcm, [0, len(pcm)]))
    bounds = [0, *pause_cuts(first_pass, pause_seconds), len(pcm)]
    return run_pieces(session, pcm, size_limited(pcm, bounds))


def untitled_notes(note_names, recordings):
    numbered = [name for name in note_names if file_number(name) is not None]
    counts = {}
    for name in numbered:
        counts[file_number(name)] = counts.get(file_number(name), 0) + 1
    untitled = []
    for name in sorted(numbered, key=file_number):
        match = UNTITLED_PATTERN.match(name)
        if match and int(match.group(1)) in recordings and counts[int(match.group(1))] == 1:
            untitled.append(name)
    return untitled


def ordinal(day):
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def recording_date(recording):
    date = datetime.fromtimestamp(recording.stat().st_mtime)
    return f"{date:%B} {ordinal(date.day)} {date.year}"


def clean_title(raw):
    lines = raw.strip().splitlines()
    title = FORBIDDEN_IN_NAMES.sub(" ", lines[0] if lines else "")
    title = " ".join(title.split()).strip(" .'\u201c\u201d\u2018\u2019")
    if len(title) > MAX_TITLE_CHARS:
        title = title[:MAX_TITLE_CHARS + 1].rsplit(" ", 1)[0].rstrip(" .")
    return title


def title_notes(notes_dir, names, recordings, make_title):
    renamed = 0
    for name in names:
        path = notes_dir / name
        number = file_number(name)
        try:
            words = path.read_text(encoding="utf-8").split()
            if len(words) < MIN_WORDS_TO_TITLE:
                print(f"Left {name} untitled: too short to title")
                continue
            title = clean_title(make_title(" ".join(words[:TITLE_WORDS])))
            if not title:
                print(f"Left {name} untitled: the model gave no usable title")
                continue
            target = notes_dir / f"T{number} {recording_date(recordings[number])} {title} (auto).md"
            if target.exists():
                print(f"Left {name} untitled: {target.name} already exists")
                continue
            os.rename(path, target)
            renamed += 1
            print(f"Titled {name} -> {target.name}")
        except Exception as error:
            print(f"Left {name} untitled: {error}")
    return renamed


def load_titler(model_path):
    from llama_cpp import Llama
    llm = Llama(model_path=str(model_path), n_gpu_layers=-1, n_ctx=4096, verbose=False)

    def make_title(text):
        reply = llm.create_chat_completion(
            messages=[{"role": "system", "content": TITLE_PROMPT}, {"role": "user", "content": text}],
            temperature=0, max_tokens=24)
        return reply["choices"][0]["message"]["content"]
    return make_title


def title_untitled_notes(config, recordings):
    names = untitled_notes([p.name for p in config.notes_dir.glob("*.md")], recordings)
    if not names:
        return
    try:
        make_title = load_titler(config.title_model_path)
    except Exception as error:
        print(f"Titling skipped: the title model failed to load: {error}")
        return
    renamed = title_notes(config.notes_dir, names, recordings, make_title)
    print(f"Titled {renamed} of {len(names)} untitled note(s).")


def print_library_problem(level, message):
    if level in (2, 3):
        print(f"  {message.strip()}", file=sys.stderr)


def transcribe_jobs(config, jobs):
    failures = []
    with tempfile.TemporaryDirectory() as temp_dir:
        copies = [Path(shutil.copy2(job.recording, temp_dir)) for job in jobs]
        print(f"Copied {len(jobs)} recording(s). The recorder can be unplugged.")

        import transcribe_cpp
        transcribe_cpp.set_log_callback(print_library_problem)
        with transcribe_cpp.Model(config.model_path) as model, model.session() as session:
            for job, copy in zip(jobs, copies):
                print(f"Transcribing {job.recording.name} -> {job.note.name}", flush=True)
                try:
                    words, full_text = transcribe(session, decode(copy), config.pause_seconds)
                    note = build_paragraphs(words, config.pause_seconds)
                    if not same_words(note, full_text):
                        print("  Warning: word timings don't match the text; writing it without paragraphs")
                        note = full_text.strip() + "\n"
                    write_note_atomically(job.note, note)
                except Exception as error:
                    failures.append(f"{job.recording.name}: {error}")
                    print(f"  Failed: {error}")

    print(f"\nDone: {len(jobs) - len(failures)} transcribed, {len(failures)} failed.")
    for failure in failures:
        print(f"  {failure}")
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe new voice recordings into Markdown notes.")
    parser.add_argument("--dry-run", action="store_true", help="list what would be transcribed, then stop")
    parser.add_argument("--force", type=int, nargs="+", default=[], metavar="N",
                        help="transcribe these recording numbers again, beside their existing notes")
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        config = load_config()
        recordings, warnings = find_recordings(config.audio_dir)
        jobs = plan_jobs(recordings, existing_note_numbers(config.notes_dir), config.notes_dir, args.force)
        if jobs and not args.dry_run and not config.model_path.is_file():
            raise ConfigError(f"MODEL_PATH is not a file: {config.model_path}")
    except ConfigError as error:
        sys.exit(f"Error: {error}")
    for warning in warnings:
        print(f"Warning: {warning}")

    if args.dry_run:
        if not jobs:
            print("No new recordings to transcribe.")
        for job in jobs:
            print(f"Would transcribe {job.recording.name} -> {job.note.name}")
        if config.title_model_path:
            names = [p.name for p in config.notes_dir.glob("*.md")] + [job.note.name for job in jobs]
            for name in untitled_notes(names, recordings):
                print(f"Would title {name}")
        return 0

    failures = transcribe_jobs(config, jobs) if jobs else []
    if not jobs:
        print("No new recordings to transcribe.")
    if config.title_model_path:
        title_untitled_notes(config, recordings)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
