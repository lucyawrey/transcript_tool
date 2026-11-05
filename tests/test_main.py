import os
from array import array
from types import SimpleNamespace

import pytest

import main


def touch(folder, *names):
    for name in names:
        (folder / name).write_text("")


@pytest.mark.parametrize("name, number", [
    ("ZOOM0028.MP3", 28),
    ("ZOOM0000.MP3", 0),
    ("zoom0100.wav", 100),
    ("T28.md", 28),
    ("T28 November 15th 2025 Transcription Test.md", 28),
    ("._ZOOM0018.MP3", None),
    ("0028.MP3", None),
    ("ZOOM.MP3", None),
])
def test_file_number(name, number):
    assert main.file_number(name) == number


def test_find_recordings_skips_metadata_other_files_and_duplicates(tmp_path):
    touch(tmp_path, "ZOOM0001.MP3", "ZOOM0002.wav", "ZOOM0003.m4a", "._ZOOM0001.MP3",
          "notes.txt", "README.mp3", "REC0002.MP3")
    recordings, warnings = main.find_recordings(tmp_path)
    assert {n: p.name for n, p in recordings.items()} == {
        1: "ZOOM0001.MP3", 2: "REC0002.MP3", 3: "ZOOM0003.m4a"}
    assert warnings == [
        "Skipping README.mp3: no number in its name",
        "Skipping ZOOM0002.wav: REC0002.MP3 already has number 2",
    ]


def test_existing_note_numbers_ignores_archive(tmp_path):
    touch(tmp_path, "T10.md", "T28 A renamed note.md", "Ideas.md")
    (tmp_path / "Archive").mkdir()
    touch(tmp_path / "Archive", "T11.md")
    assert main.existing_note_numbers(tmp_path) == {10, 28}


def test_new_note_path_never_overwrites(tmp_path):
    assert main.new_note_path(tmp_path, 5, retranscribe=False).name == "T5.md"
    assert main.new_note_path(tmp_path, 5, retranscribe=True).name == "T5 (retranscribed).md"
    touch(tmp_path, "T5 (retranscribed).md", "T5 (retranscribed 2).md")
    assert main.new_note_path(tmp_path, 5, retranscribe=True).name == "T5 (retranscribed 3).md"


def test_plan_jobs_new_and_forced(tmp_path):
    recordings = {1: tmp_path / "ZOOM0001.MP3", 2: tmp_path / "ZOOM0002.MP3", 3: tmp_path / "ZOOM0003.MP3"}
    jobs = main.plan_jobs(recordings, existing={1, 2}, notes_dir=tmp_path, force=[2])
    assert [(j.number, j.note.name) for j in jobs] == [(2, "T2 (retranscribed).md"), (3, "T3.md")]


def test_plan_jobs_force_unknown_number(tmp_path):
    with pytest.raises(main.ConfigError, match="no recording numbered 9"):
        main.plan_jobs({}, existing=set(), notes_dir=tmp_path, force=[9])


def test_build_paragraphs_breaks_only_at_long_pauses():
    words = [(0, 400, "Hello,"), (500, 900, "hello."), (4000, 4300, "This"), (4300, 4600, "is"),
             (7500, 7900, "it.")]
    assert main.build_paragraphs(words, 3) == "Hello, hello.\n\nThis is it.\n"
    assert main.build_paragraphs(words, 2.9) == "Hello, hello.\n\nThis is\n\nit.\n"
    assert main.build_paragraphs([], 3) == "\n"


def test_build_paragraphs_keeps_every_word():
    words = [(i * 100, i * 100 + 50, w) for i, w in enumerate("I... I don't, know.".split())]
    note = main.build_paragraphs(words, 3)
    assert main.same_words(note, "I... I don't, know.")


def test_split_points_short_audio_is_not_split():
    assert main.split_points(array("f", [0.1] * 100), max_samples=100) == []


def test_split_points_cuts_at_the_quiet_spot():
    rate = main.SAMPLE_RATE
    pcm = array("f", [0.5] * (12 * rate))
    pcm[7 * rate:8 * rate] = array("f", [0.0] * rate)
    points = main.split_points(pcm, max_samples=10 * rate)
    assert len(points) == 1 and 7 * rate <= points[0] <= 8 * rate


def test_write_note_atomically(tmp_path):
    note = tmp_path / "T1.md"
    main.write_note_atomically(note, "text\n")
    assert note.read_text() == "text\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["T1.md"]


def config_env(tmp_path, **overrides):
    (tmp_path / "in").mkdir(exist_ok=True)
    (tmp_path / "out").mkdir(exist_ok=True)
    env = {"AUDIO_IN_DIR": str(tmp_path / "in"), "TEXT_OUT_DIR": str(tmp_path / "out"),
           "MODEL_PATH": str(tmp_path / "model.gguf")}
    env.update(overrides)
    return {k: v for k, v in env.items() if v is not None}


def test_load_config_defaults(tmp_path):
    config = main.load_config(config_env(tmp_path))
    assert config.pause_seconds == 3


def test_load_config_relative_model_path_is_inside_the_repo(tmp_path):
    config = main.load_config(config_env(tmp_path, MODEL_PATH="models/m.gguf"))
    assert config.model_path == main.REPO_DIR / "models/m.gguf"


@pytest.mark.parametrize("overrides, message", [
    ({"AUDIO_IN_DIR": None}, "AUDIO_IN_DIR is not set"),
    ({"AUDIO_IN_DIR": "/no/such/folder"}, "AUDIO_IN_DIR is not a folder"),
    ({"TEXT_OUT_DIR": ""}, "TEXT_OUT_DIR is not set"),
    ({"MODEL_PATH": None}, "MODEL_PATH is not set"),
    ({"PARAGRAPH_PAUSE_SECONDS": "long"}, "must be a number"),
])
def test_load_config_errors(tmp_path, overrides, message):
    with pytest.raises(main.ConfigError, match=message):
        main.load_config(config_env(tmp_path, **overrides))


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_load_config_unwritable_notes_dir(tmp_path):
    env = config_env(tmp_path)
    os.chmod(env["TEXT_OUT_DIR"], 0o500)
    try:
        with pytest.raises(main.ConfigError, match="not writable"):
            main.load_config(env)
    finally:
        os.chmod(env["TEXT_OUT_DIR"], 0o700)


def test_load_config_needs_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(main.shutil, "which", lambda name: None)
    with pytest.raises(main.ConfigError, match="ffmpeg"):
        main.load_config(config_env(tmp_path))


def test_pause_cuts_at_the_middle_of_long_pauses():
    words = [(0, 1000, "One."), (1500, 2000, "two."), (6000, 6500, "Three."), (7000, 7500, "four.")]
    assert main.pause_cuts(words, 3) == [4000 * main.SAMPLE_RATE // 1000]
    assert main.pause_cuts(words, 5) == []
    assert main.pause_cuts([], 3) == []


def test_size_limited_splits_only_long_pieces(monkeypatch):
    rate = main.SAMPLE_RATE
    pcm = array("f", [0.5] * (30 * rate))
    pcm[22 * rate:23 * rate] = array("f", [0.0] * rate)
    monkeypatch.setattr(main, "MAX_PIECE_SECONDS", 10)
    bounds = main.size_limited(pcm, [0, 5 * rate, 30 * rate])
    assert bounds[:2] == [0, 5 * rate] and bounds[-1] == 30 * rate
    assert len(bounds) > 3 and all(b - a <= 10 * rate for a, b in zip(bounds, bounds[1:]))


class FakeSession:
    def __init__(self):
        self.lengths = []

    def run(self, pcm, language, timestamps):
        self.lengths.append(len(pcm))
        return SimpleNamespace(text="word.", words=[SimpleNamespace(t0_ms=100, t1_ms=200, text="word.")])


def test_run_pieces_offsets_word_times_by_piece_start():
    rate = main.SAMPLE_RATE
    session = FakeSession()
    words, text = main.run_pieces(session, array("f", [0.0] * (10 * rate)), [0, 4 * rate, 10 * rate])
    assert session.lengths == [4 * rate, 6 * rate]
    assert words == [(100, 200, "word."), (4100, 4200, "word.")]
    assert text == "word. word."
