import shutil
import sys
import urllib.request
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_DIR / "models"
MODELS = {
    "parakeet-tdt-0.6b-v2-F16.gguf":
        "https://huggingface.co/handy-computer/parakeet-tdt-0.6b-v2-gguf/resolve/main/parakeet-tdt-0.6b-v2-F16.gguf",
    "qwen3-4b-instruct-2507-Q5_K_M.gguf":
        "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q5_K_M.gguf",
}


def show_progress(blocks, block_size, total):
    if total > 0:
        print(f"\r  {min(blocks * block_size / total, 1):.0%}", end="", flush=True)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODELS.items():
        target = MODELS_DIR / name
        if target.exists():
            print(f"{name} is already downloaded")
            continue
        print(f"Downloading {name}")
        partial = target.with_name(name + ".part")
        urllib.request.urlretrieve(url, partial, show_progress)
        partial.rename(target)
        print()
    env = REPO_DIR / ".env"
    if env.exists():
        print(".env already exists")
    else:
        shutil.copy(REPO_DIR / ".env.template", env)
        print("Created .env from .env.template: set AUDIO_IN_DIR and TEXT_OUT_DIR in it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
