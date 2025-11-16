from pywhispercpp.model import Model
from dotenv import load_dotenv
import glob
import os
import re
import warnings

load_dotenv()
AUDIO_IN_DIR = os.getenv("AUDIO_IN_DIR")
TEXT_OUT_DIR = os.getenv("TEXT_OUT_DIR")
warnings.filterwarnings("ignore")
model = Model("large-v3-turbo")

regex = r".*[\/\\][A-Z]+0*([1-9][0-9]*)[^0-9]*"
def get_number_str_from_path(file_path):
    return re.match(regex, file_path).group(1)

audio_paths = glob.glob(f"{AUDIO_IN_DIR}/*.MP3") + glob.glob(f"{AUDIO_IN_DIR}/*.mp3")
text_paths = glob.glob(f"{TEXT_OUT_DIR}/*.md")
print(f"\nStarting transcript generation.")
for audio_path in audio_paths:
    audio_file_number = get_number_str_from_path(audio_path)
    text_file_exists = False
    for text_path in text_paths:
        text_file_number = get_number_str_from_path(text_path)
        if text_file_number == audio_file_number:
            text_file_exists = True
    if not text_file_exists:
        print(f"Generating transcript for audio file {audio_path}.")

        last_segment = ""
        text_segments = []
        segments = model.transcribe(audio_path)
        for segment in segments:
            if last_segment != segment.text:
                text_segments.append(segment.text)
                text_segments.append("\n")
            last_segment = segment.text
        text = "".join(text_segments)

        with open(f"{TEXT_OUT_DIR}/T{audio_file_number}.md", "w") as file:
            file.write(text)
print("\nDone!")
