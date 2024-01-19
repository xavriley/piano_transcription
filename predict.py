# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from pathlib import Path

from cog import BasePredictor, Input, Path
import librosa
from typing import Any

import os
import requests

# model repo: https://github.com/bytedance/piano_transcription
# package repo: https://github.com/qiuqiangkong/piano_transcription_inference
from piano_transcription_inference import PianoTranscription, sample_rate

class Predictor(BasePredictor):
    transcriptor: PianoTranscription

    # can't use setup() yet because we don't have the model path ready at instantiation time
    #def setup(self):
    #    self.transcriptor = PianoTranscription(
    #        device="cuda", checkpoint_path="./model.pth"
    #    )

    def predict(
        self,
        audio_input: Path = Input(description="Piano audio to transcribe"),
        model_path: str = Input(
            description="Optional URL to specify different model weights",
            default="./model.pth"),
        device: str = Input(description="Device to run inference on", default="cuda")
    ) -> Any:

        model_path = self.download_file_if_url(model_path)

        print(Path(model_path).resolve())

        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist")
            print("Using ./model.pth instead")
            model_path = Path("./model.pth")

        self.transcriptor = PianoTranscription("Regress_onset_offset_frame_velocity_CRNN",
                                                device=device,
                                               checkpoint_path=str(model_path),
                                               segment_samples=10 * sample_rate,
                                               batch_size=8)

        midi_intermediate_filename = f"{audio_input.stem}.mid"
        
        load_audio_start = os.times()[4]
        audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
        load_audio_end = os.times()[4]

        print(f"Loaded audio in {load_audio_end - load_audio_start} seconds")


        # Transcribe audio
        transcribe_start_time = os.times()[4]
        self.transcriptor.transcribe(audio, midi_intermediate_filename)
        transcribe_end_time = os.times()[4]

        print(f"Transcribed audio in {transcribe_end_time - transcribe_start_time} seconds")

        # to return a list see https://github.com/replicate/cog/blob/main/docs/python.md#returning-a-list
        return Path(midi_intermediate_filename)

    def download_file_if_url(self, url_str, save_dir="."):
        """
        Download the file at the given URL and save it in the specified directory.
        If the string is not a valid URL or the download fails, the function does nothing.

        :param url_str: The string which may be a URL.
        :param save_dir: Directory where the downloaded file will be saved (defaults to current directory).
        """
        # Check if it looks like a valid URL
        if not (url_str.startswith("http://")
                or url_str.startswith("https://")):
            return url_str

        # Get the file name from the URL or use a default one if not found
        filename = os.path.basename(url_str)
        if not filename:
            filename = "downloaded_file"
            
        if os.path.exists(os.path.join(save_dir, filename)):
            print(f"using cached {filename} from {url_str}")
            print(f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes")
            return f"./{filename}"

        response = requests.get(url_str, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file in the given directory
        with open(os.path.join(save_dir, filename), 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded {filename} from {url_str}")
        print(f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes")

        return f"./{filename}"
