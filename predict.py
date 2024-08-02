# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from pathlib import Path

from cog import BasePredictor, BaseModel, Input, Path
import librosa
from typing import Any
import madmom
import numpy as np
import pretty_midi as pm
from BeatNet.BeatNet import BeatNet

import os
import requests
from time import sleep

# model repo: https://github.com/bytedance/piano_transcription
# package repo: https://github.com/qiuqiangkong/piano_transcription_inference
from piano_transcription_inference import PianoTranscription, sample_rate
import torch


class Output(BaseModel):
    midi: Path
    syncpoints: Path


class Predictor(BasePredictor):
    transcriptor: PianoTranscription

    # can't use setup() yet because we don't have the model path ready at instantiation time
    #def setup(self):
    #    self.transcriptor = PianoTranscription(
    #        device="cuda", checkpoint_path="./model.pth"
    #    )

    def add_downbeats_to_midi(self, midi_path, beats, beats_per_bar=4):
        tempo = int(round(60 / np.mean(np.diff(np.array(beats)[:, 0]))))
        print(f"Tempo calculated as {tempo} bpm")

        first_downbeat_idx = 0
        for t, b in beats:
            if b == 1.0:
                first_downbeat_idx = int(b)
                break

        downbeats = [
            t for idx, (t, b) in enumerate(beats[first_downbeat_idx:])
            if (idx % beats_per_bar) == 0
        ]

        if downbeats[0] > 0:
            downbeats = [0] + downbeats

        time_sig = beats_per_bar
        syncpoints_object = str([[i, d] for i, d in enumerate(downbeats)])

        syncpoints_json_path = Path(midi_path).with_suffix(".json")
        with open(syncpoints_json_path, 'w') as text_file:
            text_file.write(syncpoints_object)

        tempos = []
        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            t = round(60.0 / ((db2 - db1) / beats_per_bar), 3)
            tempos.append(t)

        tempo_changes = list(zip(downbeats[:-1], tempos))

        # create a new midi file with the correct resolution
        # for loading into Logic Pro X
        #
        # initial tempo is only relevant in cases where the tempo is fixed
        # e.g. played to a click track, in which case you wouldn't need to
        # copy the downbeats over
        mid = pm.PrettyMIDI(midi_path)
        out_mid = pm.PrettyMIDI(resolution=960, initial_tempo=60.0)
        for i in range(len(mid.instruments)):
            out_mid.instruments.append(pm.Instrument(program=0))
        out_mid.time_signature_changes.append(
            pm.TimeSignature(beats_per_bar, 4, 0))

        # clear out the existing downbeats
        out_mid._tick_scales = []

        # copy the downbeats from the Filosax midi
        for time, tempo in tempo_changes:
            out_mid._tick_scales.append(
                (int(out_mid.time_to_tick(time)),
                 60.0 / int(tempo * out_mid.resolution)))
            out_mid._update_tick_to_time(out_mid.get_end_time())

        # with downbeats copied over, we can now add the notes
        for i in range(len(out_mid.instruments)):
            out_mid.instruments[i].notes.extend(mid.instruments[i].notes)

        # write out the new midi file
        midi_path = Path(midi_path)
        outpath = f'{midi_path.parent / midi_path.stem}_logic.mid'
        out_mid.write(outpath)

        return Path(outpath), Path(syncpoints_json_path)

    def predict(
        self,
        audio_input: Path = Input(description="Piano audio to transcribe"),
        model_path: str = Input(
            description="Optional URL to specify different model weights",
            default="./model.pth"),
        beats_per_bar: int = Input(description="Optional beats per bar",
                                   default=4),
        file_label: str = Input(
            description="Optional label to include in output filename",
            default=""),
        device: str = Input(description="Device to run inference on",
                            default="cuda")
    ) -> Any:

        print("Running prediction")
        print(f"torch available? {torch.cuda.is_available()}")

        model_path = self.download_file_if_url(model_path)

        print(Path(model_path).resolve())

        # predict beats
        estimator = BeatNet(1,
                            mode='offline',
                            inference_model='DBN',
                            plot=[],
                            thread=False,
                            device='cuda')
        beats = estimator.process(str(audio_input))
        # beats is [[time, beat_idx], ...]
        estimator = None

        if not os.path.exists(model_path):
            print(f"Model path {model_path} does not exist")
            print("Using ./model.pth instead")
            model_path = Path("./model.pth")

        self.transcriptor = PianoTranscription(
            "Regress_onset_offset_frame_velocity_CRNN",
            device=device,
            checkpoint_path=str(model_path),
            segment_samples=10 * sample_rate,
            batch_size=8)

        midi_intermediate_filename = f"{audio_input.stem}{file_label}.mid"

        load_audio_start = os.times()[4]
        audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
        load_audio_end = os.times()[4]

        print(f"Loaded audio in {load_audio_end - load_audio_start} seconds")

        # Transcribe audio
        transcribe_start_time = os.times()[4]
        self.transcriptor.transcribe(audio, midi_intermediate_filename)
        transcribe_end_time = os.times()[4]

        print(
            f"Transcribed audio in {transcribe_end_time - transcribe_start_time} seconds"
        )

        midi_with_downbeats_path, syncpoints_path = self.add_downbeats_to_midi(
            midi_intermediate_filename, beats, beats_per_bar)

        sleep(10)
        return Output(midi=midi_with_downbeats_path,
                      syncpoints=syncpoints_path)

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
            print(
                f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes"
            )
            return f"./{filename}"

        response = requests.get(url_str, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Save the file in the given directory
        with open(os.path.join(save_dir, filename), 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Downloaded {filename} from {url_str}")
        print(
            f"size of file is {os.path.getsize(os.path.join(save_dir, filename))} bytes"
        )

        return f"./{filename}"
