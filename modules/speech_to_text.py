from faster_whisper import WhisperModel
import numpy as np

MODEL_SIZE = "base"

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(audio_path):

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    transcription = " ".join(
        segment.text.strip() for segment in segments
    )

    return transcription


def transcribe_chunk(audio_chunk):

    audio = np.squeeze(audio_chunk).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio,
        beam_size=3,
        language="en",
        vad_filter=True
    )

    transcription = " ".join(
        segment.text.strip() for segment in segments
    )

    return transcription.strip()