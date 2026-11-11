import collections
import os
import queue
import re
import threading
import time

import numpy as np

from modules.question_detector import INTERROGATIVE

SAMPLE_RATE = 16000
MODEL_SIZE = "base.en"

# Whisper copies the punctuation style of this prompt, which gives noticeably
# better commas, full stops and question marks. It costs almost no time.
PUNCTUATION_PROMPT = (
    "Hello. Today we will talk about technology, science, and programming. "
    "What is machine learning? It is a field of AI, and it is very useful."
)

# Things Whisper invents when it hears silence or noise.
HALLUCINATIONS = {
    "you", "thank you.", "thanks for watching.", "thanks for watching!",
    "bye.", "the end.", ".", "...", "uh", "um", "hmm", "mm-hmm",
}


def _add_nvidia_dll_dirs():
    """On Windows, let Python find CUDA DLLs installed with pip (nvidia-* packages)."""
    if os.name != "nt":
        return
    try:
        import importlib.util

        spec = importlib.util.find_spec("nvidia")
        if not spec or not spec.submodule_search_locations:
            return
        for base in list(spec.submodule_search_locations):
            for sub in os.listdir(base):
                bin_dir = os.path.join(base, sub, "bin")
                if os.path.isdir(bin_dir):
                    os.add_dll_directory(bin_dir)
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # noqa: BLE001
        pass


def _cuda_available():
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def punctuate(text):
    """Light clean-up: spacing, capital letters, '.' vs '?' at sentence ends."""
    if not text:
        return ""
    text = text.replace("…", " ")
    text = re.sub(r"\.{2,}", " ", text)          # 'What is the... Table.' -> 'What is the Table.'
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.?!;:])", r"\1", text)  # no space before punctuation
    text = re.sub(r"\bi\b", "I", text)
    if not text:
        return ""

    sentences = re.split(r"(?<=[.?!])\s+", text)
    fixed = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s = s[0].upper() + s[1:]
        first = re.match(r"[A-Za-z']+", s)
        is_q = bool(first) and first.group(0).lower() in INTERROGATIVE
        if s[-1] not in ".?!":
            s += "?" if is_q else "."
        elif s[-1] == "." and is_q and len(s.split()) >= 3:
            s = s[:-1] + "?"
        fixed.append(s)
    return " ".join(fixed)


class SpeechToText:
    def __init__(self, model_size=MODEL_SIZE, prefer_gpu=True, cpu_threads=8, beam_size=1):
        self.model_size = model_size
        self.prefer_gpu = prefer_gpu and os.environ.get("PARAKEET_FORCE_CPU") != "1"
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.model = None
        self.device = None
        self.compute_type = None
        self.gpu_error = None
        self.description = "not loaded"

    def load(self):
        """Load the model. Returns a short description such as 'base.en on GPU (float16)'."""
        from faster_whisper import WhisperModel

        if self.prefer_gpu:
            _add_nvidia_dll_dirs()
            if _cuda_available():
                try:
                    model = WhisperModel(self.model_size, device="cuda", compute_type="float16")
                    self._smoke_test(model)  # missing cuBLAS/cuDNN only fails here
                    self.model, self.device, self.compute_type = model, "cuda", "float16"
                except Exception as exc:  # noqa: BLE001
                    self.gpu_error = str(exc)
                    self.model = None
            else:
                self.gpu_error = "No CUDA device visible to CTranslate2."

        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                num_workers=1,
            )
            self.device, self.compute_type = "cpu", "int8"
            self._smoke_test(self.model)

        label = "GPU" if self.device == "cuda" else "CPU"
        self.description = "%s on %s (%s)" % (self.model_size, label, self.compute_type)
        return self.description

    def _smoke_test(self, model):
        """Run the encoder once on noise. vad_filter=False so it really computes."""
        noise = (np.random.RandomState(0).randn(SAMPLE_RATE) * 0.05).astype(np.float32)
        segments, _ = model.transcribe(noise, language="en", beam_size=1, vad_filter=False)
        list(segments)

    def transcribe(self, audio):
        """audio: float32 mono 16 kHz numpy array. Returns punctuated text ('' if nothing)."""
        if self.model is None:
            raise RuntimeError("Speech model is not loaded.")
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
            without_timestamps=True,
            temperature=0.0,
            initial_prompt=PUNCTUATION_PROMPT,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text.lower() in HALLUCINATIONS:
            return ""
        return punctuate(text)


class TranscriptionWorker(threading.Thread):
    """
    Reads audio chunks (float32, 16 kHz, mono) from `audio_queue`, groups them
    into utterances, transcribes each one and calls on_text(text).

    An utterance ends after `silence_seconds` of quiet, or after
    `max_segment_seconds` of continuous sound (so loud system audio cannot
    make it wait forever).
    """

    def __init__(
        self,
        stt,
        audio_queue,
        on_text,
        on_error=None,
        sample_rate=SAMPLE_RATE,
        silence_seconds=0.6,
        max_segment_seconds=8.0,
        min_speech_seconds=0.3,
        base_threshold=0.008,
    ):
        super().__init__(daemon=True, name="TranscriptionWorker")
        self.stt = stt
        self.audio_queue = audio_queue
        self.on_text = on_text
        self.on_error = on_error or (lambda msg: None)
        self.sample_rate = sample_rate
        self.silence_seconds = silence_seconds
        self.max_segment_seconds = max_segment_seconds
        self.min_speech_seconds = min_speech_seconds
        self.base_threshold = base_threshold
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        sr = self.sample_rate
        noise = 0.004
        preroll = collections.deque(maxlen=3)   # keep 0.3 s before speech starts
        buf, buf_seconds, speech_seconds, silence, started = [], 0.0, 0.0, 0.0, False

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if chunk is None or len(chunk) == 0:
                continue

            dur = len(chunk) / float(sr)
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            is_speech = rms >= max(self.base_threshold, noise * 3.0)
            if not is_speech:
                noise = 0.97 * noise + 0.03 * rms   # learn the background level

            if is_speech:
                if not started:
                    started = True
                    buf = list(preroll)
                    buf_seconds = sum(len(c) for c in buf) / float(sr)
                    preroll.clear()
                buf.append(chunk)
                buf_seconds += dur
                speech_seconds += dur
                silence = 0.0
            elif started:
                buf.append(chunk)
                buf_seconds += dur
                silence += dur
            else:
                preroll.append(chunk)

            if started and (silence >= self.silence_seconds or buf_seconds >= self.max_segment_seconds):
                self._flush(buf, speech_seconds)
                buf, buf_seconds, speech_seconds, silence, started = [], 0.0, 0.0, 0.0, False

    def _flush(self, buf, speech_seconds):
        if speech_seconds < self.min_speech_seconds:
            return
        try:
            audio = np.concatenate(buf).astype(np.float32)
            text = self.stt.transcribe(audio)
            if text:
                self.on_text(text)
        except Exception as exc:  # noqa: BLE001  - one bad segment must not kill the worker
            self.on_error("Transcription error: %s" % exc)


if __name__ == "__main__":
    t0 = time.time()
    stt = SpeechToText()
    print("Loading model...")
    print("Loaded:", stt.load(), "in %.1fs" % (time.time() - t0))
    if stt.gpu_error:
        print("GPU not used because:", stt.gpu_error)
    noise = (np.random.RandomState(1).randn(SAMPLE_RATE * 5) * 0.05).astype(np.float32)
    t0 = time.time()
    stt.transcribe(noise)
    print("Transcribed 5 s of audio in %.2fs" % (time.time() - t0))