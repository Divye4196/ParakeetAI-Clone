import collections
import queue
import threading
import time
from math import gcd

import numpy as np

TARGET_RATE = 16000
BLOCK_SECONDS = 0.1
TARGET_BLOCK = int(TARGET_RATE * BLOCK_SECONDS)  # 1600 samples


def put_drop_oldest(q, item):
    """Put item on q; if q is full, throw away the oldest item first."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


def _to_16k(block, rate):
    from scipy.signal import resample_poly

    if rate == TARGET_RATE:
        return block.astype(np.float32)
    g = gcd(TARGET_RATE, rate)
    out = resample_poly(block, TARGET_RATE // g, rate // g)
    return out.astype(np.float32)


class _Source:
    """Collects native-rate mono audio and exposes 100 ms blocks at 16 kHz."""

    def __init__(self, name):
        self.name = name
        self.rate = TARGET_RATE
        self._native = np.zeros(0, dtype=np.float32)
        self.blocks = collections.deque(maxlen=50)  # deque appends/pops are thread-safe

    def feed(self, mono, rate):
        self.rate = rate
        self._native = np.concatenate([self._native, mono])
        need = int(rate * BLOCK_SECONDS)
        while len(self._native) >= need:
            part, self._native = self._native[:need], self._native[need:]
            block = _to_16k(part, rate)
            if len(block) != TARGET_BLOCK:  # rounding safety
                block = np.resize(block, TARGET_BLOCK)
            self.blocks.append(block)

    def pop(self):
        # If a source got far ahead, drop old blocks so it stays in sync.
        while len(self.blocks) > 3:
            self.blocks.popleft()
        try:
            return self.blocks.popleft()
        except IndexError:
            return None

    def close(self):
        pass


class _MicSource(_Source):
    def __init__(self):
        super().__init__("microphone")
        import sounddevice as sd

        info = sd.query_devices(kind="input")
        rate = int(info["default_samplerate"])
        self._stream = sd.InputStream(
            channels=1,
            samplerate=rate,
            dtype="float32",
            callback=self._callback,
        )
        self._rate = rate
        self._stream.start()

    def _callback(self, indata, frames, time_info, status):
        try:
            self.feed(indata[:, 0].copy(), self._rate)
        except Exception:  # noqa: BLE001  - never raise inside an audio callback
            pass

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass


class _LoopbackSource(_Source):
    def __init__(self):
        super().__init__("system audio")
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        try:
            wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not speakers.get("isLoopbackDevice"):
                found = None
                for dev in self._pa.get_loopback_device_info_generator():
                    if speakers["name"] in dev["name"]:
                        found = dev
                        break
                if found is None:
                    raise RuntimeError("No loopback device found for '%s'" % speakers["name"])
                speakers = found
            self._channels = int(speakers["maxInputChannels"])
            self._rate = int(speakers["defaultSampleRate"])
            self._pyaudio = pyaudio
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._channels,
                rate=self._rate,
                input=True,
                input_device_index=speakers["index"],
                frames_per_buffer=int(self._rate * 0.05),
                stream_callback=self._callback,
            )
        except Exception:
            self._pa.terminate()
            raise

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            data = np.frombuffer(in_data, dtype=np.float32)
            if self._channels > 1:
                data = data.reshape(-1, self._channels).mean(axis=1)
            self.feed(data.copy(), self._rate)
        except Exception:  # noqa: BLE001
            pass
        return (None, self._pyaudio.paContinue)

    def close(self):
        try:
            self._stream.stop_stream()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._pa.terminate()
        except Exception:  # noqa: BLE001
            pass


class AudioCapture:
    def __init__(self, use_mic=True, use_system=True, queue_size=100):
        self.use_mic = use_mic
        self.use_system = use_system
        self.audio_queue = queue.Queue(maxsize=queue_size)
        self._sources = []
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Open the sources and start mixing. Returns a list of warning strings."""
        warnings = []
        self._stop_event.clear()
        self._sources = []
        if self.use_mic:
            try:
                self._sources.append(_MicSource())
            except Exception as exc:  # noqa: BLE001
                warnings.append("Microphone unavailable: %s" % exc)
        if self.use_system:
            try:
                self._sources.append(_LoopbackSource())
            except Exception as exc:  # noqa: BLE001
                warnings.append("System audio unavailable: %s" % exc)
        if not self._sources:
            raise RuntimeError("No audio source could be opened. " + " ".join(warnings))
        self._thread = threading.Thread(target=self._mix_loop, daemon=True, name="AudioMixer")
        self._thread.start()
        return warnings

    def _mix_loop(self):
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            mixed = np.zeros(TARGET_BLOCK, dtype=np.float32)
            for src in self._sources:
                block = src.pop()
                if block is not None:
                    mixed += block
            np.clip(mixed, -1.0, 1.0, out=mixed)
            put_drop_oldest(self.audio_queue, mixed)
            next_tick += BLOCK_SECONDS
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()  # we fell behind; do not try to catch up

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        for src in self._sources:
            src.close()
        self._sources = []
        # empty the queue so the next recording starts clean
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break