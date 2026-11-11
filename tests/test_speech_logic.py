import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.audio_capture import AudioCapture, TARGET_BLOCK, _Source, put_drop_oldest  # noqa: E402
from modules.speech_to_text import TranscriptionWorker, punctuate  # noqa: E402

failures = 0


def check(name, cond):
    global failures
    print(("PASS " if cond else "FAIL ") + name)
    failures += not cond


# --- punctuation
check("adds question mark", punctuate("what is machine learning") == "What is machine learning?")
check("fixes ellipsis + period", punctuate("What is the... Periodic Table.") == "What is the Periodic Table?")
check("statement gets full stop", punctuate("python is a language") == "Python is a language.")
check("two sentences", punctuate("hello there. how are you") == "Hello there. How are you?")

# --- drop-oldest queue never blocks
q = queue.Queue(maxsize=3)
t0 = time.time()
for i in range(10):
    put_drop_oldest(q, i)
check("drop-oldest keeps newest 3", list(q.queue) == [7, 8, 9] and time.time() - t0 < 0.5)

# --- resampling gives exactly 1600 samples per 100 ms for common rates
for rate in (44100, 48000, 16000):
    src = _Source("t")
    src.feed(np.random.randn(rate).astype(np.float32) * 0.1, rate)  # 1 second
    ok = len(src.blocks) == 10 and all(len(b) == TARGET_BLOCK for b in src.blocks)
    check("resample %d Hz -> 10 blocks of %d" % (rate, TARGET_BLOCK), ok)


# --- worker with a fake model: two utterances, one transcription crash
class FakeSTT:
    def __init__(self):
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return "segment %d (%.1fs)" % (self.calls, len(audio) / 16000.0)


def tone(sec):
    n = int(16000 * sec)
    return (0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / 16000)).astype(np.float32)


stt, aq, texts, errors = FakeSTT(), queue.Queue(maxsize=100), [], []
worker = TranscriptionWorker(stt, aq, texts.append, errors.append, silence_seconds=0.5)
worker.start()


def feed(samples):
    for i in range(0, len(samples), TARGET_BLOCK):
        aq.put(samples[i:i + TARGET_BLOCK])


quiet = np.zeros(16000, dtype=np.float32)
feed(quiet); feed(tone(1.0)); feed(quiet)      # utterance 1 -> fake crash
feed(tone(1.0)); feed(quiet)                   # utterance 2 -> text
feed(tone(10.0)); feed(quiet)                  # long continuous sound -> split by max segment
time.sleep(1.0)
worker.stop()
worker.join(2)
check("worker survived a transcription error", len(errors) == 1 and len(texts) >= 2)
check("long continuous audio is split (>= 3 texts)", len(texts) >= 3)
check("worker thread stopped", not worker.is_alive())

# --- capture mixer works with a fake source and never blocks
cap = AudioCapture(use_mic=False, use_system=False, queue_size=5)
fake = _Source("fake")
fake.feed(np.full(16000, 0.25, dtype=np.float32), 16000)
cap._sources = [fake]
cap._stop_event.clear()
cap._thread = threading.Thread(target=cap._mix_loop, daemon=True)
cap._thread.start()
time.sleep(1.0)  # queue (size 5) overflows; the mixer must keep running
check("mixer keeps running with a full queue", cap._thread.is_alive() and cap.audio_queue.qsize() == 5)
cap.stop()
check("stop() empties the queue", cap.audio_queue.qsize() == 0)

print("ALL PASSED" if not failures else "%d FAILED" % failures)
sys.exit(1 if failures else 0)