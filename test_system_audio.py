import pyaudiowpatch as pyaudio
import numpy as np

audio = pyaudio.PyAudio()

try:
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=48000,
        input=True,
        input_device_index=10,
        frames_per_buffer=4096
    )

    print("Capturing system audio...")

    for i in range(10):
        data = stream.read(4096, exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16)

        peak = np.max(np.abs(samples))
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))

        print(f"Peak: {peak:.2f} | RMS: {rms:.2f}")

    stream.stop_stream()
    stream.close()

except Exception as e:
    print("ERROR:", e)

finally:
    audio.terminate()