import pyaudiowpatch as pyaudio
import numpy as np
import threading
import queue

CHUNK_SIZE = 4096
SAMPLE_RATE = 16000

system_audio_queue = queue.Queue()

audio = None
stream = None
recording = False
audio_thread = None


def capture_system_audio():

    global audio, stream, recording

    audio = pyaudio.PyAudio()

    try:
        device = audio.get_default_wasapi_loopback()

        stream = audio.open(
            format=pyaudio.paInt16,
            channels=2,
            rate=int(device["defaultSampleRate"]),
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK_SIZE
        )

        print("System audio capture started!")

        while recording:

            data = stream.read(
                CHUNK_SIZE,
                exception_on_overflow=False
            )

            samples = np.frombuffer(
                data,
                dtype=np.int16
            )

            samples = samples.reshape(-1, 2)

            mono = samples.mean(
                axis=1
            ).astype(np.int16)

            system_audio_queue.put(mono)

    except Exception as e:

        print("SYSTEM AUDIO ERROR:", str(e))

    finally:

        if stream is not None:
            stream.stop_stream()
            stream.close()

        audio.terminate()

        print("System audio capture stopped!")


def start_system_audio():

    global recording, audio_thread

    if recording:
        return

    recording = True

    audio_thread = threading.Thread(
        target=capture_system_audio,
        daemon=True
    )

    audio_thread.start()


def get_system_audio_chunk():

    try:
        return system_audio_queue.get(timeout=0.5)

    except queue.Empty:
        return None


def stop_system_audio():

    global recording

    recording = False

    if audio_thread is not None:
        audio_thread.join(timeout=3)

    while not system_audio_queue.empty():
        system_audio_queue.get_nowait()