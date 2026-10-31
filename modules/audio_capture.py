import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import pyaudiowpatch as pyaudio
import queue
import threading

SAMPLE_RATE = 16000
CHUNK_DURATION = 3
CHUNK_FRAMES = SAMPLE_RATE * CHUNK_DURATION

audio_queue = queue.Queue()
audio_data = []

recording_stream = None
system_stream = None
system_audio = None

recording = False
audio_lock = threading.Lock()

mic_buffer = []
system_buffer = []

mic_lock = threading.Lock()
system_lock = threading.Lock()


def start_recording():

    global recording_stream
    global system_stream, system_audio
    global recording, audio_data
    global mic_buffer, system_buffer

    with audio_lock:
        audio_data = []

    with mic_lock:
        mic_buffer = []

    with system_lock:
        system_buffer = []

    while not audio_queue.empty():
        audio_queue.get_nowait()

    recording = True

    # MICROPHONE CAPTURE

    def mic_callback(indata, frames, time, status):

        global mic_buffer

        if not recording:
            return

        if status:
            print("MIC STATUS:", status)

        samples = indata[:, 0].copy()

        with audio_lock:
            audio_data.append(samples.copy())

        with mic_lock:
            mic_buffer.extend(samples.tolist())

            while len(mic_buffer) >= CHUNK_FRAMES:

                chunk = np.array(
                    mic_buffer[:CHUNK_FRAMES],
                    dtype=np.int16
                )

                del mic_buffer[:CHUNK_FRAMES]

                audio_queue.put(chunk)

    recording_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        callback=mic_callback,
        blocksize=1024
    )

    recording_stream.start()

    # SYSTEM AUDIO CAPTURE

    system_audio = pyaudio.PyAudio()

    device = system_audio.get_default_wasapi_loopback()

    system_stream = system_audio.open(
        format=pyaudio.paInt16,
        channels=2,
        rate=int(device["defaultSampleRate"]),
        input=True,
        input_device_index=device["index"],
        frames_per_buffer=4096
    )

    threading.Thread(
        target=capture_system_audio,
        daemon=True
    ).start()

    print("Microphone and system audio recording started!")


def capture_system_audio():

    global system_buffer

    source_rate = int(system_stream._rate)

    while recording:

        try:

            data = system_stream.read(
                4096,
                exception_on_overflow=False
            )

            samples = np.frombuffer(
                data,
                dtype=np.int16
            ).reshape(-1, 2)

            mono = samples.mean(
                axis=1
            ).astype(np.int16)

            # Convert system audio to 16 kHz

            if source_rate != SAMPLE_RATE:

                target_length = int(
                    len(mono) * SAMPLE_RATE / source_rate
                )

                positions = np.linspace(
                    0,
                    len(mono) - 1,
                    target_length
                )

                mono = np.interp(
                    positions,
                    np.arange(len(mono)),
                    mono
                ).astype(np.int16)

            with system_lock:

                system_buffer.extend(mono.tolist())

                while len(system_buffer) >= CHUNK_FRAMES:

                    chunk = np.array(
                        system_buffer[:CHUNK_FRAMES],
                        dtype=np.int16
                    )

                    del system_buffer[:CHUNK_FRAMES]

                    audio_queue.put(chunk)

        except Exception as e:

            if recording:
                print("SYSTEM AUDIO ERROR:", str(e))

            break


def get_audio_chunk():

    try:
        return audio_queue.get(timeout=0.5)

    except queue.Empty:
        return None


def stop_recording():

    global recording_stream
    global system_stream, system_audio
    global recording

    if recording_stream is None:
        return None

    recording = False

    recording_stream.stop()
    recording_stream.close()

    recording_stream = None

    if system_stream is not None:

        system_stream.stop_stream()
        system_stream.close()

        system_stream = None

    if system_audio is not None:

        system_audio.terminate()
        system_audio = None

    with audio_lock:

        if audio_data:

            complete_audio = np.concatenate(
                audio_data
            )

            wav.write(
                "recording.wav",
                SAMPLE_RATE,
                complete_audio
            )

    audio_queue.put("STOP")

    print("Microphone and system audio recording stopped!")

    return "recording.wav"