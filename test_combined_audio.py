import time
import numpy as np

from modules.audio_capture import (
    start_recording,
    get_audio_chunk,
    stop_recording
)

try:
    start_recording()

    print("\nSpeak into your microphone and play a YouTube video.")
    print("Testing for 20 seconds...\n")

    start_time = time.time()
    chunk_count = 0

    while time.time() - start_time < 20:

        chunk = get_audio_chunk()

        if chunk is None:
            continue

        if isinstance(chunk, str):
            break

        chunk_count += 1

        peak = np.max(np.abs(chunk.astype(np.int32)))
        rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2))

        print(
            f"Chunk {chunk_count}: "
            f"Peak={peak:.2f} | RMS={rms:.2f}"
        )

    stop_recording()

    print("\nTotal chunks:", chunk_count)
    print("Combined audio test completed!")

except Exception as e:

    print("\nERROR:", e)

    stop_recording()
    