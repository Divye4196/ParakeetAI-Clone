import time

from modules.system_audio import (
    start_system_audio,
    get_system_audio_chunk,
    stop_system_audio
)

try:
    start_system_audio()

    print("Play a YouTube video now...")
    time.sleep(2)

    for i in range(5):

        chunk = get_system_audio_chunk()

        if chunk is not None:
            print(
                f"Chunk {i + 1}: "
                f"Samples={len(chunk)}, "
                f"Peak={max(abs(chunk))}"
            )

    stop_system_audio()

    print("System audio module test completed!")

except Exception as e:

    print("ERROR:", e)

    stop_system_audio()
    