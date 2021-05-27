import numpy as np
import simpleaudio as sa

def sound(freq, time, sampling_rate=44100, block=False):

    # Generate array with time*sample_rate steps, ranging between 0 and time
    t = np.linspace(0, time, time * sampling_rate, False)

    # Generate a 440 Hz sine wave
    note = np.sin(freq * t * 2 * np.pi)

    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    print(sampling_rate)
    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, sampling_rate)

    if block:
        play_obj.wait_done()

def usual():
    sound(440, 3, block=False)