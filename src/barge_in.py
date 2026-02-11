import threading
import torch
import sounddevice as sd
import time
import numpy as np

def start_barge_in_listener(
    vadmodel,
    sample_rate: int,
    frame_samples: int,
    stop_tts_event: threading.Event,
    is_tts_playing: threading.Event,
    device=None,
    threshold: float = 0.5,
    consecutive_frames: int = 3,
    ignore_first_seconds: float = 0.30,
):
    """
    Runs in a background thread:
    - reads mic frames continuously
    - computes VAD speech probability
    - if TTS is playing AND speech is detected for N consecutive frames
      (after a short ignore window), it triggers stop_tts_event.
    """
    def _run():
        speech_count = 0
        tts_start_time = 0.0

        with sd.InputStream(
            channels=1,
            samplerate=sample_rate,
            dtype="float32",
            blocksize=frame_samples,
        ) as stream:
            while True:
                data, _ = stream.read(frame_samples)  # shape (frame_samples, 1)
                frame = data[:, 0].copy()              # shape (frame_samples,)

                # If not currently playing TTS, reset counters and skip
                if not is_tts_playing.is_set():
                    speech_count = 0
                    continue

                # Track when TTS started (set by main loop via a shared timestamp trick)
                # We'll approximate by resetting when stop event is cleared.
                # If your code clears stop_tts_event before starting TTS, this works well.
                if stop_tts_event.is_set():
                    # if TTS already stopped, no need to detect speech
                    speech_count = 0
                    continue

                # Ignore early window to reduce echo-triggered barge-in
                # (main thread will clear stop_tts_event right before starting TTS)
                # We'll infer "tts_start_time" as the moment is_tts_playing becomes true.
                if tts_start_time == 0.0:
                    tts_start_time = time.time()

                if time.time() - tts_start_time < ignore_first_seconds:
                    continue

                # Silero expects shape (1, N)
                x = torch.from_numpy(frame).float().unsqueeze(0)
                if device and device.startswith("cuda"):
                    x = x.to(device)

                try:
                    p = vadmodel(x, sample_rate).item()
                except Exception:
                    # if anything odd happens, just skip this frame
                    continue

                if p >= threshold:
                    speech_count += 1
                    if speech_count >= consecutive_frames:
                        stop_tts_event.set()
                        # reset so we don't repeatedly trigger
                        speech_count = 0
                        tts_start_time = 0.0
                else:
                    speech_count = 0

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
