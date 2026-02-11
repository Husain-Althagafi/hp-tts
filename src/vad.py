from silero_vad import load_silero_vad
import sounddevice as sd
import time
import torch
import numpy as np

def build_vad(device='cuda'):
    vadmodel = load_silero_vad()
    return vadmodel.to(device)


def record_one_utterance(vadmodel, start_threshold=0.5, end_silence_ms=1000, max_utterance_s=12.0, frame_samples=None, sample_rate=None, device='cuda'):
    t0 = time.time()

    frames = []
    started = False

    with sd.InputStream(channels=1, samplerate=sample_rate, dtype='float32', blocksize=frame_samples) as stream:
        while True:
            if time.time() - t0 > max_utterance_s:
                break
            data, _ = stream.read(frame_samples)   # data.shape = (frames, channels) in this case (frame_samples, 1)
            data = data[:, 0]
            frames.append(data)
            data = torch.from_numpy(data).float().unsqueeze(0).to(device)

            probs = vadmodel(data, sample_rate).item()

            if not started:    
                if probs >= start_threshold:
                    started = True
                    silence_start = None
            else:
                if probs < start_threshold:
                    if silence_start is None:
                        silence_start = time.time()

                    elif (time.time() - silence_start)*1000 >= end_silence_ms: 
                        break
                else:
                    silence_start = None
    
    audio = np.concatenate(frames).astype(np.float32)
    return audio









    
