from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator
import argparse
import sounddevice as sd

from stt_model import STTModel
from vad import build_vad, record_one_utterance

ARGS = None

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=16000
    )

    parser.add_argument(
        'frame_ms',
        type=int,
        default=30
    )

    return parser.parse_args()


def main():
    sttmodel = STTModel(device=ARGS.device)
    vadmodel, vad_iter = build_vad(sampling_rate=ARGS.sampling_rate, device=ARGS.device)
    
    sd.default.samplerate = ARGS.sampling_rate   # set default sample rate to 16,000
    sd.default.channels = 1

    frame_samples = int(ARGS.sampling_rate * ARGS.frame_ms / 1000)  # 480 samples at 16kHz
    while True:
        audio = record_one_utterance(vadmodel=vadmodel, frame_samples=frame_samples, sample_rate=ARGS.sample_rate)  # returns audio a numpy array of shape (samples, frames, 1)
        print(len(audio))


if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()