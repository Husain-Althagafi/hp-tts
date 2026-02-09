from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, VADIterator
import argparse
import sounddevice as sd

from stt_model import STTModel
from vad import build_vad, record_one_utterance
from llm_responder import build_llm_and_tokenizer, use_chat_template

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

    parser.add_argument(
        '--llm',
        type=str,
        default='Qwen/Qwen2.5-7B-Instruct'
    )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256
    )

    return parser.parse_args()


def main():
    sttmodel = STTModel(device=ARGS.device)
    vadmodel, vad_iter = build_vad(sampling_rate=ARGS.sampling_rate, device=ARGS.device)
    
    sd.default.samplerate = ARGS.sampling_rate   # set default sample rate to 16,000
    sd.default.channels = 1

    frame_samples = int(ARGS.sampling_rate * ARGS.frame_ms / 1000)  # 480 samples at 16,000 hz
    while True:
        print(f'Recording one utterance...')
        audio = record_one_utterance(vadmodel=vadmodel, frame_samples=frame_samples, sample_rate=ARGS.sample_rate)  # returns audio a numpy array of shape (samples, frames, 1)
        print(f'Audio length: {len(audio)}')

        print(f'Beginning Speech-to-text pipeline...')
        transcription = sttmodel.transcribe(audio)  # i think theres gna be an issue here will see later
        print(f'Transcription complete...\nTranscriped audio: {transcription}')

        print(f'Loading LLM and tokenizer...')
        llm, llm_tokenizer = build_llm_and_tokenizer(ARGS.llm)
        print(f'LLM and tokenizer loaded...')

        print(f'Generating LLM response...')
        text = use_chat_template(llm_tokenizer, transcription)
        model_inputs = llm_tokenizer([text], return_tensors='pt').to(ARGS.device)

        generated_ids = llm.generate(
            **model_inputs,
            max_new_tokens=ARGS.max_new_tokens
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f'LLM response: {response}')


if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()