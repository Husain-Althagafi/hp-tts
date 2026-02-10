import argparse
import sounddevice as sd

from stt_model import STTModel
from vad import build_vad, record_one_utterance
from llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation

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
        '--frame_ms',
        type=int,
        default=32
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
    vadmodel = build_vad(device=ARGS.device)
    
    sd.default.samplerate = ARGS.sampling_rate   # set default sample rate to 16,000
    sd.default.channels = 1

    frame_samples = int(ARGS.sampling_rate * ARGS.frame_ms / 1000)  # 480 samples at 16,000 hz
    while True:
        print(f'Recording one utterance...')
        audio = record_one_utterance(vadmodel=vadmodel, frame_samples=frame_samples, sample_rate=ARGS.sampling_rate, device=ARGS.device)  # returns audio a numpy array of shape (samples, frames, 1)
        print(f'Audio length: {len(audio)}')
        print(f'Audio: {audio}')

        print(f'Beginning Speech-to-text pipeline...')
        transcription = sttmodel.transcribe(audio) 
        print(f'Transcription complete...\nTranscriped audio: {transcription}')
    
        if ARGS.llm == 'api':
            print(f'Making llm api request...')
            client = build_llm_and_tokenizer(ARGS.llm)
            print(f'Generating LLM response...')
            response = client.models.generate(
                model='gemini-3-flash-preview',
                contents = f'Respond to this in a conversational manner: {transcription}'
            ).text
            
        else:    
            print(f'Loading LLM and tokenizer...')
            llm, llm_tokenizer = build_llm_and_tokenizer(ARGS.llm)
            print(f'LLM and tokenizer loaded...')

            print(f'Applying chat template...')
            text = use_chat_template(llm_tokenizer, transcription)
            print(f'Generating LLM response...')
            response = full_generation(llm, llm_tokenizer, text, ARGS.device, ARGS.max_new_tokens)

        print(f'LLM response: {response}')

        #tts part

        

if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()