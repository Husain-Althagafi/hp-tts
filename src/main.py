import argparse
import sounddevice as sd
import time
import threading

from stt_model import STTModel
from vad import build_vad, record_one_utterance
from llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation
from tts_model import TTSModel
from barge_in import start_barge_in_listener

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
    #   instantiate models
    sttmodel = STTModel(device=ARGS.device)
    vadmodel = build_vad(device=ARGS.device)
    ttsmodel = TTSModel()

    if ARGS.llm == 'api':
                print(f'Making llm api request...')
                client = build_llm_and_tokenizer(ARGS.llm)
    else:
        print(f'Loading LLM and tokenizer...')
        llm, llm_tokenizer = build_llm_and_tokenizer(ARGS.llm)
    
    
    sd.default.samplerate = ARGS.sampling_rate   # set default sample rate to 16,000
    sd.default.channels = 1

    frame_samples = int(ARGS.sampling_rate * ARGS.frame_ms / 1000)  # 480 samples at 16,000 hz

    stop_tts_event = threading.Event()
    is_tts_playing = threading.Event()

    start_barge_in_listener(
        vadmodel=vadmodel,
        sample_rate=ARGS.sampling_rate,
        frame_samples=frame_samples,
        stop_tts_event=stop_tts_event,
        is_tts_playing=is_tts_playing,
        device=ARGS.device,
        threshold=0.5,
        consecutive_frames=3,
        ignore_first_seconds=0.30,
    )

    while True:
        print(f'Recording one utterance...')
        audio = record_one_utterance(vadmodel=vadmodel, frame_samples=frame_samples, sample_rate=ARGS.sampling_rate, device=ARGS.device)  # returns audio a numpy array of shape (samples, frames, 1)
        print(f'Audio length: {len(audio)}')
        print(f'Audio: {audio}')

        print(f'Beginning Speech-to-text pipeline...')
        current_time = time.time()
        transcription = sttmodel.transcribe(audio) 
        print(f'Transcription complete...\nTranscriped audio: {transcription}')
        current_time_diff = time.time() - current_time
        current_time += current_time_diff
        print(f'Transcription time: {current_time_diff}')
    
        if transcription.lower().strip() == 'exit':
            print(f'Ending script...')
            break
        
        if ARGS.llm == 'api':
            print(f'Generating LLM response...')
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{transcription}",
            ).text
                
        else:    
            print(f'Applying chat template...')
            text = use_chat_template(llm_tokenizer, transcription)

            print(f'Generating LLM response...')
            response = full_generation(llm, llm_tokenizer, text, ARGS.device, ARGS.max_new_tokens)

        print(f'LLM response: {response}')

        current_time_diff = time.time() - current_time
        current_time += current_time_diff
        print(f'LLM response time: {current_time_diff}')

        #   ---TTS---
        print(f'Beginning Text-to-Speech...')
        stop_tts_event.clear()
        is_tts_playing.set()

        print(f'Streaming...')
        for chunk in ttsmodel.stream_chunks(response):
            if stop_tts_event.is_set():
                print("Stopping TTS")
                sd.stop()
                break

            sd.play(chunk, samplerate=24000)

            # Instead of sd.wait(), poll in small intervals
            while sd.get_stream().active:
                if stop_tts_event.is_set():
                    print("Stopping TTS mid-chunk")
                    sd.stop()
                    break
                sd.sleep(20)

            if stop_tts_event.is_set():
                break

        is_tts_playing.clear()

if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()