from src.stt_model import STTModel
from src.tts_model import TTSModel
import os
import pandas as pd
import argparse
from src.sample_texts import english_test_set, arabic_test_set
from src.llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation
from jiwer import cer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pipe_language',
        type=str,
        default='english'
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='data'
    ) 

    parser.add_argument(
        '--llm',
        type=str,
        default='C:/Users/husain_althagafi/work/storage/Qwen2.5-7B-Instruct'
    )

    return parser.parse_args()


if __name__ == '__main__':

    MODELS = {
        'stt': 'C:/Users/husain_althagafi/work/storage/whisper-large-v3',
        'tts_english': 'kokoro',
        'tts_arabic': 'facebook/mms-tts-ara'
    }
    
    args = parse_args()

    # data_path = os.path.join(f'{args.data_path}', args.pipe_language)
    # print(data_path)

    stt = STTModel(language=args.pipe_language, model_name=MODELS['stt'])
    tts = TTSModel(model_name=MODELS[f'tts_{args.pipe_language}'])
    # llm, llm_tokenizer = build_llm_and_tokenizer(args.llm)

    if args.pipe_language == 'english':
        evalset = english_test_set
    elif args.pipe_language == 'arabic':
        evalset = arabic_test_set

    for sample in evalset:
        wav = tts.synthesize(sample['text'])
        transcription = stt.transcribe(wav)

        score = cer(sample['text'], transcription)
        print(sample["id"])
        print("Original:", sample["text"])
        print("ASR Back:", transcription)
        print("CER:", score)
        print("-" * 40)



