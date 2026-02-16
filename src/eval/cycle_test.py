from src.stt_model import STTModel
from src.tts_model import TTSModel
import os
import pandas as pd
import argparse
from src.sample_texts import english_test_set, arabic_test_set
from src.llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation
from jiwer import cer
from tqdm import tqdm
from src.tts_pipeline import normalize_text 

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

    stt = STTModel(language=args.pipe_language, model_name=MODELS['stt'])
    tts = TTSModel(model_name=MODELS[f'tts_{args.pipe_language}'])

    if args.pipe_language == 'english':
        evalset = english_test_set
    elif args.pipe_language == 'arabic':
        evalset = arabic_test_set

    total_score = 0.0
    count = len(evalset)
    for sample in tqdm(evalset):
        wav = tts.synthesize(sample['text'])
        transcription = normalize_text(stt.transcribe(wav))
        norm_text = normalize_text(sample['text'])

        score = cer(sample['text'], transcription)
        print(sample["id"])
        print(f"Original:   {sample["text"]}")
        print(f'Normalized: {norm_text}')
        print(f"ASR Back:   {transcription}")
        print("CER:", score)
        print("-" * 40)
        total_score += score

    print(f'Average cer score is: {total_score/count}')


