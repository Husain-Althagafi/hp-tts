from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
         '--lan',
        type=str,
        default='en'
    )   

    parser.add_argument(
        '--testing',
        action='store_true'
    )

    parser.add_argument(
        '--llm',
        type=str,
        default='Qwen/Qwen3-8B'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    args = parser.parse_args()
    return args


def load_dummy_data():
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    return ds


def llm_responder(model_name, device):
    model, tokenizer = AutoModelForCausalLM.from_pretrained(model_name).to(device), AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def main(args):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    if args.lan == 'en':
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")

    if args.lan == 'ar':
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")

    if args.testing:
        ds = load_dummy_data()

    else:
        ds = None

    llm, tokenizer = llm_responder(args.llm, args.device)

    sample = ds[0]['audio']
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    pred_ids = model.generate(input_features)

    transcription = processor.batch_decode(
        pred_ids,
        skip_special_tokens=True,
    )

    print(f'Transcription: {transcription[0]}')

    messages = [
        {"role": "user", "content": transcription[0]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = llm.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    output = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    ).strip('\n')

    print(f'LLM Output: {output}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
    
