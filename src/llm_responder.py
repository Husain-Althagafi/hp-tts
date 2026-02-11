from transformers import AutoModelForCausalLM, AutoTokenizer
import google.genai as genai


def build_llm_and_tokenizer(model_name:str = 'Qwen2.5-7B-Instruct'):
    if model_name == 'api':
        return genai.Client()

    return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cuda'), AutoTokenizer.from_pretrained(model_name)


def use_chat_template(tokenizer, prompt:str = 'Hi, I like cats'):
    messages = [
        {
            'role': 'system',
            'content': "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        },
        {
            'role': 'user',
            'content': prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return text


def full_generation(model, tokenizer, text, device, max_new_tokens):
    model_inputs = tokenizer([text], return_tensors='pt').to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )

    output_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response

