from transformers import AutoModelForCausalLM, AutoTokenizer


def build_llm_and_tokenizer(model_name:str = 'Qwen/Qwen2.5-7B-Instruct'):
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
