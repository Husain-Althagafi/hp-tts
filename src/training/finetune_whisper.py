from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch


def build_model(model_name='C:/Users/husain_althagafi/work/storage/whisper-large-v3', device='cuda'):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa").to(device)
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def build_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
    )

    return get_peft_model(model, config)


def main():
    model_name = 'C:/Users/husain_althagafi/work/storage/whisper-large-v3'
    model, processor = build_model(model_name) 
    lora_model = build_lora(model)
    
    training_args = Seq2SeqTrainingArguments(
            output_dir=f"../../outputs/finetunes/{model_name.split('/')[-1]}", 
            per_device_train_batch_size=16,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=5000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer", 
            greater_is_better=False,
        )
    
    trainer = Seq2SeqTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=None,  # Replace with your training dataset
        eval_dataset=None,   # Replace with your evaluation dataset
        data_collator=None,  # Replace with your data collator if needed
        compute_metrics=None,  # Replace with your metric computation function if needed
        tokenizer=processor.feature_extractor,  # Use the feature extractor as the tokenizer
    )

    trainer.train()


if __name__ == '__main__':
    main()