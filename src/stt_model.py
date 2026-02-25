from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from peft import PeftModel

class STTModel:
    def __init__(self, language:str, model_name:str = 'openai/whisper-small', task:str = 'transcribe', device:str = 'cuda', lora_model:str = None):
        self.device = device
        self.large = True if 'whisper-large' in model_name else False
        if 'whisper-large' in model_name:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.generator = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation="sdpa").to(self.device)
            if lora_model is not None:
                self.generator = PeftModel.from_pretrained(self.generator, lora_model).to(self.device)
            self.generator.eval()

            self.pipe = pipeline(
                'automatic-speech-recognition',
                model=self.generator,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=30,
                batch_size=16,
                torch_dtype=torch.float32,
                device=device,
                ignore_warning=True,
                generate_kwargs={
                    'language': language,
                    'task': task
                }
            )

    
    def process_features(self, sample, return_tensors='pt'):
        processed_features = self.processor(
            sample,
            sampling_rate=16000,
            return_tensors=return_tensors
        ).input_features

        return processed_features.to(self.device)


    def generate_ids(self, input_features):
        with torch.inference_mode():
            pred_ids = self.generator.generate(input_features)
            return pred_ids
    

    def decode(self, pred_ids, skip_special_tokens=True):
        transcription = self.processor.batch_decode(
            pred_ids,
            skip_special_tokens=skip_special_tokens
        )

        return transcription[0]
        

    def transcribe(self, sample, return_tensors='pt', skip_special_tokens=True):
        if self.large:
            return self.pipe(sample)['text']
        features = self.process_features(sample, return_tensors)
        pred_ids = self.generate_ids(features)
        transcription = self.decode(pred_ids, skip_special_tokens)

        return transcription
        

if __name__ == '__main__':
    sttmodel = STTModel(language='ar')
    