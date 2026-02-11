from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class STTModel:
    def __init__(self, language:str, model_name:str = 'openai/whisper-small', task:str = 'transcribe', device:str = 'cuda'):
        self.device = device
        if 'whisper' in model_name:
            self.processor = WhisperProcessor.from_pretrained(model_name)

            self.generator = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32).to(self.device)
            self.generator.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
            self.generator.eval()

    
    def process_features(self, sample, return_tensors='pt'):
        processed_features = self.processor(
            sample,
            sampling_rate=16000,
            return_tensors=return_tensors
        ).input_features

        return processed_features.to(self.device)


    def generate(self, input_features):
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
        features = self.process_features(sample, return_tensors)
        pred_ids = self.generate(features)
        transcription = self.decode(pred_ids, skip_special_tokens)

        return transcription
        

if __name__ == '__main__':
    sttmodel = STTModel(language='ar')
    

