from kokoro import KPipeline
from kokoro.model import KModel
import soundfile as sf
import os
import numpy as np
import sounddevice as sd
import time
from transformers import AutoTokenizer, VitsModel
import torch
from sample_texts import arabic_tts_test_sentences
from tts_pipeline import normalize_nums

class TTSModel:
    def __init__(self, model_name:str = 'kokoro', device='cuda'):
        if model_name == 'kokoro':
            model_path = "C:/Users/husain_althagafi/work/storage/models_stage1/kokoro"
            kmodel = self.load_kmodel_local(model_path)
            print(f'model loaded yay')
            self.model = KPipeline(model=kmodel, lang_code='a') # american english
            self.model_type = 'kokoro'
            self.model_rate = 24000

        if 'facebook' in model_name:
            self.model = VitsModel.from_pretrained(model_name).to(device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_type = 'facebook'
            self.model_rate = 16000

    def synthesize(self, text, voice='af_heart'):
        text = normalize_nums(text)
        if self.model_type == 'facebook':
            ids = self.tokenizer(text, return_tensors='pt').to(self.model.device)   

            with torch.inference_mode():
                output = self.model(
                    **ids,
                ).waveform
                return output.squeeze().cpu().numpy()
        
        chunks = []

        generator = self.model(text, voice=voice)
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            chunks.append(audio)

        if not chunks:
            raise RuntimeError(f'TTS produced no audio')
        
        full_audio = np.concatenate(chunks)
        return full_audio #if full_audio is not None else np.array()
    

    def synthesize_to_file(self, text, voice='af_heart', output_path=f'outputs/audios/{time.time()}.wav'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio = self.synthesize(text, voice=voice)
        if self.model_type == 'kokoro':
            sf.write(output_path, audio, samplerate=self.model_rate)
        elif self.model_type == 'facebook':
            sf.write(output_path, audio, samplerate=self.model_rate)

    
    def stream_chunks(self, text, voice='af_heart'):
        text = normalize_nums(text)
        if self.model_type == 'facebook':
            raise Exception(f'Cant stream outputs with modeltype {self.model_type}')
        generator = self.model(text, voice=voice)
        for _, _, audio in generator:
            yield np.asarray(audio, dtype=np.float32)


    def load_kmodel_local(self, model_dir: str, device: str = "cuda"):
        # 1) prefer passing config path to avoid any hf_hub_download calls
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found at {config_path}. Put config.json in the model folder.")

        # 2) try to detect the checkpoint name; common names: model.pth, best.pt, weights.pt
        possible_names = ["kokoro-v1_0.pth", "weights.pth", "best.pt", "checkpoint.pth", "model.pt"]
        ckpt_path = None
        for name in possible_names:
            p = os.path.join(model_dir, name)
            if os.path.exists(p):
                ckpt_path = p
                break

        # If the KModel constructor accepts a model argument that's just the filename,
        # pass the basename (or full path depending on your kokoro version).
        model_arg = None
        if ckpt_path is not None:
            model_arg = os.path.join(ckpt_path)  # often the constructor expects filename inside repo
        else:
            # fallback: if no checkpoint found, try letting KModel handle it (it may have HF logic)
            print("Warning: no typical checkpoint found in model_dir. Constructor may attempt downloads.")
            model_arg = None

        # 3) construct the KModel pointing at local repo and local config
        # Passing config=config_path avoids hf_hub_download for config.
        kmodel = KModel(config=config_path, model=model_arg, disable_complex=False)

        # 4) move to device & set eval; optionally use half precision on GPU
        kmodel = kmodel.to(device).eval()
        if device.startswith("cuda"):
            try:
                kmodel = kmodel
            except Exception:
                # some models may not support .half(); ignore safely
                pass

        return kmodel
    

        
        
if __name__ == '__main__':
    tts = TTSModel(model_name='facebook/mms-tts-ara')
    # tts = TTSModel()
    # txt = 'Hi, my name is Mario and I like pizza!'
    i, e, s = 0, -1, 5
    txts = arabic_tts_test_sentences[i:e:s]
    for txt in txts:
        print(f'Playing tts audio...')
        audio = tts.synthesize(txt)
        if tts.model_type == 'facebook':
            sd.play(audio, samplerate=tts.model_rate)
            sd.wait()

        elif tts.model_type == 'kokoro':
            sd.play(audio, samplerate=tts.model_rate)
            sd.wait()

            print(f'Streaming tts audio...')
            for chunk in tts.stream_chunks(txt):
                sd.play(chunk)
                sd.wait()

        tts.synthesize_to_file(txt)
        print(f'Generating audio file...')

    

