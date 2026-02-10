from kokoro import KPipeline
from kokoro.model import KModel
import soundfile as sf
import os
import numpy as np
import sounddevice as sd
import time

class TTSModel:
    def __init__(self, model_name:str = 'kokoro', device='cuda'):
        if model_name == 'kokoro':
            model_path = "C:/Users/husain_althagafi/work/storage/models_stage1/kokoro"
            kmodel = self.load_kmodel_local(model_path)
            print(f'model loaded yay')
            self.model = KPipeline(model=kmodel, lang_code='a') # american english
    

    def synthesize(self, text, voice='af_heart'):
        chunks = []

        generator = self.model(text, voice=voice)
        for _, _, audio in generator:
            audio = np.asarray(audio, dtype=np.float32)
            chunks.append(audio)

        if not chunks:
            raise RuntimeError(f'TTS produced no audio')
        
        full_audio = np.concatenate(chunks)
        return full_audio
    

    def synthesize_to_file(self, text, voice='af_heart', output_path=f'outputs/audios/{time.time()}.wav'):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        audio = self.synthesize(text, voice=voice)
        sf.write(output_path, audio, samplerate=10000000)

    
    def stream_chunks(self, text, voice='af_heart'):
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
    tts = TTSModel()
    # txt = input(f'Enter your message to tts: ')
    txt = 'Hi, my name is Mario and I like pizza!'
    tts.synthesize_to_file(txt)
    print(f'Generating audio file...')

    print(f'Playing tts audio...')
    audio = tts.synthesize(txt)
    sd.play(audio)
    sd.wait()

    print(f'Streaming tts audio...')
    for chunk in tts.stream_chunks(txt):
        sd.play(chunk)
        sd.wait()

