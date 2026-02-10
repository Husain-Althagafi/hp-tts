from kokoro import KPipeline
from kokoro.model import KModel
import soundfile as sf
import os


class TTSModel:
    def __init__(self, model_name:str = 'kokoro', device='cuda'):
        if model_name == 'kokoro':
            model_path = "C:/Users/husain_althagafi/work/storage/models_stage1/kokoro"
            kmodel = self.load_kmodel_local(model_path)
            print(f'model loaded yay')
            self.model = KPipeline(model=kmodel, lang_code='a') # american english
    

    def generate_file(self, text):
        generator = self.model(text, voice='af_heart')
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            sf.write(f'{i}.wav', audio, 24000)


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
    tts.generate_file('Hello my name is husain')
    print(f'Generating audio file')