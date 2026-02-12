import sounddevice as sd
import threading
import numpy as np
import tempfile
import soundfile as sf

from stt_model import STTModel
from vad import build_vad, record_one_utterance
from llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation
from tts_model import TTSModel
from barge_in import start_barge_in_listener


class VoicePipeline:
    def __init__(
            self,
            device='cuda',
            llm='C:/Users/husain_althagafi/work/storage/Qwen2.5-7B-Instruct',
            language='ar',
            sampling_rate=16000,
            frame_ms=32,
            max_new_tokens=256,
    ):
        
        self.device = device
        self.language = language
        self.samplerate = sampling_rate
        self.frame_ms = frame_ms
        self.max_new_tokens = max_new_tokens

        if language == 'ar':
            tts = 'facebook/mms-tts-ara'
            stt = 'C:/Users/husain_althagafi/work/storage/whisper-large-v3'
        
        elif language == 'en':
            self.language = 'english'
            tts = 'kokoro'
            stt = 'C:/Users/husain_althagafi/work/storage/whisper-large-v3'

        self.vadmodel = build_vad(device=device)
        self.sttmodel = STTModel(model_name=stt, device=device, language=language)
        self.ttsmodel = TTSModel(model_name=tts, device=device)

        if llm != 'api':
            self.llm, self.llm_tokenizer = build_llm_and_tokenizer(llm)

        else:
            self.llm, self.llm_tokenizer = build_llm_and_tokenizer(llm), None


        sd.default.samplerate = sampling_rate   # set default sample rate to 16,000
        sd.default.channels = 1 

        self.frame_samples = int(sampling_rate * frame_ms / 1000)

        self.stop_tts_event = threading.Event()
        self.is_tts_playing = threading.Event()

        start_barge_in_listener(
            vadmodel=self.vadmodel,
            sample_rate=self.samplerate,
            frame_samples=self.frame_samples,
            stop_tts_event=self.stop_tts_event,
            is_tts_playing=self.is_tts_playing,
            device=self.device,
            threshold=0.5,
            consecutive_frames=3,
            ignore_first_seconds=0.30,
        )

    
    def record(self):
        '''
        Returns audio
        '''
        print(f'Recording...')
        return record_one_utterance(
            vadmodel=self.vadmodel,
            frame_samples=self.frame_samples,
            sample_rate=self.samplerate,
            device=self.device
        )
    

    def run_llm(self, transcription):
        if self.llm_tokenizer is None: #this logic is weak since maybe a model wont have a tokenizer or the auto tokenizer will return None even when its not an api but for now np
            return self.llm.models.generate_content(
                model="gemini-3-flash-preview",
                contents=f"{transcription}",
            ).text

        text = use_chat_template(self.llm_tokenizer, transcription)
        return full_generation(self.llm, self.llm_tokenizer, text, self.device, self.max_new_tokens)


    def stt(self):
        audio = self.record()
        return self.sttmodel.transcribe(audio), audio
    
    
    def tts_play(self, text):
        self.stop_tts_event.clear()
        self.is_tts_playing.set()

        audio = self.ttsmodel.synthesize(text)

        if self.stop_tts_event.is_set():
            sd.stop()
            self.is_tts_playing.clear()
            return

        sd.play(audio, samplerate=self.ttsmodel.model_rate)

        while sd.get_stream().active:
            if self.stop_tts_event.is_set():
                sd.stop()
                break
            sd.sleep(20)

        self.is_tts_playing.clear()
            

    def stream_tts(self, text):
        self.stop_tts_event.clear()
        self.is_tts_playing.set()

        for chunk in self.ttsmodel.stream_chunks(text):
            if self.stop_tts_event.is_set():
                sd.stop()
                break

            sd.play(chunk, samplerate=self.ttsmodel.model_rate)

            while sd.get_stream().active:
                if self.stop_tts_event.is_set():
                    sd.stop()
                    break
                sd.sleep(20)

            if self.stop_tts_event.is_set():
                break

        self.is_tts_playing.clear()


    def run(self):
        transcription, sttaudio = self.stt()
        if transcription.lower().strip() == "exit":
            return transcription, ""

        llmresponse = self.run_llm(transcription)

        if self.ttsmodel.model_type != 'facebook':
            self.stream_tts(llmresponse)
        else:
            self.tts_play(llmresponse)

        return transcription, llmresponse, sttaudio
    

    def run_turn_from_gradio_audio(self, mic_audio):
        """
        mic_audio is usually (sample_rate, np.ndarray) from gr.Audio(type="numpy")
        Returns: transcription, response, wav_path
        """
        if mic_audio is None:
            return "", "", None

        sr, audio = mic_audio

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio[:, 0]  # mono

        # Whisper expects 16k
        audio_16k = self._linear_resample(audio, sr, 16000)

        transcription = self.sttmodel.transcribe(audio_16k)
        if transcription.lower().strip() == "exit":
            return transcription, "", None

        response = self.run_llm(transcription)

        # Get TTS waveform (you already have ttsmodel.synthesize)
        tts_wav = self.ttsmodel.synthesize(response).astype(np.float32)
        tts_sr = self.ttsmodel.model_rate

        # Write to temp wav for Gradio to play
        outpath = tempfile.mkstemp(suffix=".wav")
        sf.write(outpath, tts_wav, tts_sr)

        return transcription, response, outpath
        

    def _linear_resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return audio.astype(np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        n_new = int(len(audio) * (sr_out / sr_in))
        x_new = np.linspace(0.0, 1.0, num=n_new, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


if __name__ == '__main__':
    print(f'Running Pipeline...')
    pipe = VoicePipeline()
    pipe.run()