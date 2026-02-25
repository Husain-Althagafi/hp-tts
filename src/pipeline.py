import sounddevice as sd
import threading
import soundfile as sf
import time as time
import os

from src.stt_model import STTModel
from src.vad import build_vad, record_one_utterance
from src.llm_responder import build_llm_and_tokenizer, use_chat_template, full_generation
from src.tts_model import TTSModel
from src.barge_in import start_barge_in_listener


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

        if language == 'arabic':
            tts = 'facebook/mms-tts-ara'
            stt = 'D:/storage/whisper-large-v3'
        
        elif language == 'english':
            tts = 'kokoro'
            stt = 'D:/storage/whisper-large-v3'

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


    def stt(self, audio):
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

        return audio
            

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


    def save_run(self, original_audio, transcription, llm_response, response_audio, output_path):
        
        sf.write(output_path + '/src_audio.wav', original_audio, 16000)
        sf.write(output_path + '/response_audio.wav', response_audio, self.ttsmodel.model_rate)

        with open(output_path + '/outputs.txt', 'w', encoding='utf-8') as f:
            f.write(f'Transcription: {transcription}')
            f.write(f'LLM Response: {llm_response}')


    def run(self):
        save_time = time.time()
        output_path = f'./outputs/runs/{save_time}'
        os.makedirs(output_path, exist_ok=True)

        audio = self.record()
        transcription, src_audio = self.stt(audio)
        if transcription.lower().strip() == "exit":
            return transcription, ""

        llmresponse = self.run_llm(transcription)
        print(f'LLM response: {llmresponse}')
        response_audio = self.ttsmodel.synthesize(llmresponse)

        if self.ttsmodel.model_type != 'facebook':
            self.stream_tts(llmresponse)
        else:
            self.tts_play(llmresponse)

        self.save_run(src_audio, transcription, llmresponse, response_audio, output_path)


if __name__ == '__main__':
    print(f'Running Pipeline...')
    pipe = VoicePipeline()
    pipe.run()