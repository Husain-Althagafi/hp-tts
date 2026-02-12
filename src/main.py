import argparse

from pipeline import VoicePipeline

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    parser.add_argument(
        '--sampling_rate',
        type=int,
        default=16000
    )

    parser.add_argument(
        '--frame_ms',
        type=int,
        default=32
    )

    parser.add_argument(
        '--llm',
        type=str,
        default='C:/Users/husain_althagafi/work/storage/Qwen2.5-7B-Instruct'
    )

    # parser.add_argument(
    #     '--sttmodel',
    #     type=str,
    #     default='openai/whisper-small'
    # )

    # parser.add_argument(
    #     '--ttsmodel',
    #     type=str,
    #     default='kokoro'
    # )

    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=256
    )

    parser.add_argument(
        '--language',
        type=str,
        default='en'
    )

    return parser.parse_args()


def main():
    pipe = VoicePipeline(
        device=ARGS.device,
        llm=ARGS.llm,
        language=ARGS.language,
        sampling_rate=ARGS.sampling_rate,
        frame_ms=ARGS.frame_ms,
        max_new_tokens=ARGS.max_new_tokens
    )
    

    while True:
       trancription, llmresponse, sttaudio = pipe.run() #add a method for the pipeline to save the audio files of the speaker and the output of the tts
       print(trancription, llmresponse)


if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()