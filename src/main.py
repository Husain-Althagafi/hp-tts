import argparse

from src.pipeline import VoicePipeline


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
        default='D:/storage/Qwen2.5-7B-Instruct'
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

    parser.add_argument(
        '--lora_model',
        type=str,
        default=None
        # D:/storage/my_model
    )

    return parser.parse_args()


def main():
    pipe = VoicePipeline(
        device=ARGS.device,
        llm=ARGS.llm,
        language=ARGS.language,
        sampling_rate=ARGS.sampling_rate,
        frame_ms=ARGS.frame_ms,
        max_new_tokens=ARGS.max_new_tokens,
        lora_model=args.lora_model
    )
    
    while True:
        pipe.run()


if __name__ == '__main__':
    global ARGS
    ARGS = parse_args()
    main()