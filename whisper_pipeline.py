import torch
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
import time
from app import model_dict


def run(model_id, audio):
    model = model_dict[model_id]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): ## 애플 M1맥북
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(device)
    language = "ko"
    task = "transcribe"
    chunk_length_s = 20
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language=language, task=task)
    processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language=language, task=task)
    feature_extractor = processor.feature_extractor
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    pipe = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        chunk_length_s=chunk_length_s,
        device=device,
    )

    def transcribe(audio):
        start = time.time()
        with torch.cuda.amp.autocast():
            text = pipe(audio, generate_kwargs={"forced_decoder_ids": forced_decoder_ids}, max_new_tokens=255,
                        return_timestamps=True)["text"]
        end = time.time()
        print(f"Elapsed time for transcription: {end - start} seconds")
        return text

    return transcribe(audio)
