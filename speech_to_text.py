from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 

    WhisperProcessor,
    WhisperForConditionalGeneration,
)

import numpy as np
import logging
import torch

logging.basicConfig(level=logging.INFO, format='[STT_MODULE]%(asctime)s %(levelname)s %(message)s')

def speech_to_text_whisper(
        pretrained_model_name_or_path: str = None,
        audio: np.ndarray = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
    ):
    if pretrained_model_name_or_path is None: raise ValueError(f"pretrained_model_name_or_path argument is required. Excepted: str, but got {pretrained_model_name_or_path}")
    if audio is None: raise ValueError(f"audio argument is required. Excepted: np.ndarray, but got {audio}")
    if audio_sample_rate is None:
        audio_sample_rate = 16_000
        logging.info(f"audio_sample_rate argument is not provided. Using default value: {audio_sample_rate}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device argument is not provided. Using default value: {device}")
    
    if verbose: logging.info("Loading model...")
    processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path).to(device)
    model.config.forced_decoder_ids = None
    for param in model.parameters():    param.requires_grad = False

    if verbose: logging.info("Converting audio...")
    input_features = processor(
        audio,
        sampling_rate=audio_sample_rate,
        return_tensors="pt"
    ).input_features.to(device)

    if verbose: logging.info("Inference...")
    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    if verbose: logging.info("Decoding...")
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription

def speech_to_text(
        processor_pretrained_argument: str = None,
        audio: torch.Tensor = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
    ):
    if processor_pretrained_argument is None: raise ValueError(f"processor_pretrained_argument argument is required. Excepted: str, but got {processor_pretrained_argument}")
    if audio is None: raise ValueError(f"audio argument is required. Excepted: torch.Tensor, but got {audio}")
    if audio_sample_rate is None:
        audio_sample_rate = 16_000
        logging.info(f"audio_sample_rate argument is not provided. Using default value: {audio_sample_rate}")
    if device is None: 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"device argument is not provided. Using default value: {device}")

    if verbose: logging.info("Loading model...")
    processor = Wav2Vec2Processor.from_pretrained(processor_pretrained_argument)
    model = Wav2Vec2ForCTC.from_pretrained(processor_pretrained_argument).to(device)

    if verbose: logging.info("Converting audio...")
    input_values = processor(
        audio,
        sampling_rate=audio_sample_rate,
        return_tensors="pt"
    ).input_values.to(device)

    if verbose: logging.info("Inference...")
    with torch.no_grad():
        logits = model(input_values).logits

    if verbose: logging.info("Decoding...")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription
    