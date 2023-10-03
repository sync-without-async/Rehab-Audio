from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,

    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    Wav2Vec2ProcessorWithLM,
)

from transformers.pipelines import (
    AutomaticSpeechRecognitionPipeline
)

from pyctcdecode import build_ctcdecoder

from datasets import load_dataset

import soundfile as sf
import logging

import numpy as np
import torchaudio
import torch

import librosa

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def speech_to_text_with_pipeline(
        model_for_ctc_pretrained_argument: str = None,
        audio: np.ndarray = None,
        audio_sample_rate: int = None,
        verbose: bool = False,
):
    if verbose: logging.info("Loading model...")
    model = AutoModelForCTC.from_pretrained(model_for_ctc_pretrained_argument)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_for_ctc_pretrained_argument)
    tokenizer = AutoTokenizer.from_pretrained(model_for_ctc_pretrained_argument)
    beam_decoder = build_ctcdecoder(
        labels=tokenizer.get_vocab(),
        kenlm_model_path=None,
    )
    processor = Wav2Vec2ProcessorWithLM(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder=beam_decoder
    )

    if verbose: logging.info("Converting audio...")
    asr_pipeline = AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        decoder=beam_decoder,
        device=-1
    )

    kwargs = {
        "decoder_kwargs": {
            {"beam_width": 100}
        }
    }

    if verbose: logging.info("Inference...")
    transcription = asr_pipeline(audio, **kwargs)["text"]

    if verbose: logging.info("Done!") 
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
    