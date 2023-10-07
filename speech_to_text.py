from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
)

import logging
import torch

logging.basicConfig(level=logging.INFO, format='[STT_MODULE]%(asctime)s %(levelname)s %(message)s')

def speech_to_text(
        processor_pretrained_argument: str = None,
        audio: torch.Tensor = None,
        audio_sample_rate: int = None,
        device: torch.device = None,
        verbose: bool = False,
    ) -> list:
    """Speech to text (a.k.a. STT) using pretrained model. Only support 16kHz audio.
    Before using this function, you should load the audio using load_audio function.
    This function will return the transcription as a list. 
    Speech to text model will be selected using processor_pretrained_argument argument. By default, this function will use kresnik/wav2vec2-large-xlsr-korean model.
    You can find the list of pretrained model in https://huggingface.co/models?filter=wav2vec2.
    You can also use your own model. If you want to use your own model, you should pass the path of the model to processor_pretrained_argument argument.
    For example, if you want to use your own model, you can use the following command:
        >>> speech_to_text(
        >>>     processor_pretrained_argument="path/to/your/model",
        >>>     audio=audio,
        >>>     audio_sample_rate=audio_sample_rate,
        >>>     device=device,
        >>>     verbose=True
        >>> )

    Args:
        processor_pretrained_argument (str): The pretrained model name or path.
        audio (torch.Tensor): The audio tensor. 
        audio_sample_rate (int): The sample rate of the audio.
        device (torch.device): The device that you want to use. You can use "cuda" or "cpu".
        verbose (bool): If True, print the log.

    Returns:
        list: The transcription as a list."""
    if processor_pretrained_argument is None: 
        processor_pretrained_argument = "kresnik/wav2vec2-large-xlsr-korean"
        logging.info(f"processor_pretrained_argument argument is not provided. Using default value: {processor_pretrained_argument}")
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
        print(input_values)
        logits = model(input_values).logits

    if verbose: logging.info("Decoding...")
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription
    