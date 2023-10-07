from denoiser.dsp import convert_audio
from denoiser import pretrained
import torchaudio
import torch

from typing import Tuple

import logging

logging.basicConfig(level=logging.INFO, format='[DSR_MODULE]%(asctime)s %(levelname)s %(message)s')

def _check_parallel_device_list():
    if not torch.cuda.is_available():
        return ["cpu"]

    device_list = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    return device_list

def load_audio(
        path: str = None
    ) -> Tuple[torch.Tensor, int]:
    """Load audio file and return the audio and the sample rate.
    Audio file should be a wav file. If you want to load other file types, you should convert the file to a wav file.
    You can use ffmpeg to convert the file to a wav file. 
    For example, if you want to convert a mp3 file to a wav file, you can use the following command:
        ffmpeg -i input.mp3 output.wav
        
    Args:
        path (str): The path of the audio file.
        
    Returns:
        tuple: 
            audio (torch.Tensor): The audio tensor.
            sample_rate (int): The sample rate of the audio file."""
    if path is None:    raise ValueError(f"path argument is required. Excepted: str, but got {path}")

    audio, sample_rate = torchaudio.load(path)

    return audio, sample_rate

def denoising(
        audio: torch.Tensor = None,
        sample_rate: int = None,
        device: torch.device = None,
        inference_parallel: bool = False,
        verbose: bool = False
    ):
    """Denoising audio using pretrained model. Only support 16kHz audio. 
    Before using this function, you should load the audio using load_audio function.
    This function will return the denoised audio and the sample rate of the denoised audio.
    Denosier model is pretrained that is trained using DNS64 by Meta.

    Args:
        audio (torch.Tensor): The audio tensor. 
        sample_rate (int): The sample rate of the audio.
        device (torch.device): The device that you want to use. You can use "cuda" or "cpu".
        inference_parallel (bool): If True, use parallel inference.
        verbose (bool): If True, print the log.

    Returns:
        tuple: 
            output (np.ndarray): The denoised audio.
            sample_rate (int): The sample rate of the denoised audio."""
    if device is None:  raise ValueError(f"device argument is required. Excepted: 'cuda' or 'cpu', but got {device}")
    if sample_rate is None:  raise ValueError(f"sample_rate argument is required. Excepted: int, but got {sample_rate}")
    if audio is None:   raise ValueError(f"audio argument is required. Excepted: torch.Tensor, but got {audio}")

    if verbose: logging.info("Loading model...")
    model = pretrained.dns64(pretrained=True).to(device)
    model_sample_rate = model.sample_rate
    model_chin = model.chin

    if inference_parallel:
        device_list = _check_parallel_device_list()
        if verbose: logging.info(f"Parallel inference... Device list: {device_list}")
        model = torch.nn.DataParallel(model, device_ids=device_list)

    if audio.ndim == 1: audio = audio.unsqueeze(0)

    if verbose: logging.info("Converting audio...")
    wav = convert_audio(wav=audio, 
                        from_samplerate=sample_rate, 
                        to_samplerate=model_sample_rate, 
                        channels=model_chin
                    ).to(device)

    if verbose: logging.info("Inference...") 
    with torch.no_grad():
        output = model(wav)

    if verbose: logging.info("Converting output...")
    output = output.squeeze(0).cpu().numpy()

    if verbose: logging.info("Done!")
    return output, model_sample_rate