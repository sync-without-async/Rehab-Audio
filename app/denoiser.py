from denoiser.dsp import convert_audio
from app.denoiser import pretrained

from scipy.signal import spectrogram

import numpy as np
import torchaudio
import torch

class AudioDenoiser:
    def __init__(self,
                 device: torch.device = torch.device("mps"),
            ):
        self.device = device
        self.denoiser_model = pretrained.dns64().to(self.device)
        
    def denoise(self, 
                audio_path: str = None
            ):
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.denoiser_model.sample_rate, self.denoiser_model.chin).to(self.device)
        with torch.no_grad():
            wav_denoised = self.denoiser_model(wav)

        return wav_denoised.cpu().numpy(), sr
