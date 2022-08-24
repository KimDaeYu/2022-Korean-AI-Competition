import os
import numpy as np
import torchaudio
import librosa

import torch
import torch.nn as nn
from torch import Tensor

from modules.vocab import KoreanSpeechVocabulary
from modules.data import load_audio
from modules.model import DeepSpeech2


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)
    return torch.FloatTensor(feature).transpose(0, 1)



from datasets import Dataset

def single_infer(model, audio_path, processor):
    device = 'cuda'

    signal = np.memmap(audio_path, dtype='h', mode='r').astype('float32')
    test_dataset = Dataset.from_dict({"speech":[signal]})

    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    # inputs = processor(feature, sampling_rate=16_000, return_tensors="pt", padding=True)
    # logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    # predicted_ids = torch.argmax(logits, dim=-1)
    
    return processor.batch_decode(predicted_ids)