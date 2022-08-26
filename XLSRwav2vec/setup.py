#nsml: pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
from distutils.core import setup

setup(
    name='kospeech_wav2vec',
    version='latest',
    install_requires=[
        #'torch==1.7.0',
        'librosa',
        'numpy',
        'pandas',
        'tqdm',
        'matplotlib',
        'astropy',
        'sentencepiece',
        #'torchaudio',
        'pydub',
        'glob2',
        'omegaconf',
        "datasets",
        "transformers",
        "jiwer",
        "sklearn"
        #"python-Levenshtein"
        # "python-Levenshtein-wheels",
        
    ],
)

