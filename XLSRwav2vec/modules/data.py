import sys, os
import threading
import torch
import torch.nn as nn
import numpy as np
#import torchaudio
import random
import math
import pydub

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import Dataset

from sklearn.model_selection import train_test_split

from modules.vocab import Vocabulary
#from modules.audio.core import load_audio
from modules.audio.parser import SpectrogramParser

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor

class SpectrogramDataset(Dataset, SpectrogramParser):
    """
    Dataset for feature & transcript matching

    Args:
        audio_paths (list): list of audio path
        transcripts (list): list of transcript
        sos_id (int): identification of <start of sequence>
        eos_id (int): identification of <end of sequence>
        spec_augment (bool): flag indication whether to use spec-augmentation or not (default: True)
        config (DictConfig): set of configurations
        dataset_path (str): path of dataset
    """

    def __init__(
            self,
            audio_paths: list,  # list of audio paths
            transcripts: list,  # list of transcript paths
            sos_id: int,  # identification of start of sequence token
            eos_id: int,  # identification of end of sequence token
            config,  # set of arguments
            spec_augment: bool = False,  # flag indication whether to use spec-augmentation of not
            dataset_path: str = None,  # path of dataset,
            audio_extension: str = 'pcm'  # audio extension
    ) -> None:
        super(SpectrogramDataset, self).__init__(
            feature_extract_by=config.feature_extract_by, sample_rate=config.sample_rate,
            n_mels=config.n_mels, frame_length=config.frame_length, frame_shift=config.frame_shift,
            del_silence=config.del_silence, input_reverse=config.input_reverse,
            normalize=config.normalize, freq_mask_para=config.freq_mask_para,
            time_mask_num=config.time_mask_num, freq_mask_num=config.freq_mask_num,
            sos_id=sos_id, eos_id=eos_id, dataset_path=dataset_path, transform_method=config.transform_method,
            audio_extension=audio_extension
        )
        self.audio_paths = list(audio_paths)
        self.transcripts = list(transcripts)
        self.augment_methods = [self.VANILLA] * len(self.audio_paths)
        self.dataset_size = len(self.audio_paths)
        self._augment(spec_augment)
        self.shuffle()

    def __getitem__(self, idx):
        """ get feature vector & transcript """
        feature = self.parse_audio(os.path.join(self.dataset_path, self.audio_paths[idx]), self.augment_methods[idx])

        if feature is None:
            return None

        transcript, status = self.parse_transcript(self.transcripts[idx])

        if status == 'err':
            print(self.transcripts[idx])
            print(idx)
        return feature, transcript

    def parse_transcript(self, transcript):
        """ Parses transcript """
        tokens = transcript.split(' ')
        transcript = list()

        transcript.append(int(self.sos_id))
        for token in tokens:
            try:
                transcript.append(int(token))
                status='nor'
            except:
                print(tokens)
                status='err'
        transcript.append(int(self.eos_id))

        return transcript, status

    def _augment(self, spec_augment):
        """ Spec Augmentation """
        if spec_augment:
            print("Applying Spec Augmentation...")

            for idx in range(self.dataset_size):
                self.augment_methods.append(self.SPEC_AUGMENT)
                self.audio_paths.append(self.audio_paths[idx])
                self.transcripts.append(self.transcripts[idx])

    def shuffle(self):
        """ Shuffle dataset """
        tmp = list(zip(self.audio_paths, self.transcripts, self.augment_methods))
        random.shuffle(tmp)
        self.audio_paths, self.transcripts, self.augment_methods = zip(*tmp)

    def __len__(self):
        return len(self.audio_paths)

    def count(self):
        return len(self.audio_paths)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    #signal = load_audio(audio_path, del_silence, extension=audio_extension)
    # feature = torchaudio.compliance.kaldi.fbank(
    #     waveform=Tensor(signal).unsqueeze(0),
    #     num_mel_bins=80,
    #     frame_length=20,
    #     frame_shift=10,
    #     window_type='hamming'
    # ).transpose(0, 1).numpy()

    feature = np.mean(np.zeros(10,10))
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)


def load_dataset(transcripts_path):
    """
    Provides dictionary of filename and labels

    Args:
        transcripts_path (str): path of transcripts

    Returns: target_dict
        - **target_dict** (dict): dictionary of filename and labels
    """
    audio_paths = list()
    korean_transcripts = list()
    transcripts = list()

    with open(transcripts_path) as f:
        for idx, line in enumerate(f.readlines()):
            try:
                audio_path, korean_transcript, transcript = line.split('\t')
            except:
                print(line)
            transcript = transcript.replace('\n', '')

            audio_paths.append(audio_path)
            korean_transcripts.append(korean_transcript)
            transcripts.append(transcript)

    data = Dataset.from_dict({"path":audio_paths,"text":korean_transcripts})
    data = data.shuffle(seed=42)
    #dataset = data.train_test_split(test_size=0.2,seed=42)

    return data

def collate_fn(batch):
    pad_id = 0
    """ functions that pad to the maximum sequence length """

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    try:
        batch = [i for i in batch if i != None]
        batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

        seq_lengths = [len(s[0]) for s in batch]
        target_lengths = [len(s[1]) - 1 for s in batch]

        max_seq_sample = max(batch, key=seq_length_)[0]
        max_target_sample = max(batch, key=target_length_)[1]

        max_seq_size = max_seq_sample.size(0)
        max_target_size = len(max_target_sample)

        feat_size = max_seq_sample.size(1)
        batch_size = len(batch)

        seqs = torch.zeros(batch_size, max_seq_size, feat_size)

        targets = torch.zeros(batch_size, max_target_size).to(torch.long)
        targets.fill_(pad_id)

        for x in range(batch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(0)

            seqs[x].narrow(0, 0, seq_length).copy_(tensor)
            targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

        seq_lengths = torch.IntTensor(seq_lengths)
        return seqs, targets, seq_lengths, target_lengths
    except Exception as e:
        print(e)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch