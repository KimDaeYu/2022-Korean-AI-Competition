import torch
import queue
import os
import random
import warnings
import time
import json
import argparse
from glob import glob
import numpy as np
from datasets import load_metric

from modules.preprocess import preprocessing, remove_special_characters
from modules.trainer import trainer
from modules.utils import (
    get_optimizer,
    get_criterion,
    get_lr_scheduler,
)
from modules.audio import (
    FilterBankConfig,
    MelSpectrogramConfig,
    MfccConfig,
    SpectrogramConfig,
)
from modules.model import build_model
from modules.data import load_dataset, DataCollatorCTCWithPadding
from modules.utils import Optimizer
from modules.metrics import get_metric
from modules.inference import single_infer
from modules.vocab import make_wav2vec_vocab

from modules.audio.core import speech_file_to_array_fn, resample

from torch.utils.data import DataLoader

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import TrainingArguments, Trainer

import nsml
from nsml import DATASET_PATH
#DATASET_PATH="../data/t2-conf"

def bind_model(model,processor,trainer, optimizer=None):
    def save(path, *args, **kwargs):
        print("save!!! " + path)
        
        trainer.save_model(path)
        print('Model saved')

    def load(path, *args, **kwargs):
        print("load!!! " + path)
        processor.from_pretrained(path)
        model.from_pretrained(path)
        print('Model loaded')

    # 추론
    def infer(path, **kwargs):
        return inference(path, model, processor)

    nsml.bind(save=save, load=load, infer=infer)  # 'nsml.bind' function must be called at the end.

def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["target_text"]).input_ids
    return batch

cer_metric = load_metric('cer')
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    #ref = [pred.replace(' ', '') for pred in pred_str] 
    #hyp = [label.replace(' ', '') for label in label_str]
    print("pred: ",pred_str)
    print("ref : ",label_str)

    cer = cer_metric.compute(references=label_str, predictions=pred_str)

    return {"cer": cer}

def inference(path, model, **kwargs):
    model.eval()

    results = []
    for i in glob(os.path.join(path, '*')):
        results.append(
            {
                'filename': i.split('/')[-1],
                'text': single_infer(model, i, processor)[0]
            }
        )
    return sorted(results, key=lambda x: x['filename'])



if __name__ == '__main__':

    args = argparse.ArgumentParser()

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    args.add_argument('--use_cuda', type=bool, default=True)
    args.add_argument('--seed', type=int, default=777)
    args.add_argument('--num_epochs', type=int, default=10)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--save_result_every', type=int, default=10)
    args.add_argument('--checkpoint_every', type=int, default=1)
    args.add_argument('--print_every', type=int, default=50)
    args.add_argument('--dataset', type=str, default='kspon')
    args.add_argument('--output_unit', type=str, default='character')
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--num_threads', type=int, default=16)
    args.add_argument('--init_lr', type=float, default=1e-06)
    args.add_argument('--final_lr', type=float, default=1e-06)
    args.add_argument('--peak_lr', type=float, default=1e-04)
    args.add_argument('--init_lr_scale', type=float, default=1e-02)
    args.add_argument('--final_lr_scale', type=float, default=5e-02)
    args.add_argument('--max_grad_norm', type=int, default=400)
    args.add_argument('--warmup_steps', type=int, default=1000)
    args.add_argument('--weight_decay', type=float, default=1e-05)
    args.add_argument('--reduction', type=str, default='mean')
    args.add_argument('--optimizer', type=str, default='adam')
    args.add_argument('--lr_scheduler', type=str, default='tri_stage_lr_scheduler')
    args.add_argument('--total_steps', type=int, default=200000)

    args.add_argument('--architecture', type=str, default='deepspeech2')
    args.add_argument('--use_bidirectional', type=bool, default=True)
    args.add_argument('--dropout', type=float, default=3e-01)
    args.add_argument('--num_encoder_layers', type=int, default=3)
    args.add_argument('--hidden_dim', type=int, default=1024)
    args.add_argument('--rnn_type', type=str, default='gru')
    args.add_argument('--max_len', type=int, default=400)
    args.add_argument('--activation', type=str, default='hardtanh')
    args.add_argument('--teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--teacher_forcing_step', type=float, default=0.0)
    args.add_argument('--min_teacher_forcing_ratio', type=float, default=1.0)
    args.add_argument('--joint_ctc_attention', type=bool, default=False)

    args.add_argument('--audio_extension', type=str, default='pcm')
    args.add_argument('--transform_method', type=str, default='fbank')
    args.add_argument('--feature_extract_by', type=str, default='kaldi')
    args.add_argument('--sample_rate', type=int, default=16000)
    args.add_argument('--frame_length', type=int, default=20)
    args.add_argument('--frame_shift', type=int, default=10)
    args.add_argument('--n_mels', type=int, default=80)
    args.add_argument('--freq_mask_para', type=int, default=18)
    args.add_argument('--time_mask_num', type=int, default=4)
    args.add_argument('--freq_mask_num', type=int, default=2)
    args.add_argument('--normalize', type=bool, default=True)
    args.add_argument('--del_silence', type=bool, default=True)
    args.add_argument('--spec_augment', type=bool, default=True)
    args.add_argument('--input_reverse', type=bool, default=False)

    config = args.parse_args()
    warnings.filterwarnings('ignore')

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = 'cuda' if config.use_cuda == True else 'cpu'
    if hasattr(config, "num_threads") and int(config.num_threads) > 0:
        torch.set_num_threads(config.num_threads)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        config.dataset_path = os.path.join(DATASET_PATH, 'train', 'train_data')
        label_path = os.path.join(DATASET_PATH, 'train', 'train_label')
        preprocessing(label_path, os.getcwd())
        dataset = load_dataset(os.path.join(os.getcwd(), 'transcripts.txt'))
        
        train_dataset = dataset['train']
        test_dataset = dataset['test']
        train_dataset = train_dataset.map(remove_special_characters)
        test_dataset = test_dataset.map(remove_special_characters)
        # make vocab
        make_wav2vec_vocab(train_dataset, test_dataset)
        
        # tokenizer & feature_extractor & processor
        tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        model = build_model(config, processor)
        

        train_dataset = train_dataset.map(speech_file_to_array_fn, remove_columns=train_dataset.column_names)
        test_dataset = test_dataset.map(speech_file_to_array_fn, remove_columns=test_dataset.column_names)
        #train_dataset = train_dataset.map(resample)
        #test_dataset = test_dataset.map(resample)
        train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, batch_size=8, num_proc=1, batched=True)
        test_dataset = test_dataset.map(prepare_dataset, remove_columns=test_dataset.column_names, batch_size=8, num_proc=1, batched=True)

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model = build_model(config, processor)
        model.freeze_feature_extractor()

        training_args = TrainingArguments(
            output_dir="container_1/ckpts/",
            logging_dir = "container_1/runs/",
            group_by_length=True,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=config.num_epochs,
            fp16=True,
            save_steps=10000,
            eval_steps=200,
            logging_steps=200,
            learning_rate=4e-4,
            warmup_steps=int(0.1*1320), #10%
            save_total_limit=2,
        )
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=processor.feature_extractor,
        )

        bind_model(model,processor,trainer)
        

        trainer.train()
        #trainer.save_model("container_1/wav2vec2-large-960h")
        # print("save!!! " + path)
        # state = {
        #     'model': model.state_dict(),
        #     #'optimizer': optimizer.state_dict()
        # }
        # torch.save(state, os.path.join(path, 'model.pt'))
        nsml.save(10)
        # 1. wav2vec2-large-xlsr-kn  #0.3191 cer

        print('[INFO] train process is done')




