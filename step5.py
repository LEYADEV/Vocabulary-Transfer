import argparse
import os
from typing import List
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForMaskedLM
from transformers import get_linear_schedule_with_warmup, AdamW
import sentencepiece as spm

from data_utils import QuoraMLMDataset
from common import check_dir, mask_tokens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default="0")
    parser.add_argument('--experiment_folder', required=True)  # default='random/'
    parser.add_argument('--sp_model_pth', default='./models/snp/Quora8k.model')
    parser.add_argument('--experiments_dir', default='quora_ex1')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    ROOT_DIR = './output/models/matched_models/'
    PRETRAIN_MODEL_PTH = os.path.join(ROOT_DIR, args.experiments_dir, args.experiment_folder)
    SAVE_MODEL_PHT = os.path.join(ROOT_DIR, args.experiments_dir+"_2mlm", args.experiment_folder)
    check_dir(os.path.join(ROOT_DIR, args.experiments_dir+"_2mlm"))
    check_dir(SAVE_MODEL_PHT)

    model = BertForMaskedLM.from_pretrained(PRETRAIN_MODEL_PTH)
    model.to(device='cuda')
    print('model load', PRETRAIN_MODEL_PTH)

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model_pth)
    print('SentencePiece size', sp.get_piece_size(), args.sp_model_pth)

    def collate(examples: List[torch.Tensor]):
        if sp.pad_id() is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=sp.pad_id())

    dataset = QuoraMLMDataset(sp)
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=64, collate_fn=collate)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)

    t_total = len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)

    model.train()
    model.zero_grad()
    tr_loss = []
    for batch in tqdm(train_dataloader):
        inputs, labels = mask_tokens(batch)
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        outputs = model(inputs, masked_lm_labels=labels)
        loss = outputs[0]

        loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        tr_loss.append(loss.item())
        optimizer.step();
        scheduler.step()
        model.zero_grad();

    with open(SAVE_MODEL_PHT + '/step_mlm_batch_loss.txt', 'w') as f:
        for item in tr_loss:
            f.write("%s\n" % item)

    model.save_pretrained(SAVE_MODEL_PHT)


if __name__ == '__main__':
    main()
