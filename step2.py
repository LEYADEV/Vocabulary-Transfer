import os
import argparse
from tqdm import tqdm
from typing import List
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import RandomSampler
from transformers import BertForMaskedLM, BertConfig, AdamW, get_linear_schedule_with_warmup
import sentencepiece as spm

from data_utils import wikiADataset
from common import check_dir, mask_tokens

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--dataset_pth', type=str, default='./data/wiki/wiki_en.txt')
    parser.add_argument('--snp_path', type=str, default='./models/snp/wikiA8k.model')  #
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--save_path', type=str, default='./models/rawBert_rawTokenizer/')
    parser.add_argument('--model_name', type=str, default='Bert8_WikiData_tok8k')
    return parser.parse_args()


def collate(examples: List[torch.Tensor]):
    if sp.pad_id() is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=sp.pad_id())


def train_model(model, train_dataloader, optimizer, scheduler):
    print(args.save_path, args.save_path + f'{args.model_name}/')
    check_dir(args.save_path)
    check_dir(args.save_path + f'{args.model_name}/')

    model = model.train()
    model.zero_grad()

    tr_loss = []
    for batch in tqdm(train_dataloader):
        inputs, labels = mask_tokens(batch)
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        outputs = model(inputs, masked_lm_labels=labels)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        tr_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        model.zero_grad()

    with open(args.save_path + f'{args.model_name}/train_loss.txt', 'w') as f:
        for item in tr_loss:
            f.write("%s\n" % item)

    model.save_pretrained(args.save_path + f'{args.model_name}')
    # torch.save(model, args.save_path + f'{args.model_name}.model')


def main():
    global args, sp
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    sp = spm.SentencePieceProcessor()
    sp.load(args.snp_path)

    wiki_dataset = wikiADataset(sp, args.dataset_pth)
    train_sampler = RandomSampler(wiki_dataset)

    train_dataloader = DataLoader(wiki_dataset, sampler=train_sampler, batch_size=64, collate_fn=collate)

    conf = BertConfig()
    conf.num_hidden_layers = 8
    conf.vocab_size = args.vocab_size

    model = BertForMaskedLM(conf)
    model = model.to('cuda')

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader))

    train_model(model, train_dataloader, optimizer, scheduler)


if __name__ == '__main__':
    main()
