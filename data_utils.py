from torch.utils.data import DataLoader, Dataset
import torch
from typing import List
from tqdm import tqdm
import pandas as pd


class wikiADataset(Dataset):
    def __init__(self, sp, input_pth='./data/wiki/wiki_en.txt', block_size=256):
        with open(input_pth) as f:
            data = f.readlines()
        self.examples = []

        for row in tqdm(data):
            tokenized_text = sp.encode_as_ids(row)

            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i: i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class QuoraMLMDataset(Dataset):
    def __init__(self, sp, block_size = 256):
        df = pd.read_csv('./data/Quora/clear_quora.csv')
        df.question_text = df.question_text.fillna(" ")
        data = df['question_text'][:700000] #train
        self.examples = []
        self.target = []
        for row in tqdm(data):
            tokenized_text = sp.encode_as_ids(row)
            if len(tokenized_text) < 3:
                continue
            if len(tokenized_text) > block_size:
                for i in range(0, len(tokenized_text) - block_size + 1, block_size):
                    block = tokenized_text[i : i + block_size]
                    block = [4] + block + [5]
                    self.examples.append(block)
            else:
                self.examples.append([4] + tokenized_text + [5])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class QuoraCLFDataset(Dataset):
    def __init__(self, sp, train=False, dev=False, test=False):
        df = pd.read_csv('./data/Quora/clear_quora.csv')
        df.question_text = df.question_text.fillna(" ")
        if train:
            data = df['question_text'].values[:700000]
            labels = df['target'].values[:700000]
        if dev:
            data = df['question_text'].values[700000:1003061]
            labels = df['target'].values[700000:1003061]
        if test:
            data = df['question_text'].values[1003061:]
            labels = df['target'].values[1003061:]
        self.examples = []
        self.target = []
        i = 0
        for _ in tqdm(range(len(data))):
            row = data[i]
            tokenized_text = sp.encode_as_ids(row)
            if len(tokenized_text) > 412 or len(tokenized_text) < 3:
                i += 1
                continue
            self.examples.append([4] + tokenized_text + [5])
            self.target.append(labels[i])
            i += 1

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long), torch.tensor(int(self.target[i]))