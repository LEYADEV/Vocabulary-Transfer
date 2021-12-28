import argparse
import os
import torch
from tqdm import tqdm
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification
from transformers import AdamW

from common import check_dir
from data_utils import QuoraCLFDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gpu', default='0')
    parser.add_argument('--experiment_folder', required=True)  # default='random/'
    parser.add_argument('--sp_model_pth', default='./models/snp/Quora8k.model')
    parser.add_argument('--experiments_dir', default='./models/matched_models/quora_ex1/')
    parser.add_argument('--save_dir', type='str', default='./models/matched_models/quora_ex1_3clf/')
    parser.add_argument('--num_epoch', type=int, default=15)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu

    check_dir(args.save_dir)
    model_pth = os.path.join(args.experiments_dir, args.experiment_folder)
    model = BertForSequenceClassification.from_pretrained(model_pth)
    model.to(device='cuda')

    print(f"model load: {model_pth}")

    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model_pth)
    print('SentencePiece size', sp.get_piece_size())

    BPEDataset_train = QuoraCLFDataset(sp, train=True)
    BPEDataset_dev = QuoraCLFDataset(sp, dev=True)
    BPEDataset_test = QuoraCLFDataset(sp, test=True)

    def collate(examples):
        result_input, result_labels = [], []
        for inputs, labels in examples:
            result_input.append(inputs)
            result_labels.append(labels)
        result_input = pad_sequence(result_input, batch_first=True, padding_value=sp.pad_id())
        return result_input, torch.tensor(result_labels)

    train_dataloader = DataLoader(BPEDataset_train, batch_size=128, collate_fn=collate, shuffle=True)
    dev_dataloader = DataLoader(BPEDataset_dev, batch_size=128, collate_fn=collate, shuffle=True)
    test_dataloader = DataLoader(BPEDataset_test, batch_size=128, collate_fn=collate, shuffle=True)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)

    model.zero_grad()
    epoch_train_acc = []
    epoch_train_loss = []
    train_loss = []

    epoch_test_acc = []
    epoch_test_loss = []
    test_loss = []

    epoch_dev_acc = []
    epoch_dev_loss = []
    dev_loss = []

    epoch_couter = 0
    for _ in range(args.num_epoch):
        print('epoch', epoch_couter)
        epoch_couter += 1
        model.train()
        tmp_epoch_train_loss = 0
        tmp_epoch_test_loss = 0

        tmp_epoch_train_acc = 0
        tmp_epoch_test_acc = 0

        tmp_epoch_dev_loss = 0
        tmp_epoch_dev_acc = 0

        for batch in tqdm(train_dataloader):
            inputs, labels = batch
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            outputs = model(inputs, labels=labels)
            loss = outputs[0]
            tmp_epoch_train_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()

            loss.backward();
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            train_loss.append(loss.item())
            tmp_epoch_train_loss += loss.item()

            optimizer.step();
            model.zero_grad();

        model.eval()
        with torch.no_grad():
            for batch in tqdm(dev_dataloader):
                inputs, labels = batch
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs, labels=labels)
                loss = outputs[0]
                tmp_epoch_dev_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()
                dev_loss.append(loss.item())
                tmp_epoch_dev_loss += loss.item()

            for batch in tqdm(test_dataloader):
                inputs, labels = batch
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = model(inputs, labels=labels)
                loss = outputs[0]
                tmp_epoch_test_acc += (torch.argmax(outputs[1], dim=1) == labels).sum().data.cpu().numpy()
                test_loss.append(loss.item())
                tmp_epoch_test_loss += loss.item()

        epoch_train_acc.append(100 * float(tmp_epoch_train_acc) / float(len(BPEDataset_train)))
        epoch_train_loss.append(tmp_epoch_train_loss / len(train_dataloader))

        epoch_test_acc.append(100 * float(tmp_epoch_test_acc) / float(len(BPEDataset_test)))
        epoch_test_loss.append(tmp_epoch_test_loss / len(test_dataloader))

        epoch_dev_acc.append(100 * float(tmp_epoch_dev_acc) / float(len(BPEDataset_dev)))
        epoch_dev_loss.append(tmp_epoch_dev_loss / len(dev_dataloader))

        stage = 'stage3' if 'mlm' in args.save_dir else 'stage2'
        check_dir(os.path.join(args.save_dir, args.experiment_folder))

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_train_loss.txt'), 'w') as f:
            for item in train_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_test_loss.txt'), 'w') as f:
            for item in test_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_batch_dev_loss.txt'), 'w') as f:
            for item in dev_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_train_loss.txt'), 'w') as f:
            for item in epoch_train_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_test_loss.txt'), 'w') as f:
            for item in epoch_test_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_dev_loss.txt'), 'w') as f:
            for item in epoch_dev_loss:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_test_acc.txt'), 'w') as f:
            for item in epoch_test_acc:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_train_acc.txt'), 'w') as f:
            for item in epoch_train_acc:
                f.write("%s\n" % item)

        with open(os.path.join(args.save_dir, args.experiment_folder, stage+'_clf_epoch_dev_acc.txt'), 'w') as f:
            for item in epoch_dev_acc:
                f.write("%s\n" % item)


if __name__ == '__main__':
    main()
