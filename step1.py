import sentencepiece as spm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--prefix', type=str, required=True)
    parser.add_argument('--vocab_size', type=int, required=True)
    return parser.parse_args()


def create_vocab(dataset_pth, prefix, vocab_size):
    """output: {prefix}.model {prefix}.vocab"""
    print(dataset_pth, prefix, vocab_size)
    spm.SentencePieceTrainer.train(
        f"--input={dataset_pth} --model_prefix={prefix}  --vocab_size={vocab_size} \
        --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
        --pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] \
        --user_defined_symbols=[CLS],[SEP],[MASK]")


def main():
    args = parse_args()
    create_vocab(args.dataset, args.prefix, args.vocab_size)


if __name__ == '__main__':
    main()
