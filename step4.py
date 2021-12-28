import argparse
import torch
from transformers import BertConfig, BertForSequenceClassification

from common import check_dir


def get_pretrained_model(model_dir):
    return BertForSequenceClassification.from_pretrained(model_dir)


def get_random_embeds(config):
    random_model = BertForSequenceClassification(config)
    random_embeds = random_model.bert.embeddings.word_embeddings.weight
    return random_embeds


def get_mapping_matrices(mapping_file, new_vocab_size, old_vocab_size, embedding_dim=768, use_bad_shift=False,
                         use_one_to_one=False):
    mapping_matrix = torch.zeros((new_vocab_size, old_vocab_size))
    mask_matrix = torch.zeros((new_vocab_size, embedding_dim))

    with open(mapping_file) as f:
        for _ in range(7):
            f.readline()  # skip first 7 lines

        for line in f:
            fields = line.rstrip().split('\t')
            if fields[-1] == '--unknown--':
                continue
            # only transfer wholesome old token embeds if use_one_to_one
            if ',' in fields[2] and use_one_to_one:
                continue

            new_idx = -int(fields[0])
            mask_matrix[new_idx, :] = 1

            old_ids_variants = fields[2].split(';')
            denominator = float(len(old_ids_variants))

            for old_ids in old_ids_variants:
                old_ids = old_ids.split(',')

                # replace embeds with bad ones if use_bad_shift
                shift = 3 if use_bad_shift and len(old_ids) > 1 else 0

                old_ids = [min(old_vocab_size, -int(idx) + shift) for idx in old_ids]
                local_denominator = len(old_ids)
                for idx in old_ids:
                    mapping_matrix[new_idx, idx] += 1. / local_denominator

            mapping_matrix[new_idx, :] /= denominator
    return (mapping_matrix, mask_matrix)


def create_experiment(mapping_file, old_vocab_size, new_vocab_size, use_bad_shift, use_one_to_one, save_pht, name):
    conf = BertConfig()
    conf.vocab_size = new_vocab_size
    conf.num_hidden_layers = 8
    random_embeds = get_random_embeds(conf)

    model = get_pretrained_model(args.source_bert_model);
    mapping_matrix, mask_matrix = get_mapping_matrices(mapping_file,
                                                       new_vocab_size,
                                                       old_vocab_size,
                                                       use_bad_shift=use_bad_shift,
                                                       use_one_to_one=use_one_to_one)

    new_embeds = mapping_matrix.matmul(model.bert.embeddings.word_embeddings.weight)
    new_embed_matrix = (1. - mask_matrix) * random_embeds + new_embeds
    model.bert.embeddings.word_embeddings = torch.nn.Embedding.from_pretrained(new_embed_matrix, freeze=False)
    check_dir(save_pht + name)
    model.save_pretrained(save_pht + name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file_1', type=str, default='./models/matcher/wiki_quora_match_matcher_f1.tsv')
    parser.add_argument('--mapping_file_2', type=str, default='./models/matcher/wiki_quora_match_matcher_f2.tsv')
    parser.add_argument('--source_bert_model', type=str, default='./models/rawBert_rawTokenizer/Bert8_WikiData_tok8k/')
    parser.add_argument('--new_vocab_size', type=int, default=8000)
    parser.add_argument('--old_vocab_size', type=int, default=8000)

    parser.add_argument('--save_pht', type=str, default='./models/matched_models/ex1/')

    parser.add_argument('--use_random_embeds', type=bool, default=False)
    parser.add_argument('--use_bad_shift', type=bool, default=False)
    parser.add_argument('--use_one_to_one', type=bool, default=False)

    return parser.parse_args()


def main():
    global args
    args = parse_args()

    check_dir(args.save_pht)

    create_experiment(args.mapping_file_2, args.old_vocab_size, args.new_vocab_size, use_one_to_one=False,
                      use_bad_shift=False, save_pht=args.save_pht, name='avg')

    create_experiment(args.mapping_file_2, args.old_vocab_size, args.new_vocab_size, use_one_to_one=False,
                      use_bad_shift=False, save_pht=args.save_pht, name='avgavg')

    create_experiment(args.mapping_file_1, args.old_vocab_size, args.new_vocab_size, use_one_to_one=True,
                      use_bad_shift=False, save_pht=args.save_pht, name='matched')

    create_experiment(args.mapping_file_1, args.old_vocab_size, args.new_vocab_size, use_one_to_one=False,
                      use_bad_shift=True, save_pht=args.save_pht, name='shifted')

    create_experiment(args.mapping_file_2, args.old_vocab_size, args.new_vocab_size, use_one_to_one=False,
                      use_bad_shift=True, save_pht=args.save_pht, name='shifted2')

    conf = BertConfig()
    conf.vocab_size = args.new_vocab_size
    conf.num_hidden_layers = 8

    model = get_pretrained_model(args.source_bert_model);
    check_dir(args.save_pht + 'pretrain')
    model.save_pretrained(args.save_pht + 'pretrain')

    random_embeds = get_random_embeds(conf)
    model.bert.embeddings.word_embeddings = torch.nn.Embedding.from_pretrained(random_embeds, freeze=False)
    check_dir(args.save_pht + 'randomhead')
    model.save_pretrained(args.save_pht + 'randomhead')

    random_model = BertForSequenceClassification(conf)
    check_dir(args.save_pht + 'random')
    random_model.save_pretrained(args.save_pht + 'random')


if __name__ == '__main__':
    main()
