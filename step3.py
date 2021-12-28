import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wiki_vocab', type=str, default='./models/snp/wikiA8k.vocab')
    parser.add_argument('--task_vocab', type=str, default='./models/snp/Quora8k.vocab')
    parser.add_argument('--out_vocab', type=str, default="./models/matcher/wiki_quora_match")
    parser.add_argument('--matcher', type=int, default=1)
    return parser.parse_args()


def update_dict(vocab_pth):
    with open(vocab_pth, 'r') as f:
        output = f.readlines()

    o = [output[i] for i in range(0, 7)]
    idx = 7
    for row in output[7:]:
        s = row.split('\t')[0]
        s += "\t-" + str(idx) + '\n'
        o.append(s)
        idx += 1
    return o


def match(s, dict16):
    res = []
    if s in dict16: return [[s, ], ]
    for idx in range(len(s) - 1):
        if s[:idx] in dict16:
            pieces = match(s[idx:], dict16)
            if pieces:
                for p in pieces:
                    res.append([s[:idx], ] + p)
    return res


def matcher(wiki_voc, sent_voc):
    dict16 = dict()
    for line in wiki_voc:
        tok16, id16 = line.strip().split('\t')
        dict16[tok16] = id16
    out = []
    for line in sent_voc:
        tok64, id64 = line.strip().split('\t')
        if tok64 in dict16:
            out.append("\t".join(map(str, [id64, tok64, dict16[tok64], tok64])))
            pass
        else:
            r = match(tok64, dict16)
            if r:
                if args.matcher == 1:
                    optimal = list(sorted(r, key=lambda x: -max(map(len, x)) * 1000 + len(x)))[0]
                    toks = []
                    ids = []
                    for q in optimal:
                        toks.append(q)
                        ids.append(dict16[q])
                    out.append("\t".join(map(str, [id64, tok64, ','.join(ids), ','.join(toks)])))
                elif args.matcher == 2:
                    if len(r) > 1:
                        ml = min(map(len, r))
                        r = list(filter(lambda x: len(x) == ml, r))
                    toks = ";".join(map(lambda x: ",".join(x), r))
                    ids = ";".join(map(lambda x: ",".join(map(lambda a: dict16[a], x)), r))
                    out.append("\t".join(map(str, [id64, tok64, ids, toks])))
            else:
                out.append("\t".join(map(str, [id64, tok64, '--unknown--'])))

    with open(f'{args.out_vocab}_matcher_f{args.matcher}.tsv', 'w') as f:
        for item in out:
            f.write("%s\n" % item)


def main():
    global args
    args = parse_args()

    sent_voc = update_dict(args.task_vocab)
    wiki_voc = update_dict(args.wiki_vocab)
    matcher(wiki_voc, sent_voc)


if __name__ == '__main__':
    main()
