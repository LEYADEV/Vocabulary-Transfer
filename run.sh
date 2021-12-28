#!/bin/bash

python3 step1.py --dataset=./data/Quora/clear_quora.csv --prefix=Quora8k --vocab_size=8000
python3 step2.py --dataset_pth=./data/wiki/wiki_en.txt --snp_path=./models/snp/wikiA8k.model --vocab_size=8000 --save_path=./models/rawBert_rawTokenizer/ --model_name='Bert8_WikiData_tok8k'

python3 step3.py --wiki_vocab=./models/snp/wikiA8k.vocab --task_vocab=./models/snp/Quora8k.vocab --out_vocab=./models/matcher/wiki_quora_match --matcher=1
python3 step3.py --wiki_vocab=./models/snp/wikiA8k.vocab --task_vocab=./models/snp/Quora8k.vocab --out_vocab=./models/matcher/wiki_quora_match --matcher=2

python3 step4.py --mapping_file_1=./models/matcher/wiki_quora_match_matcher_f1.tsv --mapping_file_2=./models/matcher/wiki_quora_match_matcher_f2.tsv --source_bert_model=./models/rawBert_rawTokenizer/Bert8_WikiData_tok8k/ --new_vocab_size=8000 --old_vocab_size=8000 --save_pht=./models/matched_models/quora_ex1/

EXP=('avg/' 'avgavg/' 'matched/' 'pretrain/' 'randomhead/' 'shifted/' 'shifted2/','random')

for exp in "${EXP[@]}"
do
  echo "Start $exp please wait ..."
  python3 step5.py --experiments_dir=quora_ex1 --experiment_folder="$exp" --sp_model_pth=./models/snp/Quora8k.model
done


for exp in "${EXP[@]}"
do
  echo "MLM + CLF model "
  echo "Start $exp please wait ..."
  python3 step6.py \
  --experiment_folder="$exp" \
  --sp_model_pth=./models/snp/Quora8k.model \
  --experiments_dir=./models/matched_models/quora_ex1_2mlm/ \
  --save_dir=./models/matched_models/quora_ex1_3clfmlm/
  --num_epoch=3
done

for exp in "${EXP[@]}"
do
  echo "Clf only model"
  echo "Start $exp please wait ..."
  python3 step6.py \
  --experiment_folder="$exp" \
  --sp_model_pth=./models/snp/Quora8k.model \
  --experiments_dir=./models/matched_models/quora_ex1/ \
  --save_dir=./models/matched_models/quora_ex1_3clf/
  --num_epoch=3
done