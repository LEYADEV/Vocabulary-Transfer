# Transformer-vocabulary-transfer
Implementation of the paper "Fine-Tuning Transformers: Vocabulary Transfer"




## Description
step 1 - Create SentencePiece vocabulary for dataset  
step 2 - Train the first level model (BertForMaskedLM) on English Wikipedia from scratch  
step 3 - Match vocabulary (first level model dataset & downstream task dataset)  
step 4 - Transfer dictionary using mapping. Сreate folders and raw models for experiments.  
step 5 - Train 1 epoch BertForMaskedLM on downstream task  
step 6 - Train final (BertForSequenceClassification) downstream model  


## Citation
I. Samenko, A. Tikhonov, B. Kozlovsky, I. P. Yamshchikov. Fine-Tuning Transformers: Vocabulary Transfer. 


## Contact
*Igor Samenko: <i.samenko@gmail.com>
