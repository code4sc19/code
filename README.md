## code
Structure biased self-attention network (# Anonymized #)
*Requirements*
- en_core_web_lg (download by running "python -m spacy download en_core_web_lg") 
- bert-base-uncased ([download here](https://github.com/google-research/bert))
- bert-base-uncased.30522.768d.vec (download and put it in the main directory.)

### Training and test
- To train the model, please prepare sentence compression data by following the intruction [here](https://github.com/code4sc19/data)
- Then, runing the following:
`python train.py --train_s==train_s.txt --train_l==train_l --val_s==val_s.txt --val_l==val_l --test_s==test_s.txt --test_l==test_l`
                 
## State-of-the-art-result Reproduction
