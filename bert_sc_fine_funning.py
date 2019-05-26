#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch, math, time, nltk
import collections as col
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from torch.utils import data
from torch.autograd import Variable
from pytorch_pretrained_bert import WordpieceTokenizer, BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#%%
class PosDataset(data.Dataset):
    def __init__(self, tagged_sents):
        sents, tags_li = [], [] # list of lists
        for sent in tagged_sents:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list
        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y), len(is_heads))
        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen

def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()
    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)
    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens


class Net(nn.Module):
    def __init__(self, vocab_size=None):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(1024, vocab_size)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(device)
        y = y.to(device)
        
        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]
        
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%10==0: # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))

def eval(model, iterator):
    model.eval()
    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("result", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("{} {} {}\n".format(w, t, p))
            fout.write("\n")
    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    acc = (y_true==y_pred).astype(np.int32).sum() / float(len(y_true))
    print("acc=%.3f"%acc)

def read(path):
    data=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.lower().split())
    return data

def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_s'.encode('utf-8'))
    parser.add_argument('--data_c'.encode('utf-8'))
    parser.add_argument('--data_label'.encode('utf-8'))
    return parser.parse_args()

#%%
if __name__=='__main__':
    args = args_init()     
    tags = ["<pad>"] + ["0"] + ["1"]
    tag2idx = {tag:idx for idx, tag in enumerate(tags)}
    idx2tag = {idx:tag for idx, tag in enumerate(tags)}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=False)
    
    data_s=read(args.data_s)
    data_c=read(args.data_c)
    data_label=read(args.label)
    
    data_with_label=[]
    for sent, label in zip(data_s, data_label):
        temp=[]
        for w, l in zip(sent, label):
            temp.append((w.decode('utf-8'), l.decode('utf-8')))
        data_with_label.append(temp)
    
    train_data=data_with_label[2000:]
    val_data=data_with_label[1000:2000]
    test_data=data_with_label[:1000]
    
    train_data_=[]
    test_data_=[]
    for e in train_data:
        if len(e)>=10 and len(e)<=60:
            train_data_.append(e)
    
    for e in test_data:
        if len(e)>=10 and len(e)<=60:
            test_data_.append(e)
    train_data=train_data_
    test_data=test_data_
        
    
    train_dataset = PosDataset(train_data)
    eval_dataset = PosDataset(test_data)
    model = Net(vocab_size=len(tag2idx))
    model.to(device)
    model = nn.DataParallel(model)
    
    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)
    
    test_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=8,
                                 shuffle=False,
                                 num_workers=1,
                                 collate_fn=pad)
    
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    train(model, train_iter, optimizer, criterion)
    eval(model, test_iter)
    