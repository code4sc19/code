#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch
import numpy as np
from pytorch_pretrained_bert import WordpieceTokenizer, BertTokenizer, BertModel
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#conf = config.config()
#%%

class Text2Vec(object):
    def __init__(self):
        super(Text2Vec, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.wordpiece_tokenizer  = WordpieceTokenizer(vocab= self.tokenizer.vocab)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_word2vec=[]
        
        bert_word2vec_={}
        with open('bert-base-uncased.30522.768d.vec', 'r') as f:
            for e in f.readlines():
                self.bert_word2vec.append(e.split())
                
        for e in self.bert_word2vec[1:]:
            bert_word2vec_[e[0]]=np.array(e[1:]).astype('float32')
        self.bert_word2vec=bert_word2vec_
        
    def _subword(self, sent):
        index=0
        sub_ids=[]
        for w in sent:
            if w[:2]!='##':
                sub_ids.append(index)
                index+=1
            else:
                sub_ids.append(index-1)
        return sub_ids
    
    def _merge_emb(self, sub_ids, embs):
        assert len(sub_ids)==len(embs)
        d={}
        for i in range(sub_ids[-1]+1):
            d[i]=[]
        
        for index, emb in zip(sub_ids, embs):
            d[index].append(emb)
        
        merged_emb=[]
        for i in range(sub_ids[-1]+1):
            merged_emb.append(torch.mean(torch.stack(d[i], dim=0), dim=0))
        
        return merged_emb
        
    def text2vec(self, text, only_emb=False):
        tokenized_sent = self.wordpiece_tokenizer.tokenize(text.lower())
        subword_ids = self._subword(tokenized_sent)
        
        if only_emb:
            embs=[]
            for token in tokenized_sent:
                embs.append(self.bert_word2vec[token.encode('utf-8')])
            embs=np.array(embs)
            
            d={}
            for i in range(subword_ids[-1]+1):
                d[i]=[]
            
            for index, emb in zip(subword_ids, embs):
                d[index].append(emb)
            
            merged_emb=[]
            for i in range(subword_ids[-1]+1):
                merged_emb.append(np.mean(np.stack(d[i], axis=0), axis=0))
            
            embs = np.array(merged_emb)
            
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
            segments_ids = [0]*len(indexed_tokens)
            
            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            
            # Predict hidden states features for each layer
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)
            #s1_reps = torch.mean(encoded_layers[-2][0], 1)
            
            embs = encoded_layers[-2][0]
                
            embs = self._merge_emb(subword_ids, embs)
            embs = torch.stack(embs, dim=0).cpu().detach().numpy()
        return embs

