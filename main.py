#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import torch, bert_word2vec, argparse, Model, config, DepBias
import numpy as np
from pytorch_pretrained_bert import WordpieceTokenizer, BertTokenizer, BertModel
from masked_cross_entropy import compute_loss
conf = config.config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
#%% data preparetion
def read(path):
    data=[]
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line.lower().split())
    return data

def read_multi(path):
    data=[]
    temp=[]
    with open(path,'r') as f:
        for line in f.readlines():
            if line=='\n':
                data.append(temp)
                temp=[]
            else:
                temp.append(line.lower().split())
    return data

def batch_padded(embs_vec, label, dep_matrix):
    emb_dim=len(embs_vec[0][0])
    embs_vec1=[]
    label1=[]
    mask=[]
    max_len = max([len(e) for e in embs_vec])
    for e in embs_vec:
        embs_vec1.append(np.concatenate((e, np.zeros((max_len-len(e), emb_dim))), axis=0))
        mask.append(len(e)*[1]+(max_len-len(e))*[0])
    seq_len = [sum(e) for e in mask]
    
    for e in label:
        label1.append(e+(max_len-len(e))*[0])
    
    dep_matrix_padded=[]
    for e in dep_matrix:
        n=max_len-len(e)  
        dep_matrix_padded.append(np.pad(e, ((0,n),(0, n)),'constant',constant_values = (1e-9, 1e-9)))
    
    return FloatTensor(np.array(embs_vec1)), np.array(label1).astype('int32'), np.array(mask), np.array(seq_len), dep_matrix_padded
#%%
def test(encoder, test_x_vec, test_x_mask, test_label, test_dep, bz=10):
    with torch.no_grad():
        nb = len(test_x_vec)/bz 
        pred=[]
        pred1=[]
        att_list=[]
        encoder.eval()
        for k in range(nb):
            logits, att_by_layer = encoder(test_x_vec[k*bz:(k+1)*bz], test_x_mask[k*bz:(k+1)*bz], test_dep[k*bz:(k+1)*bz])
            pred.extend(logits.argmax(2).cpu().numpy().tolist())
            att_list.append(att_by_layer)
       
        lens=[sum(e) for e in test_x_mask]
        total=0
        correct=0
        for sent, pred_sent, l in zip(test_label, pred, lens):
            pred1.append(pred_sent[:l])
            total+=l
            for i in range(l):
                if int(sent[i])==int(pred_sent[i]):
                    correct+=1
        print correct, total
        micro_acc=correct/float(total)
        print "micro_acc: %f"%micro_acc
    return micro_acc, pred1, att_list

def toConll(string_doc, nlp):
   doc = nlp(string_doc)
   block = []
   for i, word in enumerate(doc):
          if word.head == word:
                  head_idx = 0
          else:
                  head_idx = word.head.i - doc[0].i + 1
          head_idx = str(head_idx)
          line = [str(i+1), head_idx, str(word), word.lemma_,  word.dep_, 
          		word.pos_, word.tag_, word.ent_type_]
          block.append(line)
   return [e[2] for e in block], [e[:2] for e in block]

def dep_matrix(args, sample=True):
    #training 
    train_s=read(args.train_s)
    train_label=read(args.train_l)
    train_dep=read_multi(args.train_dep)
    #validation
    val_s=read(args.val_s)
    val_label=read(args.val_label)
    val_dep=read_multi(args.val_dep)
    #test
    test_s=read(args.test_s)
    test_label=read(args.test_label)
    test_dep=read_multi(args.test_dep)
    
    num=10000 if sample else len(train_s) 
    
    train_dep_input=[]
    test_dep_input=[]
    for i, e in enumerate(train_dep):
        if i%10000==0:
            print "finished %d sentences"%i
        train_dep_input.append(DepBias._get_matrix(e))
    for e in test_dep:
        test_dep_input.append(DepBias._get_matrix(e))
    return train_s[:num], train_label[:num], train_dep_input[:num], \
            test_s, test_label, test_dep_input

def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_s'.encode('utf-8'))
    parser.add_argument('--train_l'.encode('utf-8'))
    parser.add_argument('--train_dep'.encode('utf-8'))
    
    parser.add_argument('--val_s'.encode('utf-8'))
    parser.add_argument('--val_l'.encode('utf-8'))
    parser.add_argument('--val_dep'.encode('utf-8'))
    
    parser.add_argument('--test_s'.encode('utf-8'))
    parser.add_argument('--test_l'.encode('utf-8'))
    parser.add_argument('--test_dep'.encode('utf-8'))
    return parser.parse_args()

#%%
if __name__=='__main__':
    args = args_init()     
    train_x, train_label, train_dep_input, test_x, test_label, test_dep_input= dep_matrix(args, sample=False)
    bert = bert_word2vec.Text2Vec()
    train_x_emb=[]
    test_x_emb=[]
    print "graduation", len(train_x_emb)+len(test_x_emb)
    for i, sent in enumerate(train_x):
        if i%1000==0:
            print "finished %d sentences"%i
        train_x_emb.append(bert.text2vec(" ".join(sent).decode('utf-8'), only_emb=True))      
    
    for i, sent in enumerate(test_x):
        if i%1000==0:
            print "finished %d sentences"%i
        test_x_emb.append(bert.text2vec(" ".join(sent).decode('utf-8'), only_emb=True))      
    
    test_x_emb_=[]
    test_label_=[]
    for x, y in zip(test_x_emb, test_label):
        if len(y)<=60:
 	    test_x_emb_.append(x)
	    test_label_.append(y)
    test_x_emb=test_x_emb_
    test_label=test_label_
    print len(test_label)
    test_x_vec, test_y, test_x_mask, test_seq_len, test_dep_matrix = batch_padded(test_x_emb, test_label, test_dep_input)
    
    #%%
    bz = conf.batch_size
    nb = len(train_x_emb)/bz
    encoder=Model.Encoder(emb_dim=conf.emb_dim, heads=conf.heads, num_layers=conf.num_layers)
    if USE_CUDA:
        encoder=encoder.cuda()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=conf.lr, betas=(0.9, 0.98), eps=1e-9)

    total_loss = 0
    for epoch in range(200):
        train_data = zip(train_x_emb, train_label, train_dep_input)
        np.random.shuffle(train_data)
        train_x_emb, train_label, train_dep_input = zip(*train_data)
        
        for b_i in range(nb):       
            train_x_vec, train_y, train_x_mask, train_seq_len, train_dep_matrix = \
            batch_padded(train_x_emb[b_i*bz:(b_i+1)*bz], train_label[b_i*bz:(b_i+1)*bz], train_dep_input[b_i*bz:(b_i+1)*bz])
            optimizer.zero_grad()
            logits, att_list = encoder(train_x_vec, train_x_mask, train_dep_matrix)
            pred = logits.argmax(2)
            loss = compute_loss(logits, LongTensor(train_y), LongTensor(train_seq_len))    
            if b_i%30==0:
                acc, test_pred, att_list=test(encoder, test_x_vec, test_x_mask, test_label, test_dep_matrix)
                print "acc: %f"%acc
                print "lr:%f_num_layers:%d_bz:%d_epoch:%d_nb:%d_loss:%f"%(conf.lr, conf.num_layers, conf.batch_size, epoch, b_i, loss)
            total_loss = 0
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.data
