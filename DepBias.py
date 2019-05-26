#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

def get_matrix(sent_dep, m_type='undirectional_all_nodes'):
    l=len(sent_dep)
    dep_sqr = np.array([[-10000 for col in range(l)] for row in range(l)]).astype('float32')     
    if m_type=='undirectional_all_nodes':
        parent={}
        for e in sent_dep:
            curr=int(e[0])-1
            nxt =int(e[1])-1
            parent[curr]=nxt
        
        for key in parent.keys():
            if parent[key]==-1:
                parent[key]=key
                
        #find all children
        children={}
        for i in range(len(sent_dep)):
            children[i]=[]
            
        for e in sent_dep:
            col = int(e[1])-1
            if col==-1:
                continue
            children[col].append(int(e[0])-1)
        
        neighbor={}
        for i in range(len(sent_dep)):
            neighbor[i]=[parent[i]]+children[i]
            
        for i, row in enumerate(dep_sqr):
            rank={}
            for j in range(len(sent_dep)):
                rank[j+1]=[]
            
            memory=[0]*len(sent_dep)
            rank_nb=0
            pool=neighbor[i]
            prev=-1
            while prev<sum(memory):
                prev=sum(memory)
                if rank_nb>1000:
                    break
                    
                rank_nb+=1
                temp=[]
                for node in pool:
                    if memory[node]==0:
                        rank[rank_nb].append(node)
                        memory[node]=1
                    
                    temp+=neighbor[node]
                pool=set(temp)
                
            level_nb = 0
            for key in rank.keys():
                if rank[key]!=[]:
                    level_nb+=1

            for key in rank.keys():
                if rank[key]!=[]:
                    for node in rank[key]:
                        row[node]=-key**2 / float(level_nb**2)

    return dep_sqr