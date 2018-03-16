import numpy as np
from polyglot.mapping import Embedding
import pickle
from pos_helper import *
from nltk import pos_tag


src_embeddings = Embedding.load("/home/krishna/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2")
tar_embeddings = Embedding.load("/home/krishna/polyglot_data/embeddings2/de/embeddings_pkl.tar.bz2")
def make_align_dict(inp,nwords):
    inplist = inp.split()
    aldict={}
    for j in range(nwords):
        aldict[j] = []
    for j in inplist:
        a,b = j.split('-')
        a,b = int(a),int(b)
        if b not in aldict:
            aldict[b] = []
        aldict[b].append(a)
    return aldict

def get_target_embedding(ind,inlist):
    try:
        e2 = tar_embeddings[inlist[ind]]
    except:
        e2 = tar_embeddings['<UNK>']
    if ind==0:
        e1 = tar_embeddings['<S>']
    else:
        try:
            e1 = tar_embeddings[inlist[ind-1]]
        except:
            e1 = tar_embeddings['<UNK>']
    if ind==len(inlist)-1:
        e3 = tar_embeddings['</S>']
    else:
        try: 
            e3 = tar_embeddings[inlist[ind+1]]
        except:
            e3 = tar_embeddings['<UNK>']
    return np.concatenate((e1,e2,e3),axis=0)

def get_source_embedding(ind,tar_list,inlist,adct):
    e1 = np.zeros(64,dtype=float)
    e2 = np.zeros(64,dtype=float)
    e3 = np.zeros(64,dtype=float)
    #No alignment
    if len(adct[ind])==0:
        e2 = src_embeddings['<UNK>']
    else:
        for l in adct[ind]:
            try:
                e2 += src_embeddings[inlist[l]]
            except:
                e2 += src_embeddings['<UNK>']
        e2 = e2/len(adct[ind])
    if ind==0 or len(adct[ind-1])==0 :
        e1 = src_embeddings['<S>']
    else:
        for l in adct[ind-1]:
            try:
                e1 += src_embeddings[inlist[l]]
            except:
                e1 += src_embeddings['<UNK>']
        e1 = e1/len(adct[ind-1])
    if ind==len(tar_list)-1 or len(adct[ind+1])==0:
        e3 = src_embeddings['</S>']
    else:
        for l in adct[ind+1]:
            try:
                e3 += src_embeddings[inlist[l]]
            except:
                e3 += src_embeddings['<UNK>']
        e3 = e3/len(adct[ind+1])
    return np.concatenate((e1,e2,e3),axis=0)

def get_source_pos(ind,tar_list,inlist,adct,pdict):
    if ind<0 or ind>= len(tar_list):
        return pdict['NIL']
    if len(adct[ind]) == 0:  #If no alignment give none.
        return pdict['NIL']
    else:  #Incase of alignment giving postag of first src.
        wd,ptag = pos_tag([inlist[adct[ind][0]]])[0]
        if ptag in pdict:
            return pdict[ptag]
        else:
            return pdict['NIL']

def get_target_pos(ind,tlist,tagger,pdict):
    if ind<0 or ind>= len(tlist):
        return pdict['NIL']
    else:
        wd,ptag = tagger.tag([tlist[ind]])[0]
        print ptag
        if ptag in pdict:
            print "The token ",pdict[ptag]
            return pdict[ptag]
        else:
            print "The dict has failed."
            return pdict['NIL']

def get_sentense_inputs(sent,gpt,gpd):
    tl = sent[3].split()
    l1 = np.array([1 if x=='OK' else 0 for x in tl])
    l2 = np.array([1 if x==0 else 0 for x in l1])
    labels = np.transpose(np.stack([l1,l2]))
    embedlist=[]
    spl1=[]
    spl2=[]
    spl3=[]
    tpl1=[]
    tpl2=[]
    tpl3=[]
    sc_words = sent[1].split()
    tr_words = sent[0].split()
    #gives a dict key-> source index value list of aligned words.
    align_dict = make_align_dict(sent[2],len(tr_words))
        
    for j in range(len(tr_words)):
        target_embed = get_target_embedding(j,tr_words)
        source_embed = get_source_embedding(j,tr_words,sc_words,align_dict)
        #The POS tags are initialised randomly in embedding layer.
        pdct = gen_pos_index()
        sp1 = get_source_pos(j-1,tr_words,sc_words,align_dict,pdct)
        sp2 = get_source_pos(j,tr_words,sc_words,align_dict,pdct)
        sp3 = get_source_pos(j+1,tr_words,sc_words,align_dict,pdct)
        tp1 = get_target_pos(j-1,tr_words,gpt,gpd)
        tp2 = get_target_pos(j,tr_words,gpt,gpd)
        tp3 = get_target_pos(j+1,tr_words,gpt,gpd)
        word_embed = np.concatenate((target_embed,source_embed),axis=0)
        embedlist.append(word_embed)
        spl1.append(sp1)
        spl2.append(sp2)
        spl3.append(sp3)
        tpl1.append(tp1)
        tpl2.append(tp2)
        tpl3.append(tp3)
    embedlist = np.array(embedlist)
    spl1 = np.array(spl1)
    spl2 = np.array(spl2)
    spl3 = np.array(spl3)
    tpl1 = np.array(tpl1)
    tpl2 = np.array(tpl2)
    tpl3 = np.array(tpl3)
    #dimensionality: nwords_target x size_of_embedding.
    #Sending input for the functional API.
    #In the current state, send a list of word embedding list,
    #posidlist and posidlist.
    return [embedlist,spl1,spl2,spl3,tpl1,tpl2,tpl3],labels

#    flist=['train.mt','train.src','train.align','train.tags']
#    datadir='/home/krishna/Summarizartion/TQE/data/t2/train/'
def get_data_mats(flist,datadir,tr_fl):
    #loading the german postagger and german pos dict.
    with open('nltk_german_classifier_data.pickle','rb') as fp:
        gpostagger = pickle.load(fp)
    with open('german_pos_dict','rb') as f:
        gposdict = pickle.load(f)
    
    dmat=[]   #The main list having the sentense info.
    for fn in flist:
        with open(datadir+fn,'r') as fp:
            fl=[]
            for j in fp:
                fl.append(j.decode('utf-8').strip())
            dmat.append(list(fl))
    dmat = np.array(dmat)
    dmat = list(np.transpose(dmat))
    #Sorting the sentenses based on target length.
    if tr_fl:
        dmat = sorted(dmat,key=lambda x: len(x[0].split()))
    dmat = np.array(dmat)
    x_train=[]
    y_train=[]
    for j in dmat:
        a,b = get_sentense_inputs(j,gpostagger,gposdict)
        x_train.append(a)
        y_train.append(b)
    return zip(*x_train),y_train,x_train[0][0].shape[0]

#Returning the x vectors, y vectors and minsentense length.


#Returning the x vectors, y vectors and minsentense length.
