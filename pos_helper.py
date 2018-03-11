from nltk.data import load
import numpy as np

tagdict = load('help/tagsets/upenn_tagset.pickle')

#Use NIL when you got no word aligned.
#Returns a random embedding assigned dict for each postag.
def generate_pos_embedding_dict():
    pos_embeddings = {}
    for j in tagdict.keys():
        pos_embeddings[j] = np.random.rand(50)
    pos_embeddings['NIL'] = np.random.rand(50)
    return pos_embeddings

#Return a number for each tag.
def gen_pos_index():
    tdct={}
    cnt=0
    for j in tagdict.keys():
        tdct[j] = cnt
        cnt+=1
    tdct['NIL'] = cnt
    cnt+=1
    return tdct
