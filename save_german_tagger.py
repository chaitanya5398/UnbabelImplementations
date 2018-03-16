#This is to get the german tagger saved.

import nltk
import random
from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
import pickle

corp = nltk.corpus.ConllCorpusReader('.', 'tiger_release_aug07.corrected.16012013.conll09',
                                     ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                     encoding='utf-8')

tagged_sents = list(corp.tagged_sents())
gpdict={}
cnt=0
for j in tagged_sents:
    for k in j:
        a,b = k
        print b
        if b not in gpdict:
            gpdict[b] = cnt
            cnt+=1
gpdict['NIL'] = cnt
cnt +=1

print cnt
random.shuffle(tagged_sents)

# set a split size: use 90% for training, 10% for testing
split_perc = 0.1
split_size = int(len(tagged_sents) * split_perc)
train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]
tagger = ClassifierBasedGermanTagger(train=train_sents)

accuracy = tagger.evaluate(test_sents)
print accuracy

with open('nltk_german_classifier_data.pickle', 'wb') as f:
    pickle.dump(tagger, f, protocol=2)
with open('german_pos_dict','wb') as fp:
    pickle.dump(gpdict,fp,protocol=pickle.HIGHEST_PROTOCOL)
