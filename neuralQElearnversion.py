#This is a functional model, with 384 polyglot sent as input and
#two embedding layers (trainable) for POS Embeddings.

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Bidirectional,GRU,BatchNormalization,TimeDistributed,Embedding,Input,concatenate
from keras import optimizers
from keras import metrics
from keras import backend as K
from data_loadLearnEmbeds import get_data_mats
from functools import partial
from itertools import product
import numpy as np

testdir='/home/krishna/Summarizartion/TQE/data/t2/test'
traindir='/home/krishna/Summarizartion/TQE/data/t2/train'

#Custom Loss Function.
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

#The model part of the code.
def create_model():
    #There are 46 unique pos-tags
    #Take the input layer here.
    WordInput = Input(shape=(None,384))
    SrcPosInput = Input(shape=(None,))
    TarPosInput = Input(shape=(None,))

    SrcPosEmb = Embedding(50,150)(SrcPosInput)
    TarPosEmb = Embedding(50,150)(TarPosInput)
    #Complete this from here.

    x = concatenate([WordInput,SrcPosEmb,TarPosEmb])
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(400,activation='relu'))(x)
    x = TimeDistributed(Dense(400,activation='relu'))(x)
    x = Bidirectional(GRU(200,return_sequences=True),merge_mode='concat')(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(200,activation='relu'))(x)
    x = TimeDistributed(Dense(200,activation='relu'))(x)
    x = Bidirectional(GRU(100,return_sequences=True),merge_mode='concat')(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Dense(100,activation='relu'))(x)
    x = TimeDistributed(Dense(50,activation='relu'))(x)
    x = TimeDistributed(Dense(2,activation='softmax'))(x)
    model = Model(inputs=[WordInput,SrcPosInput,TarPosInput],outputs=x)
    model.compile(loss=weighted_categorical_crossentropy(np.array([3,1])),optimizer='rmsprop',metrics=['accuracy'])
    model.summary()
    return model

#The function for sentense-length batchwise fitting , testing.
#Train is a boolean to tell whetet to fit/evaluate.
def batch_wise_operate(model,x_t,y_t,minlen,train):
    cur_len=minlen
    tx=[]
    sp=[]
    tp=[]
    ty=[]
    if train==0:
        acc=[]
    spl = x_t[1]
    tpl = x_t[2]
    x_t = x_t[0]
    for j in range(len(x_t)):
        slen = x_t[j].shape[0]
        #print slen
        if cur_len==slen:
            tx.append(x_t[j])
            sp.append(spl[j])
            tp.append(tpl[j])
            ty.append(y_t[j])
            #print y_t[j].shape
        else:
            tx = np.stack(tx)
            sp = np.array(sp)
            tp = np.array(tp)
            tx = [tx,sp,tp]
            ty = np.stack(ty)
            if train:
                model.fit(tx,ty,batch_size=50)
            else:
                acc.append(model.predict(tx,batch_size=50))
            cur_len = slen
            tx=[x_t[j]]
            sp=[spl[j]]
            tp=[tpl[j]]
            ty=[y_t[j]]
    if train==0:
        return acc
    return model

if __name__=='__main__':
    tr_flist  =  ['train.mt','train.src','train.align','train.tags']
    testdir = '/home/krishna/Summarizartion/TQE/data/t2/train/'
    traindir = '/home/krishna/Summarizartion/TQE/data/t2/test/'

    #Sentense lenght wise sorted training data.
    x_t,y_t,tr_min = get_data_mats(tr_flist,testdir,1)

    #Testing the data.
    dev_flist  =  ['dev.mt','dev.src','dev.align','dev.tags']
    #Unsorted test data for predictions.
    x_dev,y_dev,dev_min = get_data_mats(dev_flist,testdir,0)
        
    #Training the model.
    model = create_model()
    model = batch_wise_operate(model,x_t,y_t,tr_min,1)

    #Saving predictions.
    outfile='NueralOutputsLearn.txt'
    hteroutfile='NueralHTERsLearn.txt'
    hfp = open(hteroutfile,'w')
    with open(outfile,'w') as fp:
        sldev = x_dev[1]
        tldev = x_dev[2]
        x_dev = x_dev[0]
        for j in range(len(x_dev)):
            k1 = x_dev[j][np.newaxis,:,:]
            print k1
            print k1.shape
            k2 = np.array(sldev[j])[np.newaxis,:]
            k3 = np.array(tldev[j])[np.newaxis,:]
            total=0
            bad=0
            op = model.predict([k1,k2,k3])
            print op
            #Iterating over each word.
            for k in op[0,:,:]:
                print k
                total+=1
                if k[0] >= k[1]:
                    fp.write("OK ")
                    print "OK ",
                else:
                    bad+=1
                    fp.write("BAD ")
                    print "BAD ",
            print ""
            fp.write("\n")
            hfp.write(str(bad/float(total)) +"\n")
            print "Hter ",bad/float(total)
    hfp.close()
