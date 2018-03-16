from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation,Bidirectional,GRU,BatchNormalization,TimeDistributed,Embedding,Input,concatenate
from keras import optimizers
from keras import metrics
from keras import backend as K
from data_loadL2 import get_data_mats
from functools import partial
from itertools import product
import numpy as np
from gruln import GRULN

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
    #There are 55 unique pos-tags
    #Take the input layer here.
    WordInput = Input(shape=(None,384))
    spi1 = Input(shape=(None,))
    spi2 = Input(shape=(None,))
    spi3 = Input(shape=(None,))
    tpi1 = Input(shape=(None,))
    tpi2 = Input(shape=(None,))
    tpi3 = Input(shape=(None,))

    spe1 = Embedding(60,50)(spi1)
    spe2 = Embedding(60,50)(spi2)
    spe3 = Embedding(60,50)(spi3)
    tpe1 = Embedding(60,50)(tpi1)
    tpe2 = Embedding(60,50)(tpi2)
    tpe3 = Embedding(60,50)(tpi3)
    #Complete this from here.

    x = concatenate([WordInput,spe1,spe2,spe3,tpe1,tpe2,tpe3])
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(400,activation='relu'))(x)
    x = TimeDistributed(Dense(400,activation='relu'))(x)
    x = Bidirectional(GRULN(200,return_sequences=True),merge_mode='concat')(x)
    #x = BatchNormalization()(x)
    x = TimeDistributed(Dense(200,activation='relu'))(x)
    x = TimeDistributed(Dense(200,activation='relu'))(x)
    x = Bidirectional(GRULN(100,return_sequences=True),merge_mode='concat')(x)
    #x = BatchNormalization()(x)
    x = TimeDistributed(Dense(100,activation='relu'))(x)
    x = TimeDistributed(Dense(50,activation='relu'))(x)
    x = TimeDistributed(Dense(2,activation='softmax'))(x)
    model = Model(inputs=[WordInput,spi1,spi2,spi3,tpi1,tpi2,tpi3],outputs=x)
    model.compile(loss=weighted_categorical_crossentropy(np.array([3,1])),optimizer='rmsprop',metrics=['accuracy'])
    model.summary()
    return model

#The function for sentense-length batchwise fitting , testing.
#Train is a boolean to tell whetet to fit/evaluate.
def batch_wise_operate(model,x_t,y_t,minlen,train):
    cur_len=minlen
    tx=[]
    sp1=[]
    sp2=[]
    sp3=[]
    tp1=[]
    tp2=[]
    tp3=[]
    ty=[]
    if train==0:
        acc=[]
    spl1 = x_t[1]
    spl2 = x_t[2]
    spl3 = x_t[3]
    tpl1 = x_t[4]
    tpl2 = x_t[5]
    tpl3 = x_t[6]
    x_t = x_t[0]
    for j in range(len(x_t)):
        slen = x_t[j].shape[0]
        #print slen
        if cur_len==slen:
            tx.append(x_t[j])
            sp1.append(spl1[j])
            sp2.append(spl2[j])
            sp3.append(spl3[j])
            tp1.append(tpl1[j])
            tp2.append(tpl2[j])
            tp3.append(tpl3[j])
            ty.append(y_t[j])
            #print y_t[j].shape
        else:
            tx = np.stack(tx)
            sp1 = np.array(sp1)
            sp2 = np.array(sp2)
            sp3 = np.array(sp3)
            tp1 = np.array(tp1)
            tp2 = np.array(tp2)
            tp3 = np.array(tp3)            
            tx = [tx,sp1,sp2,sp3,tp1,tp2,tp3]
            ty = np.stack(ty)
            if train:
                model.fit(tx,ty,batch_size=50)
            else:
                acc.append(model.predict(tx))
            cur_len = slen
            tx=[x_t[j]]
            sp1=[spl1[j]]
            sp2=[spl2[j]]
            sp3=[spl3[j]]
            tp1=[tpl1[j]]
            tp2=[tpl2[j]]
            tp3=[tpl3[j]]
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
    outfile='GRULNLearn3Pos.txt'
    hteroutfile='GRULNLearn3Pos.txt'
    hfp = open(hteroutfile,'w')
    with open(outfile,'w') as fp:
        sld1 = x_dev[1]
        sld2 = x_dev[2]
        sld3 = x_dev[3]
        tld1 = x_dev[4]
        tld2 = x_dev[5]
        tld3 = x_dev[6]
        x_dev = x_dev[0]
        for j in range(len(x_dev)):
            k1 = x_dev[j][np.newaxis,:,:]
            print k1
            print k1.shape
            k2 = np.array(sld1[j])[np.newaxis,:]
            k3 = np.array(sld2[j])[np.newaxis,:]
            k4 = np.array(sld3[j])[np.newaxis,:]
            k5 = np.array(tld1[j])[np.newaxis,:]
            k6 = np.array(tld2[j])[np.newaxis,:]
            k7 = np.array(tld3[j])[np.newaxis,:]
            total=0
            bad=0
            op = model.predict([k1,k2,k3,k4,k5,k6,k7])
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
