#This is the code, taking 684 Word Embeddings (384 Polyglot + 150 + 150.) 150 is random vector assigned to each POS Tag.


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Bidirectional,GRU,BatchNormalization,TimeDistributed,Embedding
from keras import optimizers
from keras import metrics
from keras import backend as K
from data_load import *
from functools import partial
from itertools import product

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
    model = Sequential()
    #Take the input layer here.
    model.add(TimeDistributed(Dropout(0.5),input_shape=(None,684)))
    model.add(TimeDistributed(Dense(400,activation='relu')))
    model.add(TimeDistributed(Dense(400,activation='relu')))
    model.add(Bidirectional(GRU(200,return_sequences=True),merge_mode='concat'))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(200,activation='relu')))
    model.add(TimeDistributed(Dense(200,activation='relu')))
    model.add(Bidirectional(GRU(100,return_sequences=True),merge_mode='concat'))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(100,activation='relu')))
    model.add(TimeDistributed(Dense(50,activation='relu')))
    model.add(TimeDistributed(Dense(2,activation='softmax')))
    model.compile(loss=weighted_categorical_crossentropy(np.array([3,1])),optimizer='rmsprop',metrics=['accuracy'])
    return model

#The function for sentense-length batchwise fitting , testing.
#Train is a boolean to tell whetet to fit/evaluate.
def batch_wise_operate(model,x_t,y_t,minlen,train):
    cur_len=minlen
    tx=[]
    ty=[]
    if train==0:
        acc=[]
    for j in range(len(x_t)):
        slen = x_t[j].shape[0]
        #print slen
        if cur_len==slen:
            tx.append(x_t[j])
            ty.append(y_t[j])
            #print y_t[j].shape
        else:
            print len(ty), "  sentese length ",cur_len
            tx = np.stack(tx)
            #print tx.shape
            ty = np.stack(ty)
            #print ty.shape
            if train:
                model.fit(tx,ty,batch_size=50,epochs=5)
            else:
                acc.append(model.predict(tx,batch_size=50))
            cur_len = slen
            tx=[x_t[j]]
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
    outfile='NueralOutputs.txt'
    hteroutfile='NueralHTERs.txt'
    hfp = open(hteroutfile,'w')
    with open(outfile,'w') as fp:
        for j in x_dev:
            j = j[np.newaxis,:,:]
            print j
            print j.shape
            total=0
            bad=0
            op = model.predict(j)
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
