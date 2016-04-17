from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
from keras.layers.core import Masking
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.datasets import reuters
from keras.callbacks import Callback,ModelCheckpoint
import conllutil
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, classification_report
import codecs
import lwvlib
import numpy as np
import gzip
import sys
from svm_pronouns import iter_data
import json
import copy
from data_dense import *


class CustomCallback(Callback):

    def __init__(self, dev_data,dev_labels,index2label):
        self.dev_data=dev_data
#        self.dev_right=dev_right
        self.dev_labels=dev_labels
        self.index2label=index2label

        self.dev_labels_text=[]
        for l in self.dev_labels:
            self.dev_labels_text.append(index2label[np.argmax(l)])

        

    def on_epoch_end(self, epoch, logs={}):
        print logs

        corr=0
        tot=0
        preds = self.model.predict(self.dev_data, verbose=1)

        preds_text=[]
        for l in preds:
            preds_text.append(self.index2label[np.argmax(l)])

        print "Micro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"micro")
        print "Macro f-score:", f1_score(self.dev_labels_text,preds_text,average=u"macro")
        print classification_report(self.dev_labels_text, preds_text)

        for i in xrange(len(self.dev_labels)):

        #    next_index = sample(preds[i])
            next_index = np.argmax(preds[i])
            # print preds[i],next_index,index2label[next_index]

            l = self.index2label[next_index]

            # print "correct:", index2label[np.argmax(dev_labels[i])], "predicted:",l
            if self.index2label[np.argmax(self.dev_labels[i])]==l:
                corr+=1
            tot+=1
        print corr,"/",tot
        

vocab=set()
vocab2index = None
index2vocab = None
dist_labels= None
label2index = None
index2label = None
window=100


def get_labels(data, vs):

    vectors = []
    for i in data['labels']:
        tv = np.zeros(len(vs.label),dtype='int32')
        tv[i] = 1
        vectors.append(tv)

    return np.asarray(vectors)

#labels_v=np.array([labels2index[i] for i in next_chars ])
#from keras.utils import np_utils, generic_utils
#labels_v = np_utils.to_categorical(next_chars_v, len(chars)) # http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


vec_size = 50

#TODO: for this night
#1. Get dev set from Filip's machine
ms=make_matrices(563,100) #minibatchsize,window
vs=make_vocabularies()
f=codecs.open(u"dev_data/TEDdev.en-fr.data.filtered.withids", u"rt", u"utf-8")
raw_data=iter_data(f, max_examples=20000) #make this some absurdly large number later

#I know it's hacky, but I'm tired!
dev_data = None
for minibatch in fill_batch(ms,vs,raw_data):
    dev_data = copy.deepcopy(minibatch)
    break

#2439746 = train_data size
ms=make_matrices(20550,100) #minibatchsize,window
#vs=make_vocabularies()
f=codecs.open(u"train_data/all.en-fr.filtered.withids", u"rt", u"utf-8")
raw_data=iter_data(f, max_examples=2000000) #make this some absurdly large number later

train_data = None
for minibatch in fill_batch(ms,vs,raw_data):
    train_data = copy.copy(minibatch)
    break#import pdb;pdb.set_trace()

#3. Use Filip's code as a generator which one will fit
#Just do this later! 

#Let's build a fancy functional model a'la new keras
print 'Build model...'

#First the inputs
left_target = Input(shape=(window, ), name='target_word_left', dtype='int32')
right_target = Input(shape=(window, ), name='target_word_right', dtype='int32')

left_target_pos = Input(shape=(window, ), name='target_pos_left', dtype='int32')
right_target_pos = Input(shape=(window, ), name='target_pos_right', dtype='int32')

#Then the embeddings
from keras.layers.embeddings import Embedding
shared_emb_pos = Embedding(len(vs.target_pos), vec_size, input_length=window)#, mask_zero=True)
shared_emb = Embedding(len(vs.target_word), vec_size, input_length=window)#, mask_zero=True)

vector_left_target_pos = shared_emb_pos(left_target_pos)
vector_right_target_pos = shared_emb_pos(right_target_pos)

vector_left_target = shared_emb(left_target)
vector_right_target = shared_emb(right_target)

merged_right = merge([vector_right_target, vector_right_target_pos], mode='concat', concat_axis=-1)
merged_left = merge([vector_left_target, vector_left_target_pos], mode='concat', concat_axis=-1)

#The lstms
right_lstm = LSTM(128)
left_lstm = LSTM(128)

#This is either clever or very silly :D
#This is to circumvent merge's incompatibility with masks
#For this to work, the first embedding (index 0) will be all zeros
#Horay! It works!
vector_mask = Masking(mask_value=np.zeros(vec_size*2, dtype=np.float32))
masked_m_right = vector_mask(merged_right)
masked_m_left = vector_mask(merged_left)

left_target_lstm_out = left_lstm(masked_m_left)#merged_left)
right_target_lstm_out = right_lstm(masked_m_right)#merged_right)

merged_vector = merge([left_target_lstm_out, right_target_lstm_out], mode='concat', concat_axis=-1)

#The prediction layer
dense_out = Dense(128, activation='relu')(merged_vector)
predictions = Dense(len(vs.label), activation='softmax', name='labels')(dense_out)

model = Model(input=[left_target, right_target, left_target_pos, right_target_pos], output=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#Let's make the first vector all zeros on all embeddings:
#I know very hacky! But I suppose the only way to get masking to work :/
for emb in [shared_emb, shared_emb_pos]:
    W = emb.W.get_value()
    W[0] = np.zeros(W[0].shape, dtype=np.float32)
    emb.W.set_value(W)

#Hack, just so that this works for now.
train_labels = get_labels(train_data, vs)
dev_labels = get_labels(dev_data, vs)
index2label = {v:k for k,v in vs.label.items()}

evalcb=CustomCallback(dev_data,dev_labels,index2label) # evaluate after each epoch
savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # save model (I have not tested this!)

#import pdb;pdb.set_trace()
model.fit(train_data, train_labels , batch_size=100, nb_epoch=100, callbacks=[evalcb, savecb])

    

