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
window=50
vec_size = 50
minibatch_size = 300

#labels_v=np.array([labels2index[i] for i in next_chars ])
#from keras.utils import np_utils, generic_utils
#labels_v = np_utils.to_categorical(next_chars_v, len(chars)) # http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


#Let us load the data
vs=read_vocabularies(u"train_data/all.en-fr.filtered.withids",force_rebuild=False)
#vs.trainable = False Doesnt work :/

training_data_size = get_example_count(u"train_data/all.en-fr.filtered.withids")

#Let's get the dev data generator
dev_ms=make_matrices(428,50,len(vs.label))
raw_dev_data=infinite_iter_data(u"dev_data/TEDdev.en-fr.data.filtered.withids")
dev_data = fill_batch(dev_ms,vs,raw_dev_data).next()

#Let's get the training data
train_ms=make_matrices(minibatch_size,50,len(vs.label))
raw_train_data=infinite_iter_data(u"train_data/all.en-fr.filtered.withids")

#Let's build a fancy functional model a'la new keras
print 'Build model...'

#First the inputs
left_target = Input(shape=(window, ), name='target_word_left', dtype='int32')
right_target = Input(shape=(window, ), name='target_word_right', dtype='int32')

left_target_pos = Input(shape=(window, ), name='target_pos_left', dtype='int32')
right_target_pos = Input(shape=(window, ), name='target_pos_right', dtype='int32')

left_source = Input(shape=(window, ), name='source_word_left', dtype='int32')
right_source = Input(shape=(window, ), name='source_word_right', dtype='int32')

#Then the embeddings
from keras.layers.embeddings import Embedding
shared_emb_pos = Embedding(len(vs.target_pos), vec_size, input_length=window, mask_zero=True)
shared_emb = Embedding(len(vs.target_word), vec_size, input_length=window, mask_zero=True)
shared_emb_src = Embedding(len(vs.source_word), vec_size, input_length=window, mask_zero=True)

vector_left_source = shared_emb_src(left_source)
vector_right_source = shared_emb_src(right_source)

vector_left_target_pos = shared_emb_pos(left_target_pos)
vector_right_target_pos = shared_emb_pos(right_target_pos)

vector_left_target = shared_emb(left_target)
vector_right_target = shared_emb(right_target)

#Here I merged pos and word into one
#merged_right = merge([vector_right_target, vector_right_target_pos], mode='concat', concat_axis=-1)
#merged_left = merge([vector_left_target, vector_left_target_pos], mode='concat', concat_axis=-1)

#The lstms
right_lstm = LSTM(50)
left_lstm = LSTM(50)

right_lstm_pos = LSTM(50)
left_lstm_pos = LSTM(50)

source_right_lstm = LSTM(50)
source_left_lstm = LSTM(50)

left_source_lstm_out = source_left_lstm(vector_left_source)
right_source_lstm_out = source_right_lstm(vector_right_source)

left_target_lstm_out = left_lstm(vector_left_target)
right_target_lstm_out = right_lstm(vector_right_target)

left_target_pos_lstm_out = left_lstm_pos(vector_left_target_pos)
right_target_pos_lstm_out = right_lstm_pos(vector_right_target_pos)

#A monster!
merged_vector = merge([left_target_pos_lstm_out, right_target_pos_lstm_out, left_target_lstm_out, right_target_lstm_out, left_source_lstm_out, right_source_lstm_out], mode='concat', concat_axis=-1)

#The prediction layer
dense_out = Dense(128, activation='relu')(merged_vector)
predictions = Dense(len(vs.label), activation='softmax', name='labels')(dense_out)

model = Model(input=[left_target, right_target, left_target_pos, right_target_pos, left_source, right_source], output=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

index2label = {v:k for k,v in vs.label.items()}
evalcb=CustomCallback(dev_data[0],dev_data[1],index2label)
savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#import pdb;pdb.set_trace()
model.fit_generator(fill_batch(train_ms,vs,raw_train_data), samples_per_epoch=training_data_size, nb_epoch=50, callbacks=[evalcb,savecb])

