from keras.models import Sequential, Graph, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
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

class CustomCallback(Callback):

    def __init__(self, dev_left,dev_right,dev_labels,index2label):
        self.dev_left=dev_left
        self.dev_right=dev_right
        self.dev_labels=dev_labels
        self.index2label=index2label

        self.dev_labels_text=[]
        for l in self.dev_labels:
            self.dev_labels_text.append(index2label[np.argmax(l)])

        

    def on_epoch_end(self, epoch, logs={}):
        print logs

        corr=0
        tot=0
        preds = self.model.predict([self.dev_left, self.dev_right], verbose=1)

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

window=10

def data(f, batch=50):

    global vocab2index, dist_labels, label2index, vocab, window, index2label

    contexts_left=[]
    contexts_right=[]
    labels=[]

    for sent,replace_tokens in iter_data(f, max_examples=batch): # label, category, source sent, target sent, alignment, doc id

        label=sent[0]
        target=sent[3].strip().split(u" ")


        for l,replace in zip(label.split(u" "),replace_tokens):
            # l = correct label
            # replace = index of the token in target sentence

            context_left=[]
            context_right=[]

#            print l,replace,target[replace]

            # left sided context
            if replace<window:
                context_left=target[:replace]
                context_left.insert(0,u"<BOS>")
            else:
                context_left=target[replace-window:replace]

            
            # right sided context
            if replace>len(target)-window:
                context_right=target[replace:]
                context_right.append(u"<EOS>")
            else:
                context_right=target[replace:replace+window]

 
            contexts_right.append(u" ".join(t.lower() for t in context_right)) # make it a string, why...?
            contexts_left.append(u" ".join(t.lower() for t in context_left))
            labels.append(l)

    text=u" ".join(l for l in contexts_left+contexts_right)
    
    # create index dictionaries, these must then be saven as json 
    if not vocab2index:

        vocab=set(text.split(u" "))
        print >> sys.stderr, "Vocabulary size:", len(vocab)
        print >> sys.stderr, "Classes:", len(set(labels))

        vocab2index = dict((c, i) for i, c in enumerate(vocab))
#        index2vocab = dict((i, c) for i, c in enumerate(vocab))

        dist_labels=set(labels)
        label2index = dict((c, i) for i, c in enumerate(dist_labels))
        index2label = dict((i, c) for i, c in enumerate(dist_labels))


    # vectorize the data
    vectorized_left=np.zeros((len(contexts_left),window,len(vocab)))
    vectorized_labels=np.zeros((len(labels),len(dist_labels)))

    for i,context in enumerate(contexts_left): # context is a string
        for j,token in enumerate(context.split(u" ")): # hah, now I'm again splitting it! Should do something...
            if token not in vocab2index:
                continue
            vectorized_left[i,j,vocab2index[token]]=1.0
        vectorized_labels[i,label2index[labels[i]]]=1.0

    vectorized_right=np.zeros((len(contexts_right),window,len(vocab)))

    for i,context in enumerate(contexts_right):
        for j,token in enumerate(context.split(u" ")):
            if token not in vocab2index:
                continue
            vectorized_right[i,j,vocab2index[token]]=1.0

    #I'm so sorry you have to see this T_T
    revec_left = np.zeros((vectorized_left.shape[:2]), dtype=np.int32)
    revec_right = np.zeros((vectorized_right.shape[:2]), dtype=np.int32)

    for a, example in enumerate(vectorized_left):
        for b, sequence in enumerate(example):
            revec_left[a][b] = np.argmax(sequence)

    for a, example in enumerate(vectorized_right):
        for b, sequence in enumerate(example):
            revec_right[a][b] = np.argmax(sequence)

    return revec_left, revec_right, vectorized_labels #vectorized_left,vectorized_right,vectorized_labels

#labels_v=np.array([labels2index[i] for i in next_chars ])
#from keras.utils import np_utils, generic_utils
#labels_v = np_utils.to_categorical(next_chars_v, len(chars)) # http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))




#f=codecs.getreader(u"utf-8")(gzip.open(u"train_data/all.en-fr.filtered.withids"))

# GET DATA
with codecs.open(u"train_data/all.en-fr.filtered.withids", u"rt", u"utf-8") as f:
    train_left,train_right,train_labels=data(f,batch=10000)
print "Training examples:",len(train_labels)

#dev data
with codecs.open(u"dev_data/TEDdev.en-fr.data.filtered.withids",u"rt",u"utf-8") as f:
    dev_left,dev_right,dev_labels=data(f,batch=1000)
print "Devel examples:",len(dev_labels)

vec_size = 128

#Let's build a fancy functional model a'la new keras
print 'Build model...'

#First the inputs
left_target = Input(shape=(window, ), name='left_target', dtype='int32')
right_target = Input(shape=(window, ), name='right_target', dtype='int32')

#Then the embeddings
from keras.layers.embeddings import Embedding
shared_emb = Embedding(len(vocab), vec_size, input_length=window, mask_zero=True)

vector_left_target = shared_emb(left_target)
vector_right_target = shared_emb(right_target)

#The lstms
right_target_lstm = LSTM(128)
left_target_lstm = LSTM(128)

left_target_lstm_out = left_target_lstm(vector_left_target)
right_target_lstm_out = right_target_lstm(vector_right_target)
merged_vector = merge([left_target_lstm_out, right_target_lstm_out], mode='concat', concat_axis=-1)

#The prediction layer
dense_out = Dense(128, activation='relu')(merged_vector)
predictions = Dense(len(dist_labels), activation='softmax')(dense_out)

model = Model(input=[left_target, right_target], output=predictions)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#Make callbacks
evalcb=CustomCallback(dev_left,dev_right,dev_labels,index2label) # evaluate after each epoch
savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # save model (I have not tested this!)

model.fit([train_left, train_right], train_labels, batch_size=10, nb_epoch=20, validation_split=0.1, callbacks=[evalcb, savecb])
import pdb;pdb.set_trace()

'''
model = Graph()
        
model.add_input(name='left', input_shape=(window, ), dtype='int')
model.add_input(name='right', input_shape=(window, ), dtype='int')

#For future
#model.add_input(name='left_s', input_shape=(window, ))
#model.add_input(name='right_s', input_shape=(window, ))

vec_size=128

#Hack Embeddings in
from keras.layers.embeddings import Embedding

#Oh dear! What a hacky way to do this!
shared_emb = Embedding(len(vocab), vec_size, input_length=window, mask_zero=True)

#right_emb = copy.copy(left_emb)#Embedding(len(vocab), vec_size, input_length=window, mask_zero=True)
#right_emb.params = left_emb.params
#right_emb.W = left_emb.W

model.add_node(left_emb, name='emb_left', input='left')
model.add_node(right_emb, name='emb_right', input='right')

#Left & Right LSTM
#model.add_node(LSTM(128, return_sequences=False, dropout_W=0.1, dropout_U=0.1, input_shape=(window, vec_size)), name='left_lstm', input='emb_left')
#model.add_node(LSTM(128, return_sequences=False, dropout_W=0.1, dropout_U=0.1, input_shape=(window, vec_size)), name='right_lstm', input='emb_right')

model.add_node(LSTM(128, return_sequences=True, input_shape=(window, vec_size)), name='left_lstm_p', input='emb_left')
model.add_node(LSTM(128, return_sequences=False, input_shape=(window, vec_size)), name='left_lstm', input='left_lstm_p')

model.add_node(LSTM(128, return_sequences=True, input_shape=(window, vec_size)), name='right_lstm_p', input='emb_right')
model.add_node(LSTM(128, return_sequences=False, input_shape=(window, vec_size)), name='right_lstm', input='right_lstm_p')

#Time to Predict
model.add_node(Dense(128, activation='relu'), name='dense_1', inputs=['left_lstm', 'right_lstm'], merge_mode='concat')
model.add_node(Dense(len(dist_labels), activation='softmax'), name='softmax_1', input='dense_1') # why I have two dense layers here?
model.add_output(name='output', input='softmax_1')
model.compile(loss={'output':'categorical_crossentropy'}, optimizer='adam')

#dev data
with codecs.open(u"dev_data/TEDdev.en-fr.data.filtered.withids",u"rt",u"utf-8") as f:
    dev_left,dev_right,dev_labels=data(f,batch=1000)
print "Devel examples:",len(dev_labels)

evalcb=CustomCallback(dev_left,dev_right,dev_labels,index2label) # evaluate after each epoch
savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # save model (I have not tested this!)


#for x in range(100):
model.fit({'left':train_left, 'right':train_right, 'output':train_labels}, batch_size=10 ,nb_epoch=5, verbose=1, show_accuracy=True, validation_data={'left':dev_left, 'right':dev_right, 'output':dev_labels}, callbacks=[evalcb])
    #print model.evaluate({'left':dev_left, 'right':dev_right, 'output':dev_labels}, show_accuracy=True)

import pdb;pdb.set_trace()
'''

'''
# BUILD MODEL: 2 concatenated LSTMs
print 'Build model...'
model_left = Sequential()
model_left.add()

model_right = Sequential()
model_right.add(LSTM(128, return_sequences=False, input_shape=(window, len(vocab))))
#model_right.add(Dropout(0.2))

decoder = Sequential()
decoder.add(Merge([model_left, model_right], mode='concat'))
decoder.add(Dense(128, activation='relu'))
decoder.add(Dense(len(dist_labels), activation='softmax')) # why I have two dense layers here?

decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode='categorical')

#model.add(LSTM(128, return_sequences=False))
#model.add(Dropout(0.2))
#model.add(Dense(len(dist_labels)))
#model.add(Activation('softmax'))

#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#np.random.seed(1337)  # for reproducibility
'''

#Seems cool, but I've got no time to get them working!
#evalcb=CustomCallback(dev_left,dev_right,dev_labels,index2label) # evaluate after each epoch
#savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # save model (I have not tested this!)
#decoder.fit([train_left,train_right], train_labels, batch_size=10, nb_epoch=100, verbose=1, show_accuracy=True, validation_data=([dev_left,dev_right],dev_labels), callbacks=[evalcb,savecb])

    
    

