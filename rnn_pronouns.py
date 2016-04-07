from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Merge
from keras.layers.recurrent import LSTM
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


    return vectorized_left,vectorized_right,vectorized_labels


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



# BUILD MODEL: 2 concatenated LSTMs
print 'Build model...'
model_left = Sequential()
model_left.add(LSTM(128, return_sequences=False, input_shape=(window, len(vocab))))

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



#dev data
with codecs.open(u"dev_data/TEDdev.en-fr.data.filtered.withids",u"rt",u"utf-8") as f:
    dev_left,dev_right,dev_labels=data(f,batch=1000)
print "Devel examples:",len(dev_labels)


evalcb=CustomCallback(dev_left,dev_right,dev_labels,index2label) # evaluate after each epoch

savecb=ModelCheckpoint(u"rnn_model.model", monitor='val_acc', verbose=1, save_best_only=True, mode='auto') # save model (I have not tested this!)

decoder.fit([train_left,train_right], train_labels, batch_size=10, nb_epoch=100, verbose=1, show_accuracy=True, validation_data=([dev_left,dev_right],dev_labels), callbacks=[evalcb,savecb])

    
    

