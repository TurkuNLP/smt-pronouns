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
import numpy as np
import gzip
import sys
from svm_pronouns import iter_data
import json
import copy
from data_dense import *
from sklearn.metrics import recall_score
window = 50
#First argument is the model_json
model_json_f = sys.argv[1]
#Second argument is the model_weight file
model_weight_f = sys.argv[2]
#3. tr_dt_file
tr_dt_file = sys.argv[3]
#Fouth argument is the file to be tagged
dev_dt_file = sys.argv[4]
#5. argument is the output_file_prefix
out_prefix = sys.argv[5]

#Let us load the data
vs=read_vocabularies(tr_dt_file,force_rebuild=False)

#Hummm... okay need to think this
vs.trainable = False

dev_data_size = get_example_count(dev_dt_file, vs, window)
#print dev_data_size

#Let's get the dev data generator
dev_ms=make_matrices(dev_data_size,window,len(vs.label))
raw_dev_data=infinite_iter_data(dev_dt_file)
dev_data = fill_batch(dev_ms,vs,raw_dev_data).next()

#Let's get the training data
#train_ms=make_matrices(minibatch_size,window,len(vs.label))
#raw_train_data=infinite_iter_data(tr_dt_file)

#training_data_size = get_example_count(tr_dt_file, vs, window)
#print training_data_size

#stacked = True

#Let's build a fancy functional model a'la new keras
print 'Load model...'
from keras.models import model_from_json
model = model_from_json(open(model_json_f).read())
index2label = {v:k for k,v in vs.label.items()}
print '... Done!'
#Load weights
print 'Loading Weights ...'
model.load_weights(model_weight_f)
print '... Done!'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#import pdb;pdb.set_trace()

#Let's predict!
preds = model.predict(dev_data[0], verbose=1)
dev_labels_text = []
for l in dev_data[1]:
    dev_labels_text.append(index2label[np.argmax(l)])
preds_text=[]
pred_iter = preds_text.__iter__()
for l in preds:
    preds_text.append(index2label[np.argmax(l)])
print "Micro f-score:", f1_score(dev_labels_text,preds_text,average=u"micro")
print "Macro f-score:", f1_score(dev_labels_text,preds_text,average=u"macro")
print "Macro recall:", recall_score(dev_labels_text,preds_text,average=u"macro")
print classification_report(dev_labels_text, preds_text)

import codecs
out = codecs.open(out_prefix + '_gold', 'wt','utf8')
raw_dev_data=infinite_iter_data(dev_dt_file,max_rounds=1)

#dev_ms=make_matrices(dev_data_size,window,len(vs.label))
#raw_dev_data=infinite_iter_data(dev_dt_file)
#dev_data = fill_batch(dev_ms,vs,raw_dev_data).next()

for elns in raw_dev_data:

    eln = elns[0]
    if eln[1] == None:
        out.write(u'\t\t' + u'\t'.join(eln[0][:-1]) + '\n')
    else:
        #tpls = [pred_iter.next() for i in range(len(eln[1]))]
        out.write(u'\t'.join(eln[0][:-1]) + '\n')
out.close()


out = codecs.open(out_prefix + '_pred', 'wt','utf8')
raw_dev_data=infinite_iter_data(dev_dt_file,max_rounds=1)
for elns in raw_dev_data:

    eln = elns[0]
        
    if eln[1] == None:
        out.write(u'\t\t' + u'\t'.join(eln[0][:-1]) + '\n')
    else:
        tpls = [pred_iter.next() for i in range(len(eln[1]))]
        out.write(u' '.join(tpls) + u'\t' + u'\t'.join(eln[0][1:-1]) + '\n')
out.close()
