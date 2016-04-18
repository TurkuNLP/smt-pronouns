from svm_pronouns import iter_data
import collections
import numpy as np
import codecs
import sys
import cPickle as pickle
import os.path

# This named tuple holds the matrices which make one minibatch
# We will not be making new ones for every minibatch, we'll just wipe the existing ones
Matrices=collections.namedtuple("Matrices",["source_word_left","target_word_left","target_pos_left","target_wordpos_left","source_word_right","target_word_right","target_pos_right","target_wordpos_right","labels"])
def make_matrices(minibatch_size,context_size,label_count):
    ms=Matrices(*(np.zeros((minibatch_size,context_size),np.int) for _ in Matrices._fields[:-1]),\
                     labels=np.zeros((minibatch_size,label_count),np.int))
    return ms

class Vocabularies(object):
    def __init__(self):
        self.source_word={u"<MASK>":0,u"<UNK>":1}
        self.target_word={u"<MASK>":0,u"<UNK>":1}
        self.target_pos={u"<MASK>":0,u"<UNK>":1}
        self.target_wordpos={u"<MASK>":0,u"<UNK>":1}
        self.label={u"<MASK>":0}
        self.trainable=True #If false, it will use <UNK>

    def get_id(self,label,dict):
        if self.trainable:
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
            return dict.get(label,dict[u"<UNK>"])


def get_example_count(training_fname):
    raw_data=infinite_iter_data(training_fname,max_rounds=1)
    return len([x for x in raw_data])

def read_vocabularies(training_fname,force_rebuild):
    voc_fname=training_fname+"-vocabularies.pickle"
    if force_rebuild or not os.path.exists(voc_fname):
        #make sure no feature has 0 index
        print >> sys.stderr, "Making one pass to gather vocabulary"
        vs=Vocabularies()
        raw_data=infinite_iter_data(training_fname,max_rounds=1) #Make a single pass
        for sent,replace_tokens in raw_data: # label, category, source sent, target sent, alignment, doc id
            label=sent[0]
            target=sent[3].strip().split(u" ")
            target_wp=map(word_pos_split,target) #[[word,pos],...]
            assert len(target)==len(target_wp)
            source=sent[2].strip().split(u" ")
            for l,replace in zip(label.split(u" "),replace_tokens):
                vs.get_id(l,vs.label)
                for wp in target:
                    vs.get_id(wp,vs.target_wordpos)
                for w,p in target_wp:
                    vs.get_id(w,vs.target_word)
                    vs.get_id(p,vs.target_pos)
                for w in source:
                    vs.get_id(w,vs.source_word)
        print >> sys.stderr, "Saving new vocabularies to", voc_fname
        save_vocabularies(vs,voc_fname)
    else:
        print >> sys.stderr, "Loading vocabularies from", voc_fname
        vs=load_vocabularies(voc_fname)
    return vs


def save_vocabularies(vs,f_name):
    with open(f_name,"wb") as f:
        pickle.dump(vs,f,pickle.HIGHEST_PROTOCOL)

def load_vocabularies(f_name):
    with open(f_name,"rb") as f:
        return pickle.load(f)

def wipe_matrices(ms):
    for idx in xrange(len(ms._fields)):
        ms[idx].fill(0)


def word_pos_split(word_pos):
    ### Todo: should the target have other REPLACE instances?
    w_p=word_pos.rsplit(u"|",1)
    if len(w_p)==1: #Replace has no pos
        return w_p[0],u"RETPOS"
    else:
        return w_p



def fill_batch(ms,vs,data_iterator):
    """Iterates over the data_iterator and fills the index matrices with fresh data"""

    matrix_dict=dict(zip(ms._fields,ms)) #the named tuple as dict, what we return
    batchsize,window=ms.target_word_left.shape
    row=0
    for sent,replace_tokens in data_iterator: # label, category, source sent, target sent, alignment, doc id
        label=sent[0]
        target=sent[3].strip().split(u" ")
        target_wp=map(word_pos_split,target) #[[word,pos],...]
        assert len(target)==len(target_wp)
        source=sent[2].strip().split(u" ")
        for l,replace in zip(label.split(u" "),replace_tokens):

            ms.labels[row] = 0#np.zeros(ms.labels[row].shape)
            ms.labels[row][vs.get_id(l,vs.label)] = 1

            target_lwindow=xrange(replace-1,max(0,replace-window)-1,-1) #left window
            target_rwindow=xrange(replace+1,min(len(target),replace+window)) #right window
            for j,target_idx in enumerate(target_lwindow):
                ms.target_word_left[row,j]=vs.get_id(target_wp[target_idx][0],vs.target_word)
                ms.target_pos_left[row,j]=vs.get_id(target_wp[target_idx][1],vs.target_pos)
                ms.target_wordpos_left[row,j]=vs.get_id(target[target_idx],vs.target_wordpos)
            for j,target_idx in enumerate(target_rwindow):
                ms.target_word_right[row,j]=vs.get_id(target_wp[target_idx][0],vs.target_word)
                ms.target_pos_right[row,j]=vs.get_id(target_wp[target_idx][1],vs.target_pos)
                ms.target_wordpos_right[row,j]=vs.get_id(target[target_idx],vs.target_wordpos)
            row+=1
            if row==batchsize:

                #Oh, dear I'm at it again :/
                #left_target, right_target, left_target_pos, right_target_pos

                yield (matrix_dict, matrix_dict['labels'])#([matrix_dict['target_word_left'], matrix_dict['target_word_right'], matrix_dict['target_pos_left'], matrix_dict['target_word_right']], matrix_dict['labels'])

                row=0
                wipe_matrices(ms)


def infinite_iter_data(f_name,max_rounds=None, max_items=None):
    round_counter=0

    while True:
        yield_counter = 0
        print >> sys.stderr, "next pass"
        with codecs.open(f_name, u"rt", u"utf-8") as f:
            for r in iter_data(f, max_examples=0):
                yield r
                yield_counter +=1
                if max_items is not None and yield_counter >= max_items:
                    break
        round_counter+=1
        if max_rounds is not None and round_counter==max_rounds:
            break

if __name__=="__main__":
    vs=read_vocabularies(u"train_data/all.en-fr.filtered.withids",force_rebuild=False) #makes new ones if not found
    ms=make_matrices(3,100,len(vs.label)) #minibatchsize,window,label_count
    raw_data=infinite_iter_data(u"train_data/all.en-fr.filtered.withids")
    for minibatch in fill_batch(ms,vs,raw_data):
        pass
