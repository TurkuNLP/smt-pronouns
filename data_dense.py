#from svm_pronouns import iter_data, create_alignments
import collections
import numpy as np
import codecs
import sys
import cPickle as pickle
import os.path
import random

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

        self.source_word_counter=collections.Counter()
        self.target_word_counter=collections.Counter()
        self.target_wordpos_counter=collections.Counter()

        self.label={u"<MASK>":0}
        self.trainable=True #If false, it will use <UNK>

    def get_id(self,label,dict,counter=None):
        if self.trainable:
            if counter is not None:
                counter.update([label])
            return dict.setdefault(label,len(dict)) #auto-grows
        else:
#            return dict.get(label,dict[u"<UNK>"])
            if dict==self.label:
                return dict.get(label)
            if counter is not None:
                if counter[label]==1:
                    return dict.get(u"<UNK>")
            return dict.get(label,dict[u"<UNK>"])


def get_example_count(training_fname, vs, window):
    count = 0
    ms=make_matrices(1,window,len(vs.label))
    raw_data=infinite_iter_data(training_fname,max_rounds=1)
    for minibatch in fill_batch(ms,vs,raw_data):
        count += 1
    return count

def read_vocabularies(training_fname,force_rebuild):
    voc_fname=training_fname+"-vocabularies.pickle"
    if force_rebuild or not os.path.exists(voc_fname):
        #make sure no feature has 0 index
        print >> sys.stderr, "Making one pass to gather vocabulary"
        vs=Vocabularies()
        raw_data=infinite_iter_data(training_fname,max_rounds=1) #Make a single pass
        for (sent,replace_tokens), sent_id, document in raw_data: # label, category, source sent, target sent, alignment, doc id

            if replace_tokens:
                label=sent[0]
                target=sent[3].strip().split(u" ")
                target_wp=map(word_pos_split,target) #[[word,pos],...]
            
                source=sent[2].strip().split(u" ")
                for l,replace in zip(label.split(u" "),replace_tokens):
                    vs.get_id(l,vs.label)

            else:
                target=sent[1].strip().split(u" ")
                target_wp=map(word_pos_split,target) #[[word,pos],...]

                source=sent[0].strip().split(u" ")

            assert len(target)==len(target_wp)
                
           
            for wp in target:
                vs.get_id(wp,vs.target_wordpos,counter=vs.target_wordpos_counter)
            for w,p in target_wp:
                vs.get_id(w,vs.target_word,counter=vs.target_word_counter)
                vs.get_id(p,vs.target_pos)
            for w in source:
                vs.get_id(w,vs.source_word,counter=vs.source_word_counter)
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


def yield_context(token_id, sent_id, document, lang, window, direction):
    """ token_id = index of the token where to start (replace token, or its aligned source word)
        sent_id = id of the sentence where the token is
        document = full document
        lang = source (2) or target (3) index
        window = context window size
        direction = read backwards (-1) or forwards (1)
        returns tokens (strings)

        TODO: implement skipping useless words here!
    """
    if direction==-1:
        counter=0
        for i in xrange(sent_id,-1,-1):
            if len(document[i])==4:
                lang_idx=lang-2
            else:
                lang_idx=lang
            text=document[i][lang_idx].split(u" ")
            if i!=sent_id:
                token_id=len(text) # read from the end of sentence
            for j in xrange(token_id-1,-1,-1):
                token=text[j]
                yield token
                counter+=1
                if counter>=window:
                    break
            else:
                continue
            break
#        else:
#            print >> sys.stderr, "end of document --> sentence_id:", sent_id, "context size:", counter

    if direction==1:
        counter=0
        for i in xrange(sent_id,len(document)):
            if len(document[i])==4:
                lang_idx=lang-2
            else:
                lang_idx=lang
            text=document[i][lang_idx].split(u" ")
            for j in xrange(token_id+1,len(text)):
                token=text[j]
                yield token
                counter+=1
                if counter>=window:
                    break
            else:
                token_id=-1 # for now on, read sentences from beginning
                continue
            break
#        else:
#            print >> sys.stderr, "end of document --> sentence_id:", sent_id, "context size:", counter
                
    



def fill_batch(ms,vs,data_iterator):
    """ Iterates over the data_iterator and fills the index matrices with fresh data
        ms = matrices, vs = vocabularies
    """

    matrix_dict=dict(zip(ms._fields,ms)) #the named tuple as dict, what we return
    batchsize,window=ms.target_word_left.shape
    row=0
    for (sent,replace_tokens), sent_id, document in data_iterator: # label, category, source sent, target sent, alignment, doc id

        if not replace_tokens:
            continue

        label=sent[0]
        #target=sent[3].strip().split(u" ")
        #target_wp=map(word_pos_split,target) #[[word,pos],...]
        #assert len(target)==len(target_wp)
        #source=sent[2].strip().split(u" ")

        alignments=create_alignments(sent[4])

        for l,replace in zip(label.split(u" "),replace_tokens):

            ms.labels[row] = 0#np.zeros(ms.labels[row].shape)
            ms.labels[row][vs.get_id(l,vs.label)] = 1

            # target left
            for j, token in enumerate(yield_context(replace,sent_id,document,3,window,-1)): # token_id, sent_id, document, lang, window, direction             
                ms.target_word_left[row,j]=vs.get_id(word_pos_split(token)[0],vs.target_word,vs.target_word_counter) # word
                ms.target_pos_left[row,j]=vs.get_id(word_pos_split(token)[1],vs.target_pos) # pos
                ms.target_wordpos_left[row,j]=vs.get_id(token,vs.target_wordpos,vs.target_wordpos_counter) # wordpos
            # target right
            for j, token in enumerate(yield_context(replace,sent_id,document,3,window,1)):        
                ms.target_word_right[row,j]=vs.get_id(word_pos_split(token)[0],vs.target_word,vs.target_word_counter) # word
                ms.target_pos_right[row,j]=vs.get_id(word_pos_split(token)[1],vs.target_pos) # pos
                ms.target_wordpos_right[row,j]=vs.get_id(token,vs.target_wordpos,vs.target_wordpos_counter) # wordpos

            # source
            source_tokens=alignments[replace] # all tokens aligned with replace
            assert source_tokens==sorted(source_tokens)
#            assert source_tokens == range(source_tokens[0], source_tokens[-1]+1) # check if there are gaps in replace alignments --> yes, there are...

            # source left (start reading from last aligned word, +1 because source_token is otherwise not included)
            for j, token in enumerate(yield_context(source_tokens[-1]+1,sent_id,document,2,window,-1)):
                ms.source_word_left[row,j]=vs.get_id(token,vs.source_word,vs.source_word_counter) # word
            # source right (start reading from first aligned word)
            for j, token in enumerate(yield_context(source_tokens[0]-1,sent_id,document,2,window,1)):   
                ms.source_word_right[row,j]=vs.get_id(token,vs.source_word,vs.source_word_counter) # word


#            target_lwindow=xrange(replace-1,max(0,replace-window)-1,-1) #left window
#            target_rwindow=xrange(replace+1,min(len(target),replace+window)) #right window
#            for j,target_idx in enumerate(target_lwindow):
#                ms.target_word_left[row,j]=vs.get_id(target_wp[target_idx][0],vs.target_word)
#                ms.target_pos_left[row,j]=vs.get_id(target_wp[target_idx][1],vs.target_pos)
#                ms.target_wordpos_left[row,j]=vs.get_id(target[target_idx],vs.target_wordpos)
#            for j,target_idx in enumerate(target_rwindow):
#                ms.target_word_right[row,j]=vs.get_id(target_wp[target_idx][0],vs.target_word)
#                ms.target_pos_right[row,j]=vs.get_id(target_wp[target_idx][1],vs.target_pos)
#                ms.target_wordpos_right[row,j]=vs.get_id(target[target_idx],vs.target_wordpos)

#            # now source
#            assert replace in alignments 
#            source_token=alignments[replace][0] # TODO: not just first one...?
#            source_lwindow=xrange(source_token-1,max(0,source_token-window)-1,-1)
#            source_rwindow=xrange(source_token+1,min(len(source),source_token+window))
#            for j,source_idx in enumerate(source_lwindow):
#                ms.source_word_left[row,j]=vs.get_id(source[source_idx],vs.source_word)
#            for j,source_idx in enumerate(source_rwindow):
#                ms.source_word_right[row,j]=vs.get_id(source[source_idx],vs.source_word)
            

            row+=1
            if row==batchsize:

                #Oh, dear I'm at it again :/
                #left_target, right_target, left_target_pos, right_target_pos

                yield (matrix_dict, matrix_dict['labels'])#([matrix_dict['target_word_left'], matrix_dict['target_word_right'], matrix_dict['target_pos_left'], matrix_dict['target_word_right']], matrix_dict['labels'])

                row=0
                wipe_matrices(ms)

def create_alignments(align):
    alignment=dict()
    for pair in align.split(u" "):
        s,t=pair.split(u"-")
        if int(t) not in alignment:
            alignment[int(t)]=[]
        alignment[int(t)].append(int(s))
    return alignment

def document_iterator(f):

    document=[]
    document_id=None

    for sent in f: # label, category, source sent, target sent, alignment, doc id
        cols=sent.strip().split(u"\t")
        assert len(cols)==4 or len(cols)==6
        if len(cols)==4:
            idx=cols[3]
        else:
            idx=cols[5]
        if document_id is None:
            assert len(document)==0
            document_id=idx
        elif document_id!=idx:
            yield document
            document=[]
            document_id=idx       
        document.append(cols)
    if document:
        yield document
            
        

def sentence_iterator(doc):

    for sent in doc:
        
        if len(sent)==4:
            yield sent,None
            continue

        label=sent[0]
        target=sent[3]
        to_be_replaced=[]
        for i,tok in enumerate(target.split(u" ")):
            if tok.startswith(u"REPLACE_"):
                to_be_replaced.append(i)
        assert len(to_be_replaced)==len(label.split(u" "))

        yield sent,to_be_replaced


def infinite_iter_data(f_name,max_rounds=None, max_items=None, shuffle=False):
    round_counter=0

    while True:
        yield_counter = 0
        print >> sys.stderr, "next pass"
        with codecs.open(f_name, u"rt", u"utf-8") as f:

            documents=[]

            for doc_id, document in enumerate(document_iterator(f)):
                documents.append(document)

            if shuffle:
                random.shuffle(documents)

            for document in documents:
                for sent_id, r in enumerate(sentence_iterator(document)):
                    yield r, sent_id, document
                    yield_counter +=1
                    if max_items is not None and yield_counter >= max_items:
                        break
        round_counter+=1
        if max_rounds is not None and round_counter==max_rounds:
            break

if __name__=="__main__":
    vs=read_vocabularies(u"train_data/NCv9.en-fr.data.filtered.withids",force_rebuild=False) #makes new ones if not found
    print "***"
    print vs.source_word_counter.most_common(10)
    vs.trainable=False
    ms=make_matrices(3,100,len(vs.label)) #minibatchsize,window,label_count
    raw_data=infinite_iter_data(u"train_data/IWSLT15.en-fr.data.filtered.withids")
#    raw_data=infinite_iter_data(u"train_data/NCv9.en-fr.data.filtered.withids",shuffle=True)
    for minibatch in fill_batch(ms,vs,raw_data):
        pass

