import codecs
import sys


from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,f1_score

# Don't do this in real programs
import warnings
warnings.filterwarnings("ignore")

def create_alignments(source,target,align):
    alignment=dict()
    for pair in align.split(u" "):
        s,t=pair.split(u"-")
        if int(t) not in alignment:
            alignment[int(t)]=[]
        alignment[int(t)].append(int(s))
    return alignment

def iter_data(f,max_examples=0):
    """ reads data, also used with rnn_pronouns.py """
    counter=0
    for line in f: # line is one sentence
        if max_examples!=0 and counter>max_examples:
            break
        cols=line.strip().split(u"\t")
        assert len(cols)==4 or len(cols)==6
        if len(cols)==4: # nothing to predict here, use this context  # TODO 
            continue
        label=cols[0]
        categories=cols[1]
        source=cols[2]
        target=cols[3]
        to_be_replaced=[]
        for i,tok in enumerate(target.split(u" ")):
            if tok.startswith(u"REPLACE_"):
                to_be_replaced.append(i)
        assert len(to_be_replaced)==len(label.split(u" "))
        align=cols[4]
        doc_id=[5]
        counter+=1
        yield cols,to_be_replaced


def create_examples(f,max_examples=0):
    counter=0

    examples = []
    labels = []

    for sent in iter_data(f, max_examples): # label, category, source sent, target sent, alignment, doc id
        
        alignments=create_alignments(sent[2],sent[3],sent[4])
#        alignments=dict()
        fdict=create_features(sent[2],sent[3],alignments)

        examples.append(fdict)
        labels.append(sent[0])
#        for i, token in enumerate(tokens):
#            examples.append(create_features(i, tokens))
#            labels.append(token_labels[i])
#    

    return examples, labels
    
def create_features(source, target, alignments):
    
    features = dict()
    for token in source.split(u" "):
        features[u"source="+token]=1.0

    replace_token=None
    for i,token in enumerate(target.split(u" ")):
        if u"REPLACE" in token:
            replace_token=i
        else:
            features[u"target="+token]=1.0
        #features[token]+=1.0
    assert replace_token is not None
    for token in alignments[replace_token]:
        features[u"orig=%s" %(source.split()[token])]=1.0
    return features


def train():
    
    
    print 'Reading data and creating features'
    
    with codecs.open(u"train_data/all.en-fr.filtered.withids",u"rt",u"utf-8") as f:
        train_examples, train_labels = create_examples(f,max_examples=1000)

    with codecs.open(u"dev_data/TEDdev.en-fr.data.filtered.withids",u"rt",u"utf-8") as f:
        devel_examples, devel_labels = create_examples(f,max_examples=100)

    
    vectorizer = DictVectorizer()
    train_examples = vectorizer.fit_transform(train_examples)
    devel_examples = vectorizer.transform(devel_examples)
    
    print "Training data has %s examples and %s features" % (train_examples.shape)
    print "Evaluation data has %s examples and %s features" % (devel_examples.shape)
    
    print 'Training classifiers'
    
    c_values = [2**i for i in range(-5, 15)]
#    c_values = [2**i for i in range(-1, 5)]
    
    classifiers = [LinearSVC(C=c, random_state=1) for c in c_values]
    
    for classifier in classifiers:
        print 'Training classifier with C value %s' % classifier.C
        classifier.fit(train_examples, train_labels) # This is the actual training of the classifiers
    
    
    print 'Evaluating classifiers and selecting the best model'
    

    results = []
    for classifier in classifiers:
        f_score = f1_score(devel_labels, classifier.predict(devel_examples), labels=None, pos_label=None, average='macro') # use macro fscore
        print 'Classifier with C value %s achieved F-score %s' % (classifier.C, f_score)
        results.append((classifier, f_score))
        
    results.sort(key=lambda x: x[1], reverse=True) # Sort the classifiers by their F-score
    
    best_classifier, best_f_score = results[0]
    
    print 'Best results with C value %s, F-score %s' % (best_classifier.C, best_f_score)
    
    print classification_report(devel_labels, best_classifier.predict(devel_examples))    


    return best_classifier, vectorizer




if __name__ == '__main__':
    
    classifier, vectorizer = train()

