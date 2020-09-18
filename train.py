## DATA MANIPULATION
import pandas as pd 
import numpy as np 
import json

## STRING MANIPULATION AND NLP HELP FUNS
import re, string, copy
import nltk
from nltk import WordNetLemmatizer

## FILE SAVING
import pickle
from joblib import load, dump

## Keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
import tensorflow as tf

## TF-IDF VECTORIZER
from sklearn.feature_extraction.text import TfidfVectorizer

## SKLEARN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression

def load_and_split():
    data = pd.read_csv('data/train.csv')             # load data
    data['comment_text'].fillna("unknown", inplace=True)# fill empties
    train, valid = train_test_split(data,               # split into train & test
                                    random_state=42, 
                                    test_size=0.33, 
                                    shuffle=True)
    return train, valid

class PatternTokenizer(object):
    '''Preprocessor credit goes to fizzbuzz from kaggle 
    (https://www.kaggle.com/fizzbuzz/toxic-data-preprocessing)'''
    def __init__(self, lower=True, initial_filters=r"[^a-z0-9!@#\$%\^\&\*_\-,\.' ]", re_path='data/re_patterns.json',
                 remove_repetitions=True):
        self.lower = lower
        self.re_path = re_path
        self.initial_filters = initial_filters
        self.remove_repetitions = remove_repetitions
        self.patterns = None
        
    def process_text(self, text):
        f = open(self.re_path, 'r')
        self.patterns = json.load(f)
        x = self._preprocess(text)
        for target, patterns in self.patterns.items():
            for pat in patterns:
                x = re.sub(pat, target, x)
        x = re.sub(r"[^a-z' ]", ' ', x)
        return x.split()

    def process_ds(self, ds):
        ### ds = Data series
        f = open(self.re_path, 'r')
        self.patterns = json.load(f)
        # lower
        ds = copy.deepcopy(ds)
        if self.lower:
            ds = ds.str.lower()
        # remove special chars
        if self.initial_filters is not None:
            ds = ds.str.replace(self.initial_filters, ' ')
        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL) 
            ds = ds.str.replace(pattern, r"\1")

        for target, patterns in self.patterns.items():
            for pat in patterns:
                ds = ds.str.replace(pat, target)

        ds = ds.str.replace(r"[^a-z' ]", ' ')

        return ds.str.split()

    def _preprocess(self, text):
        # lower
        if self.lower:
            text = text.lower()

        # remove special chars
        if self.initial_filters is not None:
            text = re.sub(self.initial_filters, ' ', text)

        # fuuuuck => fuck
        if self.remove_repetitions:
            pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
            text = pattern.sub(r"\1", text)
        return text


class toxic_ensemble(object):
    '''
    ensemble model of TFIDF+NBSVM and GLOVE+LSTM to predict comment toxicity
    '''
    def __init__(self, 
                 embeddings_path = r'data/glove.6B.300d.txt',       # Path to embeddings
                 max_features=50000,    # Maximum number of features to extract from dictionary
                 embed_size=300,        # length of vector representation for each word (n-dim of glove vectors)
                 max_len = 100,         # maximum number of words in a comment to use (for padding)
                 classic_m = None,      # classic model (e.g., logistic regression) to combine with 
                                        # TFIDF & Naive Bayes transformed Features
                 nn_weights = None      # use pre-trained weights if exist (e.g., can train weights on GPU using CuDNN)
                                        # format: directory (e.g., 'artifacts/lstm_weights.h5')
                ):
        # Initialize TFIDF
        self.tfidf = TfidfVectorizer(ngram_range=(1,2),
                                    min_df=3, 
                                    max_df=0.9, 
                                    strip_accents='unicode', 
                                    use_idf=1,
                                    smooth_idf=1, 
                                    sublinear_tf=1) 
        # Initialize Keras standard preprocessing
        self.keras_tokenizer = Tokenizer(num_words=max_features)
        self.max_len = max_len
        self.embed_size = embed_size
        self.max_features = max_features
        # Initialize embedding matrix
        self.embedding_matrix = None
        # Initialize models
        self.r = None # Naieve Bayes transformation matrix 
        self.classic_m = classic_m
        self.nn_m = None
        self.nn_weights = nn_weights # use pre-trained weights if exist.
        self.embeddings_path = embeddings_path
        if classic_m is None: self.classic_m = LogisticRegression(solver='liblinear',class_weight='balanced')
    
    def fit_tfidf(self, X):
        ''' Fits TFIDF on X where X is a df column of texts, 
        eg: train['comment_text']'''
        return self.tfidf.fit(X)

    def fit_keras_tokenizer(self, X):
        ''' Fits Keras tokenizer on X where X is a df column of texts, 
        eg: train['comment_text']'''
        return self.keras_tokenizer.fit_on_texts(list(X.values)) 
        
    def transform_tfidf(self, X):
        ''' Vectorize using fitted TFIDF '''
        x_out = self.tfidf.transform(X)
        return x_out
    
    def transform_keras_tokenizer(self, X):
        ''' Vectorize using fitted Keras Tokenizer '''
        # Tokenize train and valid
        X_tokenized = self.keras_tokenizer.texts_to_sequences(list(X.values))
        # Padd
        X_out = pad_sequences(X_tokenized, maxlen=self.max_len)
        return X_out
    
    def fit_classic_m(self, X, y):
        ''' Fit classic model (eg lin regression) with NB transformed features'''
        y = y.values
        sum_1 = X[y==1].sum(axis=0)+1           # Feature Sum for Class 1
        p_1 = (sum_1) / ((y==1).sum())          # Convert to ratio of feature in class 1 - p(f|1)

        sum_0 = X[y==0].sum(axis=0)+1           # Feature Sum for Class 0
        p_0 = (sum_0) / ((y==0).sum())          # Convert to ratio of feature in class 0 - p(f|0) 

        self.r = np.log(p_1/p_0)                # Compute log ratios (the transformation matrix)
        X_nb = X.multiply(self.r)               # Obtain NB feature

        self.classic_m.fit(X_nb,y)              # Fit model
        
        return self                             # return fitted model & transformation matrix (need for X_valid / X_test)
        
    def create_embeddings(self):
        '''Build embedding matrix for NN using embeddings dictionary'''
        ## Build Glove vector dictionary
        embeddings_dict = dict()
        f = open(self.embeddings_path, encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vec
        f.close()

        ## Get mean and std from Glove vectors
        glove_mean, glove_std = (np.stack(list(embeddings_dict.values())).mean(), 
                                 np.stack(list(embeddings_dict.values())).std())
        
        wordidx_dict = self.keras_tokenizer.word_index # get word indices
        num_words = min(len(wordidx_dict), self.max_features) # should be max features, but could be less if wordidx dict contains <50000 words
        
        # random initliaization of weights
        self.embedding_matrix = np.random.normal(glove_mean, 
                                                glove_std, 
                                                size = (num_words, self.embed_size)) 
        # update embedding matrix with glove vectors
        for word, idx in wordidx_dict.items():
            if idx < self.max_features: # stay within max # of features
                # grab glove vector if exists
                vec = embeddings_dict.get(word) 
                # if glove vector exists, add to embedding matrix (i.e., replace random initialization)
                if vec is not None: self.embedding_matrix[idx] = vec 
        return self
    
    def build_nn(self):
        '''build bidirectional lstm model'''
        self.create_embeddings()
        input = Input(shape=(self.max_len,))
        x = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=False)(input) # embedding layer to obtain vectors for words
        x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x) # bidirectional lstm layer
        x = GlobalMaxPool1D()(x)
        x = Dense(50, activation="relu")(x) # 1st FC layer
        x = Dropout(0.1)(x)
        x = Dense(1, activation="sigmoid")(x) # Output label (6 outputs, 1 for each class for multi-label classification)
        self.nn_m = Model(inputs=input, outputs=x)
        self.nn_m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    def fit(self, X, y):
        '''fit ensemble'''
        
        ## FIT TFIDF-CLASSIC MODEL WITH NB FEATURES
        self.fit_tfidf(X) # fit tfidf 
        X_tfidf = self.transform_tfidf(X) # transform X using tfidf
        self.fit_classic_m(X_tfidf,y) # fit classic model (with NB features) on transformed X
        
        ## FIT BIDIRECTIONAL LSTM
        # standard keras preprocessing
        self.fit_keras_tokenizer(X)
        X_keras = self.transform_keras_tokenizer(X) 
        # build rnn
        self.build_nn() 
        # fit rnn
        if self.nn_weights is None:
            self.nn_m.fit(X_keras,y, batch_size=32, epochs=2, validation_split = 0.1)
        else:
            self.nn_m.load_weights(self.nn_weights)
        
    def predict_proba(self, X):
        '''predict & output class probability'''
        X_tfidf = self.transform_tfidf(X) # transform X using tfidf
        classic_preds = self.classic_m.predict_proba(X_tfidf.multiply(self.r))[:,1]
        X_keras = self.transform_keras_tokenizer(X)
        nn_preds = self.nn_m.predict(X_keras, batch_size = 32)
        return (nn_preds[:,0]+classic_preds)/2

    def predict(self, X):
        '''predict & output class'''
        preds = self.predict_proba(X)
        preds[preds<=0.5] = 0
        preds[preds>0.5] = 1
        return preds
        
if __name__ == "__main__":
    print('loading data')
    train, valid = load_and_split()
    
    print('preprocessing')
    tokenizer = PatternTokenizer()
    train["comment_text"] = tokenizer.process_ds(train["comment_text"]).str.join(sep=" ")
    valid["comment_text"] = tokenizer.process_ds(valid["comment_text"]).str.join(sep=" ")

    labels = train.columns[2:]
    ys_train = train[labels]
    ys_valid = valid[labels]
    ## COMBINE TOXIC CATEGORIES
    y_train = ys_train.sum(axis=1)
    y_valid = ys_valid.sum(axis=1)
    y_train.loc[y_train>1] = 1
    y_valid.loc[y_valid>1] = 1
    
    print('fitting ensemble model')
    X = train['comment_text']
    ensemble = toxic_ensemble(nn_weights='artifacts/lstm_weights.h5')
    ensemble.fit(X, y_train)
    print('testing model')
    preds = ensemble.predict(valid['comment_text'])

    # Evaluate predictions
    acc, prec, recall, f1 = (accuracy_score(y_valid, preds), 
                            precision_score(y_valid, preds), 
                            recall_score(y_valid, preds), 
                            f1_score(y_valid, preds))

    # print results
    print('Validation Results for {0}: Accuracy - {1:.2f}; Precision - {2:.2f}; Recall - {3:.2f}; F1 - {4:.2f}'.format(
                                    'Toxic Comments', 
                                    acc, 
                                    prec, 
                                    recall,
                                    f1))
    # save ensemble model
    print('Saving model')
    ensemble.nn_m.save('model/nn_m')
    to_save = {'classic_m': (ensemble.classic_m,ensemble.r),
            'tfidf': ensemble.tfidf,
            'keras_tokenizer': ensemble.keras_tokenizer}
    for i in to_save.keys():
        dump(to_save[i],'model/'+i)