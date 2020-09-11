# NLP Toxic Comment Classification
> Exploring NLP methods on a multi-label classification problem [[Data]](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Progress [[notebooks]](https://github.com/guolin1/NLP_ToxicCommentClassification/tree/master/notebooks):
- Brief EDA [[notebook]](https://github.com/guolin1/NLP_ToxicCommentClassification/blob/master/notebooks/0_eda.ipynb)
- Preprocessed Texts (Removed stop words, punctuations, digits, and Lemmatized) [[notebook]](https://github.com/guolin1/NLP_ToxicCommentClassification/blob/master/notebooks/1_preprocess.ipynb)
- Explored Tfidf vectorizer (w/ SVM, NB-SVM, Logistric Regression, XGBoost)[[notebook]](https://github.com/guolin1/NLP_ToxicCommentClassification/blob/master/notebooks/2_tfidf.ipynb)
- Explored pretrained GloVe vectorizer (w/ SVM, NB-SVM, Logistric Regression, XGBoost)[[notebook]](https://github.com/guolin1/NLP_ToxicCommentClassification/blob/master/notebooks/2_GloVe.ipynb)

## TO DO:
- Explore pretrained word2vec vectorizer
- Explore BERT, RNNs
- Additional preprocessing to consider: 
    - Group the same word written in different ways and/or written using symbols and numbers
    