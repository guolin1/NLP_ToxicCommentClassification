# NLP Toxic Comment Classification
> Exploring NLP methods on a multi-label classification problem [[Data]](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

## Progress [[notebooks]](https://github.com/guolin1/NLP_ToxicCommentClassification/tree/master/notebooks):
- Brief EDA
- Preprocessed Texts (Removed stop words, punctuations, digits, and Lemmatized)
- Explored Tfidf vectorizer (w/ SVM, NB-SVM, Logistric Regression, XGBoost)
- Explored pretrained GloVe vectorizer (w/ SVM, NB-SVM, Logistric Regression, XGBoost)

## TO DO:
- Explore pretrained word2vec vectorizer
- Explore BERT, RNNs
- Additional preprocessing to consider: 
    - Group the same word written in different ways and/or written using symbols and numbers
    