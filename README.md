# Toxic Comment Classification using NLP (deployable as a REST API)
Deployable ensemble NLP model (BiLSTM + NB-SVM) that classifies comments as "toxic" [1] or "not toxic" [0].

---

## To pull and run app container locally:
Requirement: Docker
1. Pull container from Docker Hub:
    >docker pull guolin1/toxicityclassifierapi:latest
2. Run container:
    > docker run -p 5000:5000 guolin1/toxicityclassifierapi:latest
3. To query from Jupyter Notebook:
    >import requests <br /> 
    >url = 'http://localhost:5000' <br /> 
    >params ={'query': ['Comment 1', 'Comment 2']} <br /> 
    >response = requests.get(url, params) <br /> 
    >response.json()

---
## To deploy app on kubernetes (GKE)
1. Pull repository using Git:
    > git clone https://github.com/guolin1/ToxicityClassifier_RestAPI.git
2. Deploy (after CD into ToxicityClassifier_RestAPI/k8s_deploy Directory):
    > kubectl create -f deployment.yml -f service.yml
3. Obtain Access Url:
    > kubectl get services
    - URL is the EXTERNAL-IP for toxicityclassifier-service
4. To query from Jupyter Notebook:
    >import requests <br /> 
    >url = 'EXTERNAL-IP:80' <br /> 
    >params ={'query': ['Comment 1', 'Comment 2']} <br /> 
    >response = requests.get(url, params) <br /> 
    >response.json()

--- 

## [The App](https://github.com/guolin1/ToxicityClassifier_RestAPI/tree/master/app)
---

## Exploration Notebooks [[notebooks]](https://github.com/guolin1/ToxicityClassifier_RestAPI/tree/master/notebooks):
- Brief EDA [[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/0_eda.ipynb)
- Preprocessed Texts
- Explored Tfidf vectorizer (w/ SVM, NB-SVM, Logistric Regression, XGBoost)[[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/2_tfidf.ipynb)
- Explored pretrained GloVe vectorizer (w/ SVM, Logistric Regression, XGBoost)[[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/2_GloVe.ipynb)
- Explored pretrained Word2Vec vectorizer (w/ SVM, Logistic Regression, XGBoost)[[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/2_word2vec.ipynb)
- TF-IDF weighted GloVe vectors (w/ SVM, Logistic Regression, XGBoost)[[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/3_tfidf_GloVe.ipynb)
- TF-IDF weighted word2vec vectors (w/ SVM, Logistic Regression, XGBoost)[[notebook]](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/3_tfidf_word2vec.ipynb)
- GloVe vectors and Bidirectional LSTMs [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/4_GloVe_LSTM.ipynb)
- GloVe vectors and Bidirectional GRUs [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/4_GloVe_GRU.ipynb)
- Word2Vec vectors and Bidirectional LSTMs [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/4_word2vec_LSTM.ipynb)
- Word2Vec vectors and Bidirectional GRUs [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/4_word2vec_GRU.ipynb)
- Ensemble Model (GloVe + SVM, XGBoost, BiGRU) [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/6_ensemble.ipynb)
- Compare combinations of models and vectorizers [notebook](https://github.com/guolin1/ToxicityClassifier_RestAPI/blob/master/notebooks/5_compare_models.ipynb)
