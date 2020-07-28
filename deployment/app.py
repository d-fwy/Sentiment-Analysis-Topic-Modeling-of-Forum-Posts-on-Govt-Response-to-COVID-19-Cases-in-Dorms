#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[2]:


app = Flask(__name__)

@app.route('/')

def home():
    return render_template('home.html')


# In[3]:


@app.route('/predict',methods=['POST'])

def predict_topic():

    lda_model = open('lda_model.pkl','rb')
    lda_mallet = joblib.load(lda_model)
    
    cvec = open('cvec.pkl', 'rb')
    cvec = joblib.load(cvec)
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cvec.transform(data).toarray()
        prediction = lda_mallet.predict(vect)

    return render_template('result.html',prediction = prediction)


if __name__ == '__main__':
    app.run()


# In[ ]:




