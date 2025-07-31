import pandas
import numpy
import pickle
import sklearn
import string
import streamlit as st
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('stopwords')

tfidf = pickle.load(open('vectorizerSMS.pkl','rb'))
model = pickle.load(open('modelSMS.pkl','rb'))


st.title('SMS Spam Classifier')
input_sms = st.text_input("Enter SMS here....")

# now trasform that inputed data
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)
if st.button('predict'):
    # preprocess 
    transformed_sms = transform_text(input_sms)

    # vectorize

    vectorized = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vectorized)[0]

    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")