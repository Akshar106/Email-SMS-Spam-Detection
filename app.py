import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))




st.markdown("<p style='color: #FF7F50; font-size: 36px;'>Email/SMS Spam Classifier</p>", unsafe_allow_html=True)

input_sms = st.text_area("Enter the message")

if st.button('Predict', key='predict-button'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.markdown("<p style='color: #FF7F50; font-size: 36px;'>The Message You Entered is Spam</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #FF7F50; font-size: 36px;'>The Message You Entered is Not Spam</p>", unsafe_allow_html=True)

