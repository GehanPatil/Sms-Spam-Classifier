import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download("wordnet")
nltk.download("omw-1.4")
from nltk.stem import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# Lemmatization
def transform_lema(text):
    text = text.lower() # converting to lower
    text = nltk.word_tokenize(text) # tokenizing the words
    
    y = []
    for i in text:
        if i.isalnum(): # removing special charaters
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(lm.lemmatize(i))
    
            
    return " ".join(y)


# import models
model = pickle.load(open('model.pkl','rb'))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
lm = WordNetLemmatizer()



st.title("Sms spam Detector")

input_sms = st.text_area("Enter the message")


if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_lema(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
        


 
       
    st.write("## Thank you for Visiting \nProject by Gehan P")
