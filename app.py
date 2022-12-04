import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("/Users/mac/spam classifier-/ernon.csv", encoding= 'latin-1')

data.head()
data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4','Unnamed: 5'],inplace=True)
data.dropna()

data1= data.where((pd.notnull(data)),'')

data1 = data1.drop_duplicates(keep='first')

data1 = data1[["Label", "Email"]]


x = np.array(data1["Email"])
y = np.array(data1["Label"])

cv = CountVectorizer()
X = cv.fit_transform(x) # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = MultinomialNB()
clf.fit(X_train,y_train)
import streamlit as st
st.title("Spam classifier")
def spamdetection():
    user = st.text_area("Enter any Message or Email: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        data = cv.transform([sample]).toarray()
        a = clf.predict(data)
        st.title(a)
spamdetection()