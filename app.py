import streamlit as st
import contractions
st.title('Sentiment Analysis')
import pandas as pd
import re
def special_removal(text):
  pattern=r'[^A-Za-z\s]'
  text=re.sub(pattern,' ',text)
  return text
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer=ToktokTokenizer()
nltk.download('stopwords')
stopword_list=nltk.corpus.stopwords.words('english')
len(stopword_list)
stopword_list.remove('no')
stopword_list.remove('not')
def stopwords_removal(text):
  tokens=tokenizer.tokenize(text)
  tokens=[token.strip() for token in tokens]
  filtered_tokens=[token for token in tokens if token not in stopword_list]
  filtered_text=' '.join(filtered_tokens)
  return filtered_text
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stemmer=PorterStemmer()
def stemming(text):
  updated=[]
  str=nltk.word_tokenize(text)
  for word in str:
   updated.append(stemmer.stem(word))
  sentence=' '.join(updated)
  return sentence
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
def lemmatize(text):
  str=nltk.word_tokenize(text)
  updated=[]
  for word in str:
    updated.append(lemmatizer.lemmatize(word))
  sentence=' '.join(updated)
  return sentence
df=pd.read_csv('https://raw.githubusercontent.com/puneethrajidi/Sentiment-analysis/main/Updated_dataset.csv')
x=df['review']
y=df['Three sentiments']
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
text_model=Pipeline([('tfid',TfidfVectorizer()),('model',SVC())])
text_model.fit(x,y)
select=st.text_input('Enter your msg')
select=contractions.fix(select)
select=special_removal(select)
select=stopwords_removal(select)
select=stemming(select)
select=lemmatize(select)
op=text_model.predict([select])
st.title(op)
