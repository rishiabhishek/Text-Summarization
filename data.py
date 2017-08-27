import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
import re
from collections import Counter
from contractions import contractions



def load_data():
    reviews = pd.read_csv('./Reviews.csv')
    reviews.dropna()
    reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time'], 1)
    reviews = reviews.reset_index(drop=True)
    reviews = reviews[reviews.Summary.notnull()]
    return reviews
    
def clean_text(text,remove_stopwords=False):
    
    text = text.lower()
    clean_text = []
    for word in text.split():
        if word in contractions:
            clean_text.append(contractions[word])
        else:
            clean_text.append(word)
    text = " ".join(clean_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br', ' ', text)
    text = re.sub(r'/>', ' ', text)
    text = re.sub(r'>', ' ', text)
    text = re.sub(r'<', ' ', text)
    text = re.sub(r'`', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    return text
    


dataset = load_data()
print(clean_text(str(dataset.Text[713])))
