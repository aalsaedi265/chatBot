import random
import json
import pickle #serilzations
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer #to the root of words

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import gradient_descent_v2

lemmatizer = WordNetLemmatizer
intents = json.loads(open('intents.json').read( ))

words =[]
classes =[]
documents=[]
ignore_letter =['?','!','.',',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)#given string seperate by white space
        words.append(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)