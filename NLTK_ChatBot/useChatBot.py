import random
import json
import pickle
from urllib import response #serilzations
import numpy as np
import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer #to the root of words

from tensorflow.python.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl","rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('chatbot_model.model')

def cleanUpSentence(sen):
    sentence_word = nltk.word_tokenize(sen)
    sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
    return sentence_word

def bag_Word(sen):
    sentence_word = cleanUpSentence(sen)
    bag = [0]*len(words)
   
    for w in sentence_word:
        for idx, word in enumerate(words):
            if word == w:
                bag[idx]=1
    return np.array(bag)


def prefect_class(sen):
    bag = bag_Word(sen)
    response= model.predict(np.array([bag]))[0]
    error_threshold= 0.25
    results = [[i,r] for i,r in enumerate(response) if r > error_threshold]
    
    results.sort(key = lambda x: x[1], reverse=True)
    return_list =[]
    for r in results:
        return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result
    

print('-------------The machine----------------')

while True:
    message = input("you: ")
    ints = prefect_class(message)
    res = get_response(ints, intents)
    print(res)