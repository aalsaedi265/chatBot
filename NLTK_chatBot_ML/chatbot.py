
from operator import mod
import re
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intent.json")as file: #opening json and  checking data is connected to code
    data = json.load(file) 
    # print(data['intents'])

#model exists go to training 
try:
    with open('data.pickle','rb') as f:
        words, labels,training, output = pickle.load(f)
#deos not exists run creation code
except:    
    words=[]
    labels=[]
    docs_x=[]#takes patterns
    docs_y=[]#takes intent

#if no item add, will need delete the old json, and run the code, with out try
#except to train on the new dat
    for intent in data['intents']:
        for pattern in intent["patterns"]:
            wrds= nltk.word_tokenize(pattern)#got all works in list string, in patterns
            words.extend(wrds)#varous lists are mering into one per section
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])#access tags such as age, name, gestoins

    #iterate over all words and make them lower case 
    words = [stemmer.stem(w.lower()) for w in words if w != '?'] #if statment remove question marks
    words = sorted(list(set(words))) #sort alphabtically

    labels= sorted(labels)

    training =[]
    output=[]

    out_empty =[0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag =[]#interacts with tokenized doc X and Y
        
        wrds = [stemmer.stem(w) for w in doc]
        
        for w in words:
            if w in wrds:
                bag.append(1)#word does exsist
            else:
                bag.append(0) #word does not existi
        
        output_row = out_empty[:]#clone using slice
        output_row[labels.index(docs_y[x])]=1
        
        training.append(bag)
        output.append(output_row)
        
    
    #formum taken by model
    training = numpy.array(training)
    output= numpy.array(output)
    
    with open('data.pickle', 'wb') as f:#use pickel to save the code above in pckel
        pickle.dump((words, labels,training, output) ,f)#designed to increae speed.

#resetes underline  data
tensorflow.compat.v1.reset_default_graph()
#-----AI-----

#defiens input shape for model, len(training[0]) is what length will be expected
net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net,8) #provided 8 nurones for the training[0]
net = tflearn.fully_connected(net,8)#both hidden layyers, full connectd to in and out
#provide nurones for output, actiation provied probaliityes for output
net = tflearn.fully_connected(net,len(output[0]),activation = 'softmax')
#softmax has 6 nueral networks instead of 8
net= tflearn.regression(net)

model = tflearn.DNN(net)#utlizes the network
#-----AI  ENDs-----


try:
    model.load('model.tflearn')
except:
    #n_epoch is the number of times it get to see the data, to better clasify
    model.fit(training, output, n_epoch=1000,batch_size=8, show_metric=True)
    model.save("model.tflearn")
    

def turn_sentenceFromUser_to_BagOfWOrds(s,words):#Robot only uinderstanding workd Bags
    
    bag=[0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for idx, wrd in enumerate(words):
            if wrd == se:
                bag[idx]= 1#reaplce zero with one
    return numpy.array(bag)                
    
    
def ask_user_for_sentence_and_produce_outPut():
    print("Begin you'er communication with the T800 (conclude discourse by sayin aaagggaahh)")
    
    while True:
        inp = input("Human_MeatBag: ")
        if inp.lower()=="aaagggaahh":
            break
        #will give the out put, t it will be number format not human readable
        results=model.predict([turn_sentenceFromUser_to_BagOfWOrds(inp, words)])
        results_idx = numpy.argmax(results)#returns most likeable repsone in number form
        tag = labels[results_idx]#returns tag of the response, in english
        print(tag)
ask_user_for_sentence_and_produce_outPut()