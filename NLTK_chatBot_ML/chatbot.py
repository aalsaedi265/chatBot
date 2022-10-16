
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json

with open("intent.json")as file: #opening json and  checking data is connected to code
    data = json.load(file) 
    # print(data['intents'])
    
words=[]
labels=[]
docs_x=[]#takes patterns
docs_y=[]#takes intent


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
#n_epoch is the number of times it get to see the data, to better clasify
model.fit(training, output, n_epoch=1000,batch_size=8, show_metric=True)
model.save("model.tflearn")