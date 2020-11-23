import nltk
import pickle
import numpy
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from keras.models import load_model
from model_files import chatbot_model4,covid1,words4,classes4
model = load_model('chatbot_model4.h5')
import json
import random
import numpy as np
intents = pickle.load(open('intents.pkl','rb'))
words = pickle.load(open('words4.pkl','rb'))
classes = pickle.load(open('classes4.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

from flask import Flask, render_template, request, jsonify
app = Flask(__name__,template_folder='templates')

@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def get_bot_response():
    message = request.args.get('msg')
    
    p = bow(message, words,show_details=False)
    results = model.predict(np.array([p]))[0]
    results_index = numpy.argmax(results) 
    tag = classes[results_index] 
    if results[results_index] > 0.5: 
        for tg in intents["intents"]: 
            if tg['tag'] == tag: 
                responses = tg['responses'] 
        res="COVID-BOT:"+random.choice(responses)
        return str(res)
    else: 
        res="COVID-BOT:I am sorry but I can't understand"
        return str(res)
   
    return "Missing Data!"

if __name__ == "__main__":
	app.run(debug=True)