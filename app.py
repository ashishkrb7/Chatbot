# -*- coding: utf-8 -*-
"""
@author: Ashish Kumar
"""
#Python code for "RNN bot" API
from flask import Flask, render_template, request
import json,logging
app=Flask(__name__) 
import pickle
import tensorflow as tf # Version 2.0.0
from tensorflow.keras import preprocessing
from SentimentClassifier import Sentiment
from textblob import TextBlob
import wikipedia
import numpy as np
import re
import sys
import json
DEFAULTstr=" "
with open('./model10102019/tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
enc_model1=tf.keras.models.load_model( './model10102019/enc_model.h5', compile=False)  
dec_model1=tf.keras.models.load_model( './model10102019/dec_model.h5', compile=False) 
with open('./model25102019/tokenizer.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)
enc_model2=tf.keras.models.load_model( './model25102019/enc_model.h5', compile=False)  
dec_model2=tf.keras.models.load_model( './model25102019/dec_model.h5', compile=False) 
# Flask application
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
    
@app.route('/Bot',methods=['POST'])
def Bot():
    sentences = str(request.form['Sentence'])
    prediction=Sentiment(sentences).process()
    mydict1=tokenizer1.word_index
    vocab1=list(mydict1.keys())
    mydict2=tokenizer2.word_index
    vocab2=list(mydict2.keys())
    maxlen_answers=28 # Do not change these parameters
    maxlen_questions=16 # Do not change these parameters
    def str_to_tokens(sentence):
        words = sentence.lower().split()
        tokens_list = list()
        for word in words:
            tokens_list.append( tokenizer.word_index[ word ] )
        return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=maxlen_questions , padding='post')
    data=sentences
    data=data.lower()
    data = re.sub('[^a-zA-Z0-9]', ' ', data)
    myString1=str(TextBlob(data).correct())
    #myString1=data
    myString=myString1.split()
    try:
       if set(myString)<=set(vocab1):
           tokenizer=tokenizer1
           enc_model=enc_model1
           dec_model=dec_model1
           states_values = enc_model.predict( str_to_tokens(myString1) )    
           empty_target_seq = np.zeros( ( 1 , 1 ) )
           empty_target_seq[0, 0] = tokenizer.word_index['start']
           stop_condition = False
           decoded_translation = ''
           while not stop_condition :
                dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
                sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                sampled_word = None
                for word , index in tokenizer.word_index.items() :
                     if sampled_word_index == index :
                          decoded_translation += ' {}'.format( word )
                          sampled_word = word
                if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                     stop_condition = True
                empty_target_seq = np.zeros( ( 1 , 1 ) )  
                empty_target_seq[ 0 , 0 ] = sampled_word_index
                states_values = [ h , c ] 
           decoded_translation = re.sub(r"end", "", decoded_translation).strip()# remove this particular sybmols

           Bot = decoded_translation.capitalize()
           solution={"ans":Bot,"source": "RNN1","bot_question":myString1}
       elif set(myString)<=set(vocab2):
           tokenizer=tokenizer2
           enc_model=enc_model2
           dec_model=dec_model2
           states_values = enc_model.predict( str_to_tokens(myString1) )    
           empty_target_seq = np.zeros( ( 1 , 1 ) )
           empty_target_seq[0, 0] = tokenizer.word_index['start']
           stop_condition = False
           decoded_translation = ''
           while not stop_condition :
                dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
                sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                sampled_word = None
                for word , index in tokenizer.word_index.items() :
                     if sampled_word_index == index :
                          decoded_translation += ' {}'.format( word )
                          sampled_word = word
                if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                     stop_condition = True
                empty_target_seq = np.zeros( ( 1 , 1 ) )  
                empty_target_seq[ 0 , 0 ] = sampled_word_index
                states_values = [ h , c ] 
           decoded_translation = re.sub(r"end", "", decoded_translation).strip()# remove this particular sybmols

           Bot = decoded_translation.capitalize()
           solution={"ans":Bot,"source": "RNN2","bot_question":myString1}
       elif set(myString)!=set(vocab1) & set(myString)!=set(vocab2):
            try:
                 response=wikipedia.summary(data, sentences=2)
                 Bot = response
                 solution={"ans":Bot,"source": "Wiki","bot_question":data}
            except:
                 Bot = DEFAULTstr
                 solution={"ans":Bot,"source" : "Default when wiki fail","bot_question":myString1}
       else:
            Bot = DEFAULTstr
            solution={"ans":Bot,"source": "Default if everything fail","bot_question":myString1}
    except:
         Bot = "Oops! " + str(sys.exc_info()[0]) + " occured."
         solution={"ans":Bot,"source": "Program error","bot_question":myString1}
    finally:
        f=open("./storage.txt","a+",encoding="utf-8")
        f.write(str(json.dumps(solution)) +"{ user_question: "+data+" }" "\n")
        f.close()
        		 
    return(render_template('index.html',Input=f"You: {sentences}",Output=f"Bot: {solution.get('ans')}",Prediction=f"Sentiment: {prediction}"))
   
logging.basicConfig(filename='API.log',level=logging.DEBUG)
if __name__=='__main__':
	app.run(debug=True)
