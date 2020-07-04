from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 
from textblob import TextBlob

class Sentiment:

    '''Python code for extracting the error line in the logs using assumption that error log will have negative sentiment'''
        
    def __init__(self,text):
        self.text=text
    
    def process(self):

        def VaderSentiment(sentence):  
            """Sentiment analysis using Vader model"""
            sid_obj = SentimentIntensityAnalyzer() # polarity_scores method of SentimentIntensityAnalyzer 
            sentiment_dict = sid_obj.polarity_scores(sentence)
            #In this section we can do more division based on literatures if we need to define the sentiment into more category
            if sentiment_dict['compound'] >= 0.05 : 
                a="Positive"

            elif sentiment_dict['compound'] <= - 0.05 : 
                a="Negative" 

            else : 
                a="Neutral" 
            return(a)

        def BlobSentiment(sentences):
            """ Sentiment analysis using textblob pretrained model"""
            sentiment = TextBlob(sentences)
            y=sentiment.sentiment.polarity
            if y>0:
                z="Positive"

            elif y<0:
                z="Negative"

            else:
                z="Neutral"

            return(z) 

    
        z=BlobSentiment(self.text)
        if z=="Neutral":
            z=VaderSentiment(self.text) 
        else:
            z=z
        return(z)
