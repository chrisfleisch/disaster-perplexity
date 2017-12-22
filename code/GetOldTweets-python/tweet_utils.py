#Sourced: https://gist.github.com/timothyrenner/dd487b9fd8081530509c
from datetime import datetime

import string

from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords

#Gets the tweet time.
def get_time(tweet):
    return datetime.strptime(tweet['created_at'], "%a %b %d %H:%M:%S +0000 %Y")

#Gets all hashtags.
def get_hashtags(tweet):
    return [tag['text'] for tag in tweet['entities']['hashtags']]

#Gets the screen names of any user mentions.
def get_user_mentions(tweet):
    return [m['screen_name'] for m in tweet['entities']['user_mentions']]

#Gets the text, sans links, hashtags, mentions, media, and symbols.
def get_text_cleaned(tweet):
    text = tweet['text']   
    # Sort the slices from highest start to lowest.
    slices = sorted(slices, key=lambda x: -x['start'])
    
    #No offsets, since we're sorted from highest to lowest.
    for s in slices:
        text = text[:s['start']] + text[s['stop']:]
        
    return text

#Sanitizes the text by removing front and end punctuation, 
#making words lower case, and removing any empty strings.
def get_text_sanitized(tweet):    
    return ' '.join([w.lower().strip().rstrip(string.punctuation)\
        .lstrip(string.punctuation).strip()\
        for w in get_text_cleaned(tweet).split()\
        if w.strip().rstrip(string.punctuation).strip()])

#Gets the text, clean it, make it lower case, stem the words, and split
#into a vector. Also, remove stop words.
def get_text_normalized(tweet):
    #Sanitize the text first.
    text = get_text_sanitized(tweet).split()
    
    #Remove the stop words.
    text = [t for t in text if t not in stopwords.words('english')]
    
    #Create the stemmer.
    stemmer = LancasterStemmer()
    
    #Stem the words.
    return [stemmer.stem(t) for t in text]