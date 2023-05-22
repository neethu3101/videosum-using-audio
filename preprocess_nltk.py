import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def clean(transcription):
    trans = [item for item in transcription.split('.') if item != '']
    w = []
    f= []
    for t in trans:
        words, filtered_words = preprocess(t)
        w.append(words)
        f.append(filtered_words)
    filt=[]
    for x in w:
        filt.append(' '.join(x))
    return filt


def preprocess(text):
    text = text.lower()
    
    #text_p = "".join([char for char in text if char not in string.punctuation])
    
    words = word_tokenize(text)
    
    stop_words = stopwords.words('english')
    filtered_words = [word for word in words if word not in stop_words]
    
    
    return words, filtered_words
