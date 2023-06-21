import nltk
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def stop_words_count(words):
  count=0
  freq_words=["subscribe","welcome","hi","hello","me","my","enroll","like","channel","thank","upload"]
  for word in words:
    if word in freq_words:
      count= count+1
  return count


def clean(transcription):
    text = transcription.replace('\n',' ').replace('  ',' ').split('. ')
    w=[]
    f = []
    for i in text:
        words, filtered_words = preprocess(i)
        w.append(' '.join(words))
        f.append(' '.join(filtered_words))
    return w


def preprocess(text):
    text = text.lower()
    
    text_p = "".join([char for char in text if char not in string.punctuation])
    
    words = word_tokenize(text_p)
    count = stop_words_count(words)
    if count >1:
      return [],[]
    
    freq_words=["okay","subscribe","share","like","welcome","hi","hello","i","me","my","enroll","like","channel","thank","you","upload","link","support","oh","guys","video"]
    word_list = [word for word in words if word not in freq_words]
    
    stop_words = stopwords.words('english')
   
    filtered_words = [word for word in words if word not in stop_words and word not in freq_words]
    
    return word_list,filtered_words