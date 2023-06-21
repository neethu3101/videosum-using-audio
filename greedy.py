import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import preprocess_nltk as p

def greed_sum(text, num_sent, min_df=1, max_df=1.0):
        
    #fit a TFIDF vectorizer
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    vectorizer.fit(text)
    
    #get the matrix
    X = vectorizer.transform(text).toarray()
    
    #get the sentence indices
    idx = []
    while sum(sum(X)) != 0:
        ind = np.argmax(X.sum(axis=1))
        idx.append(ind)

        #update the matrix deleting the columns corresponding to the words found in previous step
        cols = X[ind]
        col_idx = [i for i in range(len(cols)) if cols[i] > 0]
        X = np.delete(X, col_idx, 1)
        
           
    idx = idx[:num_sent]
    idx.sort()
    
    summary = [text[i] for i in idx]
    
    return summary
    
def greed_main(path):
    
    MIN_DF = 0.03 #the default min_df   
    MAX_DF = 0.5 #the default max_df       
    NUM_SENT = 20 #the default max_df   
        
    # 1. Get the text from the file provided
    with open(path, "r", encoding="UTF-8") as f:
        text = ' '.join([l.strip() for l in f.readlines()]).replace('\n',' ').replace('  ',' ')
    f.close()

    #preprocess
    clean_text= p.clean(text)
    len_trans= len(clean_text)
    # 2. Summarize
    summary = greed_sum(clean_text, NUM_SENT, min_df=MIN_DF, max_df=MAX_DF)
    len_sum = len(summary)
    temp=[]
    for i in summary:
      temp.append(i[0].upper() + i[1:])
    return '. '.join(temp),len_trans,len_sum
