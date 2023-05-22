import spacy
import pytextrank
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe("textrank")

def summarize(filt):
    transcription = '.'.join(filt)
    doc = nlp(transcription)
    for phrase in doc._.phrases:
        print(phrase.text)
        print(phrase.rank, phrase.count)
        print(phrase.chunks)
    sum=""
    for sent in doc._.textrank.summary(limit_phrases=15):
        sum = sum+ str(sent)

    return sum