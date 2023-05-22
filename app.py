from flask import Flask, render_template,request,redirect,url_for
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from werkzeug.utils import secure_filename
from moviepy.editor import *
import os
import requests 
import preprocess_nltk as p
import textrank
import tf_idf


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = ['.mp4']

#create tokenizer
#p_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
#load model
#pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")


@app.route('/')
def welcome():
    return render_template('index.html')


@app.route('/summarize', methods=['POST'])
def summarize():

    file = request.files['file']
    filename = secure_filename(file.filename)
    f_name = os.path.splitext(filename)[0]
    extension = os.path.splitext(file.filename)[1]
    
    if extension not in app.config['ALLOWED_EXTENSIONS']:
        return 'File format not supported'
    file.save(os.path.join('uploads/',filename))

    video = VideoFileClip(os.path.join('uploads/',filename))
    #Audio Extraction
    video.audio.write_audiofile(os.path.join('uploads/',f_name+".mp3"))

    #Speech-to-Text
    token = "141294eeee7843dd88a548b882ee1233"
    filename = os.path.join('uploads/',f_name+".mp3")
    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    #Upload
    headers = {'authorization': token}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                        headers=headers,
                        data=read_file(filename))

    upload_url = response.json()['upload_url']
    endpoint = "https://api.assemblyai.com/v2/transcript"
    json = { "audio_url": upload_url }
    response = requests.post(endpoint, json=json, headers=headers)
    id  = response.json()['id']
    endpoint = "https://api.assemblyai.com/v2/transcript/"+id
    #Final speech-to-text output
    transcription = None
    while transcription == None:
        print("Loading.....")
        response = requests.get(endpoint, headers=headers)
        transcription = response.json()['text']
    print(transcription)
    
    if request.form['sb'] == "Summarize with TFIDF":
        sum,no = tf_idf.get_summary(transcription,0.25)


    else:
        #Text Preprocessing
        filt = p.clean(transcription)

        #Textrank
        extractive_sum = textrank.summarize(filt)
        print(extractive_sum)
        sum = extractive_sum.split('.')
        temp=[]
        for sentence in sum:
            temp.append(sentence[0].upper()+sentence[1:]+" ")

        print(temp)
        sum = '.'.join(temp)
    
    #Pegasus
    #def divide_chunks(l, n):
    #    for i in range(0, len(l), n):
    #        yield l[i:i + n]
 
    #chunks = list(divide_chunks(temp, 6))

    #def summarize_text(trans):
    #    tokens = p_tokenizer(trans,padding="longest",return_tensors="pt")
    #    summary = pegasus_model.generate(**tokens,max_new_tokens = 32)
    #   decode_sum = p_tokenizer.batch_decode(summary,skip_special_tokens=True)
    #    return decode_sum[0]


    #final = []
    #for i in chunks:
    #    final.append(summarize_text(i).split('<n>')[1])
    #    print("chunk done")
    #sum = ' '.join(final)

    
    return render_template('output.html',message =sum) 



if __name__ == '__main__':
    app.run(debug = True)