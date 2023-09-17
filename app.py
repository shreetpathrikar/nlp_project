from flask import Flask,render_template,Response,request
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import csv
from flask import make_response
import inflect 
q = inflect.engine() 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
from nltk.stem import wordnet 
from nltk import pos_tag 
from nltk import pos_tag, ne_chunk 
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk.probability import FreqDist
from nltk import FreqDist

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


import spacy
nlp = spacy.load("en_core_web_sm")


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html',style_path="/static/style.css")


def remove_num(text): 
    result = re.sub(r'\d+', '', text) 
    return result 


def convert_num(text):
    temp_string = text.split() 
    new_str = [] 
    for word in temp_string: 
        if word.isdigit(): 
            temp = q.number_to_words(word) 
            new_str.append(temp) 
        else: 
            new_str.append(word) 
    temp_str = ' '.join(new_str) 
    return temp_str 



def rem_punct(text): 
    translator = str.maketrans(' ', ' ', string.punctuation) 
    return text.translate(translator) 

def rem_stopwords(text): 
    stop_words = set(stopwords.words("english")) 
    word_tokens = word_tokenize(text) 
    filtered_text = [word for word in word_tokens if word not in stop_words] 
    return filtered_text 
    

def word_tokenizer(text):
    tokens = word_tokenize(text)
    return tokens

def sentence_tokenizer(text):
    sentences = sent_tokenize(text)
    return sentences

stem1 = PorterStemmer() 
def s_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stem1.stem(word) for word in word_tokens] 
    return stems 


lemma = wordnet.WordNetLemmatizer()
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    lemmas = [lemma.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas 
  
  
def pos_tagg(text): 
    word_tokens = word_tokenize(text) 
    return pos_tag(word_tokens) 


def ner(text): 
    word_tokens = word_tokenize(text) 
    word_pos = pos_tag(word_tokens) 
    return (ne_chunk(word_pos)) 


def word_count(text):
    words =word_tokenize(text)
    count = len(words)
    return count


def calculate_frequency(sentence):
    words = word_tokenize(sentence)
    fdist = FreqDist(words)
    word_count_list = [f"{word}: {count}" for word, count in fdist.items()]

    return word_count_list

# def calculate_frequency(sentence):
#   words = nltk.word_tokenize(sentence)
#   word_counts = {}
#   for word in words:
#     word_counts[word] = words.count(word)

#   for word, count in word_counts.items():
#     print(f"{word}: {count}")
    
#     return calculate_frequency


# def remove_num_def(txt):
#     code_string = '''
#     def remove_num(text): 
#     result = re.sub(r'\\d+', '', text) 
#     return result 
#     '''
#     return code_string


    


@app.route('/remove__num')
def user1_html():
    return render_template('remove__num.html')

@app.route('/')
def user2_html():
    return render_template('convert__num.html')

@app.route('/remove__punctuation')
def user3_html():
    return render_template('remove__punc.html')

@app.route('/remove__stopwords')
def user4_html():
    return render_template('remove__stop.html')

@app.route('/word_tokenizer')
def user5_html():
    return render_template('word_tokenize.html')

@app.route('/sentence_tokenizer')
def user6_html():
    return render_template('sentence_tokenize.html')

@app.route('/stemming')
def user7_html():
    return render_template('stemming_.html')

@app.route('/lemmitizing')
def user8_html():
    return render_template('lemmitizing.html')
 

@app.route('/POSTAG')
def user9_html():
    return render_template('pos_tagging.html')
 
@app.route('/NER')
def user10_html():
    return render_template('ner_.html')

@app.route('/word_count')
def user11_html():
    return render_template('word_count.html')


@app.route('/frequency_dist')
def user12_html():
    return render_template('feq_dis.html')


@app.route('/process', methods=['POST'])
def process():
    
    user1_input = request.form['user1_input']
    # user1_view = request.form['user1_view']
    
    user2_input = request.form['user2_input']
    user3_input = request.form['user3_input']
    user4_input = request.form['user4_input']
    user5_input = request.form['user5_input']
    user6_input = request.form['user6_input']
    user7_input = request.form['user7_input']
    user8_input = request.form['user8_input']
    user9_input = request.form['user9_input']
    user10_input = request.form['user10_input']
    user11_input = request.form['user11_input']
    user12_input = request.form['user12_input']
    
    
    
    
    
    processed_input1 = remove_num(user1_input)
    # processed_user1_view = remove_num_def(user1_view)
    processed_input2 = convert_num(user2_input)

    processed_input3 = rem_punct(user3_input)
    processed_input4 = rem_stopwords(user4_input)
    processed_input5 = word_tokenizer(user5_input)
    processed_input6 = sentence_tokenizer(user6_input)
    processed_input7 = s_words(user7_input)
    processed_input8 = lemmatize_word(user8_input)
    processed_input9 = pos_tagg(user9_input)
    processed_input10 = ner(user10_input)
    processed_input11 = word_count(user11_input)
    processed_input12 = calculate_frequency(user12_input )
    
    
    
    
    
    with open('history.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([user1_input, processed_input1,
                            user2_input, processed_input2,
                            user3_input, ', '.join(processed_input3),
                            user4_input, ', '.join(processed_input4),
                            user5_input, ', '.join(processed_input5),
                            user6_input, ', '.join(processed_input6),
                            user7_input, ', '.join(processed_input7),
                            user8_input, ', '.join(processed_input8),
                            user9_input, processed_input9,
                            user10_input, processed_input10,
                            user11_input, processed_input11,
                            user12_input, processed_input12,]) 
        
        

    
    if processed_input1:
        return f"Processed input: {processed_input1}"
    # elif processed_user1_view:
    #     return  processed_user1_view
        
    elif processed_input2:
        return f"Processed input: {processed_input2}"
    elif processed_input3:
        return f"Processed input: {processed_input3}"
    elif processed_input4:
        return f"Processed input: {processed_input4}"
    elif processed_input5:
        return f"Processed input: {processed_input5}"
    elif processed_input6:
        return f"Processed input: {processed_input6}"
    elif processed_input7:
        return f"Processed input: {processed_input7}"
    elif processed_input8:
        return f"Processed input: {processed_input8}"
    elif processed_input9:
        return f"Processed input: {processed_input9}"
    elif processed_input10:
        return f"Processed input: {processed_input10}"
    elif processed_input11:
        return f"Processed input: {processed_input11}"
    elif processed_input12:
        return f"Processed input: {processed_input12}"

    
    else:
        print("something went wrong try again")
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
@app.route('/spacy/')
def spacy():
    return render_template('spacy.html',style_path="/static/style.css")

import spacy




def token_spc(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens

def pos_tagging(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    pos_tags = []
    for token in doc:
        pos_tags.append((token.text, token.pos_))
    return pos_tags
       
        
        
        
        
        
        
        
@app.route('/spacyy', methods=['POST'])
def spacyy(): 
    
    user13_input = request.form['user13_input']
    user14_input = request.form['user14_input']
    
    
    processed_input13 = token_spc(user13_input)
    processed_input14 = pos_tagging(user14_input)
    
    
    
    if processed_input13:
        return f"Processed input: {processed_input13}"
    elif processed_input14:
        return f"Processed input: {processed_input14}"
    else:
        print("try again ") 
    
    
        
        


    # result = (
    #     f"Processed input 1: {processed_input1}<br>"
    #     f"Processed input 2: {processed_input2}<br>"
    #     f"Processed input 3: {', '.join(processed_input3)}<br>"
    #     f"Processed input 4: {', '.join(processed_input4)}<br>"
    # )
    
    # return result







if __name__=="__main__":
    app.run(debug=True)
 