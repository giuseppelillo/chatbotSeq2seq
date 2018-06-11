#! /usr/bin/env python3.6

import sys
import re
import numpy as np
import keras
from keras.models import load_model
import os
import tensorflow as tf
import warnings
import configparser
import sklearn.metrics.pairwise as pairwise

config = configparser.ConfigParser()
config.read('chatbot.cfg')

warnings.filterwarnings( config['LOGS']['filter_warnings'] )
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config['LOGS']['tf_cpp_min_log_level']

max_seq_length = int(config['DEFAULT']['max_seq_length'])
temperature = int(config['DEFAULT']['temperature'])

l1 = config['FILTERS']['l1'].split('\t')
l2 = config['FILTERS']['l2'].split('\t')
l3 = config['FILTERS']['l3'].split('\t')
l4 = config['FILTERS']['l4'].split('\t')

tag_seq_start = config['TAGS']['seq_start']
tag_seq_end = config['TAGS']['seq_end']
tag_name = config['TAGS']['name']

# load vocab
with open('vocab_enc.csv') as venc:
    vocab_enc = venc.read().splitlines()

with open('vocab_dec.csv') as vdec:
    vocab_dec = vdec.read().splitlines()

# laod models
print("Loading encoder model")
encoder_model = load_model("encoder_model.h5")
print("Loading decoder model")
decoder_model = load_model("decoder_model.h5")

# load weights
print("Loading encoder weights")
encoder_model.load_weights("encoder_model_weights.h5")
print("Loading decoder weights")
decoder_model.load_weights("decoder_model_weights.h5")

# loading glove

def embedding_layer_from_vocab(vocab, EMBEDDING_DIM):

        embedding_matrix = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
        for i, word in enumerate(vocab):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

print("Loading glove")

GLOVE_DIR = 'glove'
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = embedding_layer_from_vocab(vocab_enc, 100)

print("Loading complete.")

def apply_temperature(distr, temp=1):
    distr = np.asarray(distr).astype('float64')
    distr[distr == 0] = 10**-10
    d = np.exp(np.log(distr) / temp)
    d = d / np.sum(d)
    
    return d

def get_most_similar(word):
    try:
        X = embeddings_index[word].reshape(1,-1)
    except KeyError:
        return None,0,0
    mval = 0.0
    mlab = None
    midx = None
    for k,v in enumerate(embedding_matrix):
        Y=v.reshape(1,-1)
        s = pairwise.cosine_similarity(X,Y)
        if s > mval and vocab_enc[k] != word and s > 0.5:
            mval = s
            mlab = vocab_enc[k]
            midx = k
    return mlab,midx,mval

def basic_tokenizer(line, normalize_digits=True):
    """ A basic tokenizer to tokenize text into tokens.
    Feel free to change this to suit your need. """
    line = re.sub('<u>', '', line)
    line = re.sub('</u>', '', line)
    line = re.sub('\[', '', line)
    line = re.sub('\]', '', line)
    words = []
    _WORD_SPLIT = re.compile(r"([.,!?\"'-<>:;)(])")
    _DIGIT_RE = re.compile(r"\d")
    
    for ltr in l3:
        line = line.replace(ltr,' ')
        
    for tbr,r in zip(l1,l2):
        line = line.replace(tbr,r)
    
    for fragment in line.strip().lower().split():
        for token in re.split(_WORD_SPLIT, fragment):
            if not token:
                continue
            if normalize_digits:
                token = re.sub(_DIGIT_RE, '#', token)
            if token in l4:
                token = tag_name
            words.append(token)
    return words

def decode_sequence(input_seq, temperature=0):
        
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0] = vocab_dec.index(tag_seq_start)
    
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        
        # Sample a token
        probs = apply_temperature(output_tokens[0, -1, :], temperature)
        #sampled_token_index = np.random.choice(range(len(probs)), p=probs)
        probs = np.random.multinomial(1, probs, 1)
        sampled_token_index = np.argmax(probs)
        sampled_word = vocab_dec[sampled_token_index]
        if sampled_word != tag_seq_end:
            decoded_sentence += ' ' + sampled_word

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == tag_seq_end or
           len(decoded_sentence) > max_seq_length):
            stop_condition = True
        
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]

    return decoded_sentence


# function to encode a new sentence
def encode_sentence(s):
    encoded_input = np.zeros((1,max_seq_length), dtype='int')
    splitted = basic_tokenizer(s)
    w = 0
    for t in splitted:
        try:
            tidx = vocab_enc.index(t)
        except ValueError:
            tlab,tidx,tsim = get_most_similar(t)
            if tidx == 0:
                print('Bot:I\'m sorry human,"'+t+'" is not in my dictionary')
                return None
            print('Bot:I\'m sorry human,"'+t+'" is not in my dictionary, but it is '+str(tsim[0][0]*100)+'% similar to "'+tlab+'"')
        
        encoded_input[0,w] = tidx
        w = w+1
    return encoded_input

temperature = float(input("Enter the temperature: "))

print("Waking up the chatbot...")
print("...")
print("...")
print("...")
print("...")

# snippet to read questions from stdin
for q in sys.stdin:
    encq = encode_sentence(q)
    if encq is not None:
        print("> " + decode_sequence(encq, temperature))
