#! /usr/bin/env python3.6

import pandas as pd
import numpy as np
import ast

print(pd.__version__)

# Import conversation structures

delim= r"\s\+{3}\$\+{3}\s"

conversation_structure=pd.read_csv('cornell/movie_conversations.txt', header=None, 
                                   sep=delim,
                                   names=['charID_1', 'charID_2', 'movieID', 'utterances'],
                                   engine='python'
                                  )

conversation_structure.utterances = conversation_structure.utterances.apply(ast.literal_eval)

utterances=pd.read_csv('cornell/movie_lines.txt', header=None, 
                       sep=delim,
                       names=['lineID', 'charID', 'movieID', 'charName', 'text'],
                       engine='python')


ut = utterances.set_index('lineID')

df = pd.DataFrame(columns=['conversation']);

def link_utterances(utt_list):
    conv = []
    # use :2 to pick only the first QA 
    for l in utt_list[:2]:
        conv.append(ut.loc[l].text)
    return conv
        
df.conversation = conversation_structure.utterances.apply(link_utterances)

print("number of conversations: " + str(len(df.conversation)))

QA = pd.DataFrame(columns=list('QA'))
for i in df.conversation[:int(len(df.conversation)*0.5)]:
    for k in range(0,len(i)-1,1):
        QA=QA.append({'Q':i[k], 'A':i[k+1]}, ignore_index=True)


# export final structure
QA.to_csv('QA_red.csv', '|')