#Here, i want to create a txt embedding file with every word having a corresponding 8 dimension emotion embedding from the NRC-Emotion-Lexicon-Wordlevel-v0.92.txt

import pandas as pd
import numpy as np
import time
#Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', header= None, delimiter='\t') #, engine='python', encoding = 'latin')


##################################### STEP 1 ###########################################
#### HERE I REMOVED THE ROWS THAT CONTAIN POSITIVE, NEGATIVE VALUES as we dont need them
#part 1
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRC-Emotion-Lexicon-v0.92/Constraint1.txt', header= None, delimiter='\t') #, engine='python', encoding = 'latin')
# indexNames = Emolex[(Emolex[1].values=='negative')].index # or (Emolex[2].values == "0")].index
# print(indexNames)
# Emolex.drop(indexNames , inplace=True)
# Emolex.to_csv('Constraint1.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=True,mode='w', encoding=None,  line_terminator=None)

#part2
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='positive')].index # or (Emolex[2].values == "0")].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1a.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label= [0,1,2],mode='w', encoding=None,  line_terminator=None)
#final file renamed to NRCLexiconWithoutPositiveNegative.csv


# ##################################### STEP 2 ###########################################
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRCLexiconWithoutPositiveNegative.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# i = 0
# print(len(Emolex))
# with open('8d_EmotionalEmbeddings.txt', 'w') as f1:
#     #for i in range(k, k + 8):
#     #for i in range(len(Emolex)):
#     while i < len(Emolex):
#         x = i
#         vec = []
#         word= Emolex.iloc[i,1]
#         for x in range(x, x + 8):
#             score = Emolex.iloc[x,3]
#             vec.append(score)
#             vector=np.array(vec)
#         f1.write(word+ " ")
#         for j in range(len(vector)):
#             f1.write(str(vector[j]) + ' ')
#         f1.write('\n')
#             #f1.write(word + '' +vector)
#         i = i + 8
#
#
# #The output of this was used in TextClassification_8dEmoEmb for classification


#I also want to weight each word and thus its embedding in the NRC Emotion lexicon with an intensity score from NRC intensity lexicon
#thus for every word in the NRC emotion lexicon, find the word in the NRC intensity lexicon, get the associated emotion and its intensity. Multiply the intensity by the emotion appropriate dimension.
#Example, If NRC emotion lexison contains "go" with emotion embedding 1 0 1 0 0 0 0 0 and the first dimension is 'joy' emotion and the 3rd dimension is 'trust' emotion. The Intensity lexicon contains the word "go" with emotion joy and intensity 0.8
#Then the new embedding for go is 0.8 0 1 0 0 0 0  0

#Question? if a word in the Emotion lexicon contains 1 for two emotion and the intensity lexicon contains the intensity for one emotion, that means that the weighting will only affect one emotion. this might be detrimental because the unaffected emotion which remains 1 might truly have a lower intensity of it were present in the intensity lexicon


##################################### STEP 3 ###########################################
# t0=time.time()
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRCLexiconWithoutPositiveNegative.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# IntensityLex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRC-Emotion-Intensity-Lexicon-v1/NRC-Emotion-Intensity-Lexicon-v1.txt', header= None, delimiter='\t')
# for i in range(len(Emolex)):
#     word = Emolex.iloc[i, 1]
#     emotion = Emolex.iloc[i, 2]
#     for j in range(len(IntensityLex)):
#         word1 = IntensityLex.iloc[j, 0]
#         emotion1 = IntensityLex.iloc[j, 1]
#         if (word == word1) and (emotion == emotion1):
#             #MULTIPLY  Emolex.iloc[i, 2] with  IntensityLex.iloc[i, 0]
#             score=Emolex.iloc[i, 3]
#             intensity=  IntensityLex.iloc[j, 2]
#             intensity_score = score * intensity
#             Emolex.iloc[i, 4]= intensity_score
#             break
#         else:
#             Emolex.iloc[i, 4] =  Emolex.iloc[i, 3]
#             continue
# print ("file creation time:", round(time.time()-t0, 3), 's')
# Emolex.to_csv('NRCEmotion_withIntensity.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label= [0,1,2,4],mode='w', encoding=None,  line_terminator=None)
# #IntensityLex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRCLexiconWithoutPositiveNegative.csv', header= None, delimiter=',')


Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRCEmotion_withIntensity.csv', header= None, delimiter=',')
i = 0
print(len(Emolex))
with open('8d_IntensityWeighted_EmotionalEmbeddings.txt', 'w') as f1:
    #for i in range(k, k + 8):
    #for i in range(len(Emolex)):
    while i < len(Emolex):
        x = i
        vec = []
        word= Emolex.iloc[i,1]
        for x in range(x, x + 8):
            score = Emolex.iloc[x,4]
            vec.append(score)
            vector=np.array(vec)
        f1.write(word+ " ")
        for j in range(len(vector)):
            f1.write(str(vector[j]) + ' ')
        f1.write('\n')
            #f1.write(word + '' +vector)
        i = i + 8


#The output of this was used in TextClassification_8dEmoEmb for classification