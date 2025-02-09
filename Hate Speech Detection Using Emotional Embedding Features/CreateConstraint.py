#Here i want to creaTE TWO CONSTRAINTS FOR INJECTING  emotions into embeddings for hate speech detection.
#The first constraint is from a lexicon that contains words with an association with the emotions (fear, anger, disgust)
#The second constraint is from a lexicon that contains words with an association with the emotions (anticipation, joy, sadness, surprise, trust)
#The third constraint is the antonyms of words in Constraint1 and the emotion

# I only used the first constraint and The third constraint in counter fitting my vectors

'''
#this block of code wasnt used anymore
import pandas as pd
Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', header= None, delimiter='\t') #, engine='python', encoding = 'latin')
Constraint1 = pd.DataFrame(columns=['0', '1'])
Constraint2 = pd.DataFrame(columns=['0', '1'])

i = 0
for j in range(7464):
    #print("i",i)
    for i in range(i, i+10):
        if ((Emolex.iloc[i,[1]].values== "fear") and (Emolex.iloc[i,[2]].values== "1")) \
                or ((Emolex.iloc[i,[1]].values == "anger") and (Emolex.iloc[i,[2]].values== "1"))  \
                or ((Emolex.iloc[i,[1]].values == "disgust") and (Emolex.iloc[i,[2]].values== "1")) :  #df.isin(["test"]).any().any()
            print('true')
            #print(Emolex.iloc[i,[2]].values)
            Constraint1 = Constraint1.append(Emolex.iloc[i,[0,1]], ignore_index=True)

        elif ((Emolex.iloc[i, [1]].values == "anticipation") and (Emolex.iloc[i,[2]].values== "1"))  \
                or ((Emolex.iloc[i, [1]].values == "joy")  and (Emolex.iloc[i,[2]].values== "1")) \
                or ((Emolex.iloc[i, [1]].values == "sadness") and (Emolex.iloc[i,[2]].values== "1"))  \
                or ((Emolex.iloc[i, [1]].values == "trust") and (Emolex.iloc[i,[2]].values== "1")) :  # df.isin(["test"]).any().any()
            print('true')
            #print(Emolex.iloc[i, [2]].values)
            Constraint2 = Constraint2.append(Emolex.iloc[i, [0, 1]], ignore_index=True)
        else:
            continue
    i = i + 10

'''



import pandas as pd
#eACH SECTION IS RUN SEPERATLEY AND EACH PART IN EACH SECTION IS RUN ONE AT ATIME

###### HERE I REMOVED THE ROWS TJHAT CONTAIN POSITIVE, NEGATIVE AND 0 VALUES
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

#part 3
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1a.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[3].values==0)].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1c.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)
# # final file renamed EmoLex_AssociationIs1.csv





############create the constraint1
#Part1
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/EmoLex_AssociationIs1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='anticipation')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part1
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='trust')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1a.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part2
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1a.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='sadness')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1b.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part3
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1b.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='joy')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1c.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part4
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1c.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='surprise')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint1d.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)
#final file is renamed to Constraint1





#create the constraint2
#oart1
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/EmoLex_AssociationIs1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='fear')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint2.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part1
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint2.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='disgust')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint2a.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)

#part2
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint2a.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = Emolex[(Emolex[2].values=='anger')].index # or (Emolex[2].values=='trust') or (Emolex[2].values=='sadness') or(Emolex[2].values=='joy') or (Emolex[2].values=='surprise')].index
# Emolex.drop(indexNames1 , inplace=True)
# Emolex.to_csv('Constraint2b.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)
#final file is renamed to Constraint2


#If theres a word that exist in both Contraint1 and 2 should htey be removed or not. WHy??

#dropping unecessary colums   #removes the index and label columns leaving only the word pairs
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# Emolex1 =Emolex.drop([0, 3], axis=1)
# Emolex1.to_csv('Constraint1a.csv', sep=',', na_rep='', float_format=None, columns=None, header=True, index=False , index_label= [0,1,2], mode='w', encoding=None,  line_terminator=None)
#

### COnvert the Constraint1a.csv file to a txt file by simply renaming it to a txt file then do a find and replace with , to white space. the copy the file the lingiustic_constraint folder

########Creating the second constraint which is the antonyms of words in Constraint1 and the emotion.
## Step1 ## Using WordNet Antonyms. this didnt work because the words in EMolex weretn found in word net antonyms
# Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# with open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/wordnet_antonyms.txt', 'r') as f1:
#     for i in range(len(Emolex)):
#         #print(Emolex.iloc[i, [1]].values)
#         entry = Emolex.iloc[i, [1]].values[0]
#         #print(entry)
#         for lines in f1:
#             antonym1 = lines.split(' ')[0]
#             antonym2 = lines.split(' ')[1].replace('\n','')
#             file2 = open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/AntonymConstraint.txt', 'w')
#             if (entry == antonym1):
#                 word = Emolex.iloc[i,[2]].values
#                 #print(Emolex.iloc[i,[1]].values)
#                 print('here')
#                 file2.write(antonym2 + word)
#             elif (entry == antonym2):
#                 #print(Emolex.iloc[i,[1]].values)
#                 file2.write(antonym1 + word)
#             else:
#                 continue

## Step1 ## We then used NLTK to generate the antonums for the words in EmoLEx
import nltk
from nltk.corpus import wordnet

synonyms = []

Emolex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
for i in range(len(Emolex)):
    antonyms = []
    entry = Emolex.iloc[i, [1]].values[0]
    for syn in wordnet.synsets(entry):
        for l in syn.lemmas():
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    with open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/AntonymConstraint_v2.txt','a') as file2:
        emotion = Emolex.iloc[i,[2]].values[0]
        for j in range(len(antonyms)):
            file2.write(antonyms[j] + " " + emotion + '\n')
# print(set(synonyms))
# print(set(antonyms))



###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
################################################### CREATING CONSTRAINT FROM HATE SPEECH LEXICON  ##########################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################
###############################################################################################################################################

#### Step 1: Remove the underscore in base lexicon
# HSLex = pd.read_csv(r'/home/kosimadukwe/Downloads/Hate Speech Lexicon/Wiegand Lexicon/Lexicons/baseLexicon_copy.txt', header= None, delimiter='\t') #, engine='python', encoding = 'latin')
# HSLex["2"] = ""
# for i in range(len(HSLex)):
#     word_pos= HSLex.iloc[i,0]
#     word = word_pos.split('_')[0]
#     HSLex.iloc[i, 2] = word
# HSLex.to_csv(r'/home/kosimadukwe/Downloads/Hate Speech Lexicon/Wiegand Lexicon/Lexicons/baseLexicon_copy.csv',index=False)


####Step 2: Select only words labelled as abusive. That is drop rows with False label

# HSLex = pd.read_csv(r'/home/kosimadukwe/Downloads/Hate Speech Lexicon/Wiegand Lexicon/Lexicons/baseLexicon_copy.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = HSLex[(HSLex[1].values=='False')].index
# HSLex.drop(indexNames1 , inplace=True)
# print(len(HSLex))
# HSLex.to_csv('HSLex_Constraint1.csv', index=False)    #sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label= [0,1,2],mode='w', encoding=None,  line_terminator=None)

### Step 3a: Find the synonyms of these abusive words. These will be pushed together in the vector space. Find the antonyms of these abusive words. These will be pushed apart in the vector space
# from nltk.corpus import wordnet
# HSLex = pd.read_csv(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/HSLex_Constraint1.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# for i in range(len(HSLex)):
#     synonyms = []
#     antonyms = []
#     entry = HSLex.iloc[i, [2]].values[0]
#     for syn in wordnet.synsets(entry):
#         for l in syn.lemmas():
#             synonyms.append(l.name())
#             if l.antonyms():
#                 antonyms.append(l.antonyms()[0].name())
#     with open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/HSLex_SynonymConstraint.txt','a') as file2:
#         for j in range(len(synonyms)):
#             file2.write(entry + " " + synonyms[j]+ '\n')
#     with open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/HSLex_AntonymConstraint.txt','a') as file2:
#         for j in range(len(antonyms)):
#             file2.write(entry + " " + antonyms[j]+ '\n')
#

### Step 3b: randomly pair words labelled as abusive and non-abusive to create another constraint that will be pushed apert during counter fitting
#i: Select only words labelled as non-abusive. That is drop rows with True label
# HSLex = pd.read_csv(r'/home/kosimadukwe/Downloads/Hate Speech Lexicon/Wiegand Lexicon/Lexicons/baseLexicon_copy.csv', header= None, delimiter=',') #, engine='python', encoding = 'latin')
# indexNames1 = HSLex[(HSLex[1].values=='True')].index
# HSLex.drop(indexNames1 , inplace=True)
# print(len(HSLex))
# HSLex.to_csv('HSLex_Constraint2.csv', index=False)    #sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label= [0,1,2],mode='w', encoding=None,  line_terminator=None)
#

###
# HSLex_true= pd.read_csv('HSLex_Constraint1.csv', header= None, delimiter=',')
# HSLex_false = pd.read_csv('HSLex_Constraint2.csv', header= None, delimiter=',')
# 
# with open(r'/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/linguistic_constraints/HSLex_RandomAbusiveNonAbusiveConstraint.txt','a') as file2:
# 
#     for i in range(len(HSLex_false)):
#         word1 = HSLex_false.iloc[i,2]
#         if i <= len(HSLex_true):
#             word2 = HSLex_true.iloc[i,2]
#         else:
#             i=i-551
#             word2 = HSLex_true.iloc[i,2]
#         file2.write(word1 + " " + word2+ '\n')





# #indexNames1 = HSLex[(HSLex[0,1].values=='True')].index
# HSLex.drop([0,1], inplace=True, axis=1)
# # HSLex.drop(indexNames1 , inplace=True)
# print(len(HSLex))
# HSLex.to_csv('HSLex_Constraint2a.csv', index=False)    #sep=',', na_rep='', float_format=None, columns=None, header=True, index=False, index_label= [0,1,2],mode='w', encoding=None,  line_terminator=None)
