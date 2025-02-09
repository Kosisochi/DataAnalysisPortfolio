# this is the code for calculating the words at the center of class label from the original datset.

# Get the TFIDF of all the sentence in  belonging to a label
#Calculate the mean
#convert the mean to words
#use to initialize clustering

##################################note: to run this code, you have to change the value in line 19 to get different targets.
################################# each run gives the top words in a target


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

#/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/en_dataset_with_stop_words_Clean.csv
#/vol/ecrg-solar/kosimadukwe/StanceDetection/HateLingo for the 5 targets/HateLingo5Targets_clean.csv
df2 = pd.read_csv("/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/MLMA_4class.csv", sep=",")
df2 = df2.dropna()
df2['tweet_clean'] = df2['tweet_clean'].astype(str)
corpus = df2['tweet_clean'].values.tolist()
df3 = df2.loc[df2['target'] == 3]               #target       #Select cluster by cluster by uisng label index 0 to 5
x_test = df3['tweet_clean'].values.tolist()

TFIDF_Clust_Cent_DF = pd.DataFrame()

stop_words = ['ha','wa',"!", "#", "'", "(", ")", "/", "-","don\\'t",'it\\', 'you\\','\\\\','i\\','na', 'don\\ut','\\ud\\ude', 'it\\us', 've','told','country', 'countries', 'people', 'shit', 'ass','fucking', 'fuck', 'um', 'ut', 'ure', 'ue', 'uc', 'ude', 'udd', 'udec', 'ud', "'re", "'ll", "'m", "'ve", "$", " ", 'a', 'u', 'you', 'the', 'they', 'them', '?', 'n\'t', 'he', 'she', 'us', 'we', 'to', 'are', 'it', 'is','do','and','your', 'ur', 'in', 'him', 'her', 'that', 'of', '\'s', 'ð¤¨', 'yeah', 'yes', 'nah', 'no', '"', "*", 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it' ,'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now', 'ur', 'lol', 'lmao', 're', 'get', 'said','even', 'go', 'one', 'think','co', 'look', 'full', 'actually', 'sound', 'like', 'something', 'bro', 'stop', 'amp', 'got', 'see', 'tweet', 'would', 'dude', 'call', 'right','tell','didn', 'want','went' ,'real', 'dont' ,'time', 'damn','oh','really','omg','say','wrong','im','keep', 'know', 'good', 'must', 'guys','makes', 'still', 'wtf', 'talking', 'going', 'saying', 'better', 'life', 'play', 'girl', 'first', 'anything', 'way', 'sorry','well','twitter','called','always', 'back','yeah' ,'mouth', 'love', 'shut', 'stfu','little','take','show','nah', 'won' ,'lil', 'knew' ,'hey','hope', 'name' ,'thank','bet','pic','ya','ll','yes','watch','cause', 'funny' ,'bruh','boy','follow','picture','profile','lmfao','game','honestly', 'come','sure', 'nothing', 'made' 'wow' 'bot','term','thinks','learn','clearly','need','proves','calling', "it's","you're","won't","don't","he'd","he's","how's","didn't","i've","she's","i'll","aren't","wouldn't","can't","that's","isn't","who's","ain't", 'youve', 'you\x98\x82', 'yr', 'yuo', 'yup', 'yur', 'zach', '\x81', '\x8d', '\x98', '\x98\x82\x98\x82', '\x98\x90\x98\x82\x98\x92', '\xad','\x98\x82']
vectorizer =TfidfVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=0.01,stop_words=stop_words)
vectorizer.fit(corpus)
tfidf_features = vectorizer.transform(x_test)  # convert all the sentences in the cluster to TFIDF representation


'''
calculate the center of each target when represented as TFIDF and select the important words 
'''
tfidf_features_array = tfidf_features.toarray()
ss =np.sum(tfidf_features_array, axis = 0)
vv=tfidf_features_array.shape[0]
xx=ss/vv
TFIDF_Clust_Center = xx.reshape(1, -1)
TFIDF_Clust_Cent_DF = TFIDF_Clust_Cent_DF.append(pd.DataFrame(TFIDF_Clust_Center))
TFIDF_Clust_Cent_DF.columns = vectorizer.get_feature_names()

                            ### method one using threshold to select important words
# #Find the threshold  which is the mean of all the values
# threshold = TFIDF_Clust_Cent_DF.mean(axis = 1)
# th=threshold.iloc[0]
#select the column names where the weight is greater than threshold
# tfidf_Important_Words = []
# for i , rows in TFIDF_Clust_Cent_DF.iterrows():
#     intermediate_list = []
#     for j , each_column in enumerate(rows):
#         if each_column >  th: #0.0005:   #th
#             important_word = TFIDF_Clust_Cent_DF.columns[j]
#             intermediate_list.append(important_word)
#     tfidf_Important_Words.append(intermediate_list)
#
# print(tfidf_Important_Words)

                        #method two using highest n = 10 scores   to select important words

TFIDF_topWordsAndScores_dict = TFIDF_Clust_Cent_DF.to_dict('list')  # convert df to dict
TFIDF_topWordsAndScores_sortedlist = sorted([(v, k) for k, v in TFIDF_topWordsAndScores_dict.items()], reverse=True)  #sort in descending order
tfidf_Important_WordsAndScores = TFIDF_topWordsAndScores_sortedlist[0:10]
print(tfidf_Important_WordsAndScores)
tfidf_Important_Words= []
for score_word in tfidf_Important_WordsAndScores:
    word = score_word[1]
    tfidf_Important_Words.append(word)

print(tfidf_Important_Words)

# # 0 = ['ching', 'chong', 'illegal', 'immigrants', 'mongoloid', 'negro', 'nigger', 'shithole', 'spic']
# # 1 = ['cunt', 'faggot', 'feminazi', 'twat']
# # 2 = ['dyke', 'faggot']
# # 3 = ['bitch', 'cunt', 'faggot', 'immigrants', 'leftist', 'raghead', 'refugees', 'retard', 'retarded', 'shithole', 'twat', 'white']
# # 4 = ['mongoloid', 'mongy', 'retard', 'retarded']
# # 5 = ['cunt', 'faggot', 'leftist', 'retard', 'retarded', 'shithole', 'spic', 'twat']



# ['redneck', 'trump', 'dumb', 'stupid', 'trash', 'white', 'racist', 'bitch', 'make', 'man']
# ['retard', 'retarded', 'stupid', 'never', 'dumb', 'make', 'fat', 'bitch', 'man', 'trump']
# ['trash', 'white', 'nigger', 'racist', 'trump', 'bitch', 'dumb', 'make', 'never', 'ugly']
# ['cunt', 'twat', 'dyke', 'stupid', 'fat', 'dumb', 'ugly', 'make', 'bitch', 'man']
# ['faggot', 'dyke', 'bitch', 'stupid', 'ugly', 'dumb', 'make', 'man', 'nigger', 'fat']