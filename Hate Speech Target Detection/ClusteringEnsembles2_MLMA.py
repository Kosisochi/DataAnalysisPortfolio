import datetime
begin_time = datetime.datetime.now()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import openensembles as oe
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import sys
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from sentence_transformers import SentenceTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


'''
SECTION 1 :   LOADING DATA AND FEATURE EXTRACTION SECTION
'''
# MLMA_4class
'''
load the data
'''
# df1 = pd.read_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/HateLingo for the 5 targets/HateLingo5Targets_clean.csv", sep=",")
# df1=df1.dropna(axis=0, how='any',)
# df1['tweet_clean'] = df1['tweet_clean'].astype(str)
# corpus = df1['tweet_clean'].values.tolist()
#
# class_col = df1['class'].values
# d = {'tweet_clean':corpus, "class":class_col}
# dfff= pd.DataFrame(d)


df1 = pd.read_csv("/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/MLMA_4class.csv", sep=",")
df1=df1.dropna(axis=0, how='any',)
df1['tweet_clean'] = df1['tweet_clean'].astype(str)
corpus = df1['tweet_clean'].values.tolist()

class_col = df1['target'].values
d = {'tweet_clean':corpus, "class":class_col}
dfff= pd.DataFrame(d)
#
'''
BERT extraction
# '''
embedder = SentenceTransformer('paraphrase-mpnet-base-v2')  #paraphrase-MiniLM-L6-v2
corpus_embeddings = embedder.encode(corpus)
df2 = pd.DataFrame(corpus_embeddings) #put the dataset into a dataframe
list_x = [i for i in range(768)]
dataObj = oe.data(df2, list_x) #instantiate the oe data object


'''
TFIDF Extraction
'''
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]

# stop_words= ['a', 'i', 'u', 'you', 'the', 'they', 'them', '?', 'n\'t', 'he', 'she', 'us', 'we', 'to', 'are', 'it', 'is','do','and','your', 'ur', 'in', 'him', 'her', 'that', 'of', '\'s', 'ð¤¨', 'yeah', 'yes', 'nah', 'no', '"', "*"]
# tfidf_vec = TfidfVectorizer(ngram_range=(1, 3), max_df=1.0, min_df=0.0,stop_words=stop_words, tokenizer=LemmaTokenizer())

stop_words = ['people','shit', 'ass','fucking', 'fuck', 'um','ut', 'ure', 'ue', 'uc', 'ude', 'udd', 'udec', 'ud',"'re", "'ll", "'m", "'ve", "$", " ", 'a', 'u', 'you', 'the', 'they', 'them', '?', 'n\'t', 'he', 'she', 'us', 'we', 'to', 'are', 'it', 'is','do','and','your', 'ur', 'in', 'him', 'her', 'that', 'of', '\'s', 'ð¤¨', 'yeah', 'yes', 'nah', 'no', '"', "*", 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it' ,'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now', 'ur', 'lol', 'lmao', 're', 'get', 'said','even', 'go', 'one', 'think','co', 'look', 'full', 'actually', 'sound', 'like', 'something', 'bro', 'stop', 'amp', 'got', 'see', 'tweet', 'would', 'dude', 'call', 'right','tell','didn', 'want','went' ,'real', 'dont' ,'time', 'damn','oh','really','omg','say','wrong','im','keep', 'know', 'good', 'must', 'guys','makes', 'still', 'wtf', 'talking', 'going', 'saying', 'better', 'life', 'play', 'girl', 'first', 'anything', 'way', 'sorry','well','twitter','called','always', 'back','yeah' ,'mouth', 'love', 'shut', 'stfu','little','take','show','nah', 'won' ,'lil', 'knew' ,'hey','hope', 'name' ,'thank','bet',
 'pic','ya','ll','yes','watch','cause', 'funny' ,'bruh','boy','follow','picture','profile','lmfao','game','honestly',
'come','sure', 'nothing', 'made' 'wow' 'bot','term','thinks','learn','clearly','need','proves','calling', "it's","you're","won't","don't","he'd","he's","how's","didn't","i've","she's","i'll","aren't","wouldn't","can't","that's","isn't","who's","ain't", 'youve', 'you\x98\x82', 'yr', 'yuo', 'yup', 'yur', 'zach', '\x81', '\x8d', '\x98', '\x98\x82\x98\x82', '\x98\x90\x98\x82\x98\x92', '\xad','\x98\x82']
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, stop_words=stop_words, tokenizer=LemmaTokenizer())

tfidf_vec.fit(corpus)  #df.text.values
tfidf_features = tfidf_vec.transform(corpus)  #df.text.values
df3 = pd.DataFrame(tfidf_features.todense()) #put the dataset into a dataframe
list_xj = [j for j in range(len(df3.columns))]
# dataObj = oe.data(df3, list_xj) #instantiate the oe data object     #Uncomment this when runinng withour BERT. Thus you have to instantiate here instead of at BERT


'''
BOW feature extraction
'''
from sklearn.feature_extraction.text import CountVectorizer
bow_vec = CountVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1,stop_words=stop_words,tokenizer=LemmaTokenizer())
bow_vec.fit(corpus)
bow_features= bow_vec.transform(corpus)
df4 = pd.DataFrame(bow_features.todense()) #put the dataset into a dataframe
list_xk = [k for k in range(len(df4.columns))]


# print(dataObj.D.keys())
# dataObj.D["parent1"] = dataObj1
# print(dataObj.D.keys())
#
# plt_data = dataObj.plot_data('parent')
# plt_data1 = dataObj.plot_data('parent1')

MWEs = ["shithole country",
        "fucking twat cunt",
        "fucking faggot dyke",
        "fucking retard retarded"]  # ethnicity, sexual orientation, gender, disability, class
# MWEs_embeddings = embedder.encode(MWEs)

#### Intialize with TFIDF
mwe_tfidf_features = tfidf_vec.transform(MWEs)
# print('shape',mwe_features)
mwe_tfidf_vectors=mwe_tfidf_features.toarray()

#### Intialize with BoW
mwe_bow_features = bow_vec.transform(MWEs)
# print('shape',mwe_features)
mwe_bow_vectors=mwe_bow_features.toarray()


#Words at the center of each target as determined by the tfidf mean. pre processing done in InitializeWithClassCenterWords.py
CenterWordsOfEachTarget = ['shithole spic ching chong nigger negro mongoloid immigrants illegal refugees',
                           'cunt twat feminazi faggot retarded bitch stupid man retard dyke',
                           'faggot dyke bitch cunt never retarded retard hate, man someone',
                       'retard retarded mongoloid mongy mongol man stupid never make word']


# cwet_embeddings = embedder.encode(CenterWordsOfEachTarget)

#### Intialize with TFIDF
cwet_tfidf_features = tfidf_vec.transform(CenterWordsOfEachTarget)
cwet_tfidf_vectors = cwet_tfidf_features.toarray()

#### Intialize with BoW
cwet_bow_features = bow_vec.transform(CenterWordsOfEachTarget)
cwet_bow_vectors = cwet_bow_features.toarray()



'''
SECTION  2 :  THE CLUSTER ENSEMBLE SECTION
'''

numRepeats = 1
d_arr=[]
for i in range(numRepeats):
    d_temp = oe.data(df3, list_xj)
    d_arr.append(d_temp)
    d_temp1 = oe.data(df4, list_xk)
    d_arr.append(d_temp1)

transdict= dataObj.merge(d_arr)     ###what does this doe. It isnt used anywhere else!!!1 10th MARCH 2022  Is it just a variariable for variable sake??
print(dataObj.D.keys())
init_b = ['random' , 'k-means++']#, MWEs_embeddings, cwet_embeddings]   #, MWEs_embeddings
init_t= ['random', 'k-means++']#,mwe_tfidf_vectors, cwet_tfidf_vectors]   #'random', 'k-means++',
init_bo = ['random', 'k-means++']#,mwe_bow_vectors, cwet_bow_vectors]  #'random',

K = 4
c_MV_arr=[]
dict_key_name=[]
c = oe.cluster(dataObj)
for name in dataObj.D.keys():
    if name == "parent":
        inits=  init_t  #init_b
    elif name == "parent_1":
        inits= init_t
    else:
        inits = init_bo
    for ind, i in enumerate(inits):
        c.cluster(name, "kmeans", "kmeans_"+ name + "_" +str(ind), K, init = i , n_init = 1, random_state=21)   #(source_name, algorithm, output_name
        c_MV_arr.append(c.finish_majority_vote(threshold=0.5))
        dict_key_name.append("kmeans_"+ name + "_" +str(ind))

print("dict_key_name: ",dict_key_name)
print("dict_key_name: ",c.labels.keys())

# MI = c.MI(MI_type="normalized")
# mi_plot = MI.plot()
#
#
'''
SECTION  3: CALCULATE PERFORMANCE OF THE BASE CLSUTERS
'''
for x in range(len(dict_key_name)):
    zzz= c.labels[dict_key_name[x]]  #this is an ndarray, the same lenght as the number of instance. Tke values afre from 0 to K-1
    dfff['Clus_Label'] = c.labels[dict_key_name[x]]
    print(dfff['Clus_Label'].value_counts())

    #save the new df to file for further analysis
    cols= ['tweet_clean',"class", "Clus_Label"]
    export_csv_file= dfff.to_csv('/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster.csv', columns = cols)

    Goldlabel_ClusterLabel_DF = pd.read_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster.csv")
    CorrectlyClassified=0
    for i in range(K):
        GL_CL_Filtered_DF =Goldlabel_ClusterLabel_DF.loc[Goldlabel_ClusterLabel_DF["Clus_Label"] == i]
        print("Purity for Cluster " + str(i) + " : " + str(max(GL_CL_Filtered_DF["class"].value_counts())) + "/" + str(len(GL_CL_Filtered_DF["Clus_Label"])))

        maxx= max(GL_CL_Filtered_DF["class"].value_counts())
        CorrectlyClassified += maxx
    ClusterPurity = CorrectlyClassified/(len(Goldlabel_ClusterLabel_DF["Clus_Label"]))  * 100
    print("Cluster Purity: ", ClusterPurity)

    ARI = adjusted_rand_score(class_col, dfff['Clus_Label'])
    print('ARI_base: %f' % ARI)


    clustered_sentences = [[] for i in range(K)]
    for sentence_id, cluster_id in enumerate(c.labels[dict_key_name[x]]):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    ## Automating the inspection of clusters to assign correct labels before the calculation of metrics

    '''
    Top TFIDF Gold Label words for each target extracted from the dataset. Done in InspectClusters.py
    '''

    Origin = ['shithole', 'spic', 'ching', 'chong', 'nigger', 'country', 'countries', 'negro', 'mongoloid',
              'immigrants']
    Gender = ['cunt', 'twat', 'feminazi', 'faggot', 'retarded', 'bitch', 'stupid', 'man', 'retard', 'dyke']
    SexualOrientation = ['faggot', 'dyke', 'bitch', 'cunt', 'never', 'retarded', 'retard', 'hate', 'man', 'someone']
    Disability = ['retard', 'retarded', 'mongoloid', 'mongy', 'mongol', 'man', 'stupid', 'never', 'make', 'word']

    '''
    Extracted the top tfid words for each cluster and use it to label to clusters
    '''
    cluster_labels_list = []

    for i, cluster in enumerate(clustered_sentences):
        vectorizer = TfidfVectorizer(stop_words='english')
        X1 = vectorizer.fit_transform(corpus)
        X = vectorizer.transform(cluster)
        importance = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
        tfidf_feature_names = np.array(vectorizer.get_feature_names())
        Top10Words_array = tfidf_feature_names[importance[:10]]
        Top10Words= Top10Words_array.tolist()
        # If this cluster contains 70% or greater of the words in any of the GoldLabel TFIDf, then it will be assigned the same label/target as that
        OriginCount = 0
        GenderCount = 0
        SexualOrientationCount = 0
        DisabilityCount = 0

        j = 0
        while j <= 0:
            for eachtopword in Top10Words:
                if eachtopword in Origin:
                    OriginCount += 1
                if eachtopword in Gender:
                    GenderCount += 1
                if eachtopword in SexualOrientation:
                    SexualOrientationCount += 1
                if eachtopword in Disability:
                    DisabilityCount += 1
                j += 1

        # for eachtopword in Top10Words:
        #     if eachtopword in Origin:
        #         OriginCount += 1
        #     elif eachtopword in Gender:
        #         GenderCount += 1
        #     elif eachtopword in SexualOrientation:
        #         SexualOrientationCount += 1
        #     elif eachtopword in Religion:
        #         ReligionCount += 1
        #     elif eachtopword in Disability:
        #         DisabilityCount += 1
        #     else:
        #         OtherCount += 1

        MaxThis = [OriginCount, GenderCount, SexualOrientationCount,  DisabilityCount]
        max_value = max(MaxThis)       #instead of 70% or greater, i did maximum instead.
        max_index = MaxThis.index(max_value)


        if max_index == 0:
            # cluster_label = 0
            cluster_label = 'Origin'
            cluster_labels_list.append(cluster_label)
            # print('Origin')
        elif max_index == 1:
            # cluster_label = 1
            cluster_label = 'Gender'
            cluster_labels_list.append(cluster_label)
            # print('Gender')
        elif max_index == 2:
            # cluster_label = 2
            cluster_label = 'SexualOrientation'
            cluster_labels_list.append(cluster_label)
            # print('SexualOrientation')
        else:
            # cluster_label = 3
            cluster_label = 'Disability'
            cluster_labels_list.append(cluster_label)
            # print('Other')

        ## Assign the cluster_label to all the sentences in that cluster
        # df2['Clus_Label'] = cluster_label
        dfff['Clus_Label'] =dfff['Clus_Label'].replace(to_replace=i, value=cluster_label)

    print("cluster_labels_list",cluster_labels_list)

    # convert to the text target labels to integers
    dfff['Clus_Label1'] = dfff['Clus_Label'].replace(to_replace='Origin', value=0)
    dfff['Clus_Label1'] = dfff['Clus_Label1'].replace(to_replace='Gender', value=1)
    dfff['Clus_Label1'] = dfff['Clus_Label1'].replace(to_replace='SexualOrientation', value=2)
    dfff['Clus_Label1'] = dfff['Clus_Label1'].replace(to_replace='Disability', value=3)
    clus_target_label = dfff['Clus_Label1'].values


    #Meaure the Aprropriate Metrics
    #accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(class_col, clus_target_label)
    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)
    precision = precision_score(class_col,clus_target_label, average='macro')
    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)
    recall = recall_score(class_col, clus_target_label, average='macro')
    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(class_col, clus_target_label, average='macro')
    print('F1 score: %f' % f1)

    ARI = adjusted_rand_score(class_col, clus_target_label)
    print('ARI_base_postlabel: %f' % ARI)

'''
SECTION 4 : Calculate performance of final model
'''

                    ####################### MIXTURE MODEL ################################
fco=c.mixture_model(K,iterations=250)
fs = fco.labels['mixture_model']   #this is an ndarry with length the same as the number of instances.
#it contains labels from 1 to K(in this case 5)
## I need to make it zero-based so that i can calculate the scores with the original class labels
# print(fs[1:20])

ARI = adjusted_rand_score(class_col, fs)
print('ARI_ensemble: %f' % ARI)

fss = fs - 1
ARI = adjusted_rand_score(class_col, fss)
print('ARI_ensemble_zeroindexed: %f' % ARI)


'''
SECTION  6:  SAVING A MODEL
'''
import pickle
model1 = fco
filename = "/vol/ecrg-solar/kosimadukwe/StanceDetection/finalclustermodel_MLMA_2022_All_forSeededRandomKmeans21.p"
pickle.dump(model1, open(filename, 'wb')) #Saving the model


end_time = datetime.datetime.now()
print("Time taken to run: ", end_time- begin_time)