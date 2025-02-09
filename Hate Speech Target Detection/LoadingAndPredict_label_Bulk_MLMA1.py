### Updated code. Also updated for four classes

## Loading a cluster ensemble model and using it to predict new instances
      ################################ new method with weights added and

#### line 25 and 185
#finalclustermodel_MixMod_MLMA_4CLASS
#finalclustermodel_MixMod_MLMA_4CLASS_sansbert         #finalclustermodel_MixMod_MLMA_4CLASS_sansbert_and randomkmeans



import openensembles as oe
import numpy as np
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import warnings
warnings.filterwarnings("ignore")
'''
LOADING A MODEL
'''
filename = "/vol/ecrg-solar/kosimadukwe/StanceDetection/Initial Results 11th October, 2021/With Increased Tf and TFIDF/ensemble models/finalclustermodel_MLMA_2022_SansBERT_MC.p" #/vol/ecrg-solar/kosimadukwe/StanceDetection/Ensemble_MLMA_4Class_SansBERT.p     #finalclustermodel_MixMod_MLMA_4CLASS
model = pickle.load(open(filename, 'rb')) #To load saved model from local directory

'''
load the data
'''
df1 = pd.read_csv("/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/MLMA_4class.csv", sep=",")
df1=df1.dropna(axis=0, how='any',)
df1['tweet_clean'] = df1['tweet_clean'].astype(str)
corpus = df1['tweet_clean'].values.tolist()

'''
calculate cluster purity and ARI of ensemble cluster model
'''
# zzz= model.labels['mixture_model']  #this is an ndarray, the same lenght as the number of instance. Tke values afre from 0 to K-1
class_col = df1['target'].values
d = {'tweet_clean':corpus, "class":class_col}

dfff= pd.DataFrame(d)
fs = model.labels['mixture_model'] -1
dfff['Clus_Label'] = fs #model.labels['mixture_model']
print(dfff['Clus_Label'].value_counts())

#save the new df to file for further analysis
cols= ['tweet_clean',"class", "Clus_Label"]
export_csv_file= dfff.to_csv('/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster_4class_Ensemble.csv', columns = cols)
K = 4
Goldlabel_ClusterLabel_DF = pd.read_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster_4class_Ensemble.csv")
CorrectlyClassified=0
target_label_list = []
for i in range(K):
    GL_CL_Filtered_DF =Goldlabel_ClusterLabel_DF.loc[Goldlabel_ClusterLabel_DF["Clus_Label"] == i]

    if len(GL_CL_Filtered_DF.index) != 0:
        print("Purity for Cluster " + str(i) + " : " + str(max(GL_CL_Filtered_DF["class"].value_counts())) + "/" + str(len(GL_CL_Filtered_DF["Clus_Label"])))
        maxx= max(GL_CL_Filtered_DF["class"].value_counts())
        CorrectlyClassified += maxx

        target_label_list.append(GL_CL_Filtered_DF["class"].value_counts().idxmax())
        print("The Most Frequent Target in this cluster is : ", GL_CL_Filtered_DF["class"].value_counts().idxmax())
    else:
        continue
ClusterPurity = CorrectlyClassified/(len(Goldlabel_ClusterLabel_DF["Clus_Label"]))  * 100
print("Cluster Purity: ", ClusterPurity)



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

# stop_words= ['a', 'i', 'u', 'you', 'the', 'they', 'them', '?', 'n\'t', 'he', 'she', 'us', 'we', 'to', 'are', 'it', 'is','do','and','your', 'ur', 'in', 'him', 'her', 'that', 'of', '\'s', 'Ã°ÂŸÂ¤Â¨', 'yeah', 'yes', 'nah', 'no', '"', "*"]
stop_words = ['told','country', 'countries','people','shit', 'ass','fucking', 'fuck', 'um', 'ut', 'ure', 'ue', 'uc', 'ude', 'udd', 'udec', 'ud', "'re", "'ll", "'m", "'ve", "$", " ", 'a', 'u', 'you', 'the', 'they', 'them', '?', 'n\'t', 'he', 'she', 'us', 'we', 'to', 'are', 'it', 'is','do','and','your', 'ur', 'in', 'him', 'her', 'that', 'of', '\'s', 'ð¤¨', 'yeah', 'yes', 'nah', 'no', '"', "*", 'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it' ,'its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now', 'ur', 'lol', 'lmao', 're', 'get', 'said','even', 'go', 'one', 'think','co', 'look', 'full', 'actually', 'sound', 'like', 'something', 'bro', 'stop', 'amp', 'got', 'see', 'tweet', 'would', 'dude', 'call', 'right','tell','didn', 'want','went' ,'real', 'dont' ,'time', 'damn','oh','really','omg','say','wrong','im','keep', 'know', 'good', 'must', 'guys','makes', 'still', 'wtf', 'talking', 'going', 'saying', 'better', 'life', 'play', 'girl', 'first', 'anything', 'way', 'sorry','well','twitter','called','always', 'back','yeah' ,'mouth', 'love', 'shut', 'stfu','little','take','show','nah', 'won' ,'lil', 'knew' ,'hey','hope', 'name' ,'thank','bet',
 'pic','ya','ll','yes','watch','cause', 'funny' ,'bruh','boy','follow','picture','profile','lmfao','game','honestly',
'come','sure', 'nothing', 'made' 'wow' 'bot','term','thinks','learn','clearly','need','proves','calling', "it's","you're","won't","don't","he'd","he's","how's","didn't","i've","she's","i'll","aren't","wouldn't","can't","that's","isn't","who's","ain't", 'youve', 'you\x98\x82', 'yr', 'yuo', 'yup', 'yur', 'zach', '\x81', '\x8d', '\x98', '\x98\x82\x98\x82', '\x98\x90\x98\x82\x98\x92', '\xad','\x98\x82']
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, stop_words=stop_words, tokenizer=LemmaTokenizer())
tfidf_vec.fit(corpus)  #df.text.values

#New ngram_range=(1, 1), max_df=1.0, min_df=1,
#Old ngram_range=(1, 1), max_df=0.95, min_df=0.01

#ngram_range=(1, 3), max_df=1.0, min_df=0.0
#ngram_range=(1, 1), max_df=0.95, min_df=0.01

'''
BOW feature extraction
'''
from sklearn.feature_extraction.text import CountVectorizer
bow_vec = CountVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, stop_words=stop_words,tokenizer=LemmaTokenizer())
bow_vec.fit(corpus)
#New ngram_range=(1, 1), max_df=1.0, min_df=1,
#Old ngram_range=(1, 1), max_df=0.95, min_df=0.01

'''
BERT extraction
'''
embedder = SentenceTransformer('paraphrase-mpnet-base-v2')

def get_cluster_row_index(model):
    ##### trying to get the cluster members
    # print("CE dict_key_names: ",model.labels.keys())   #CE dict_key_names:  dict_keys(['mixture_model'])
    solution_name = list(model.labels.keys())[0]
    # print("solution name: ", solution_name)       #solution name:  mixture_model
    row_index_of_instances = {}
    list_of_ClusterNumber = []
    for i in model.clusterNumbers[solution_name]:
        # labels{i} = fco.get_cluster_members(j, i)
        A_DF = model.get_cluster_members(solution_name, i)  # A_DF is a tuple of size 1 that contains a 1d array
        row_index_of_instances[i] = A_DF[0]  # to select the array inside A_Df as values for the dictionary

        list_of_ClusterNumber.append(i)
        # print("list of cluster number : ", list_of_ClusterNumber)
    # print(row_index_of_instances)   #a dictionary where the key is the cluster ID and the value is the row index of instance that belong to thatr cluster
    return row_index_of_instances


def search(indexes, textlist):
    listss= []
    for i in indexes:
        sent= textlist[i]
        listss.append(sent)
        # print("sent")
    return listss

row_index_of_instances = get_cluster_row_index(model)   #pass in the model here

All_TFIDF_Clust_Center= []
All_BERT_Clust_Center = []
All_BOW_Clust_Center = []
for each_cluster_in_centroid in row_index_of_instances.keys():   # getting the row indexes for each cluster and converting it to a list
    Row_index_List= list(row_index_of_instances[each_cluster_in_centroid])
    list_of_sent_in_a_cluster = search(Row_index_List, corpus)  # get the sentences belonging to to those indexes
    # print(list_of_sent_in_a_cluster)

     #### Cluster center according to TFIDF
    tfidf_features = tfidf_vec.transform(list_of_sent_in_a_cluster)  # convert all the sentences in the cluster to TFIDF representation
    tfidf_features_array = tfidf_features.toarray()
    ss =np.sum(tfidf_features_array, axis = 0)
    vv=tfidf_features_array.shape[0]
    xx=ss/vv
    TFIDF_Clust_Center = xx.reshape(1, -1)
    All_TFIDF_Clust_Center.append(TFIDF_Clust_Center)


    #### Cluster center according to BERT
    bert_sent_embeddings = embedder.encode(list_of_sent_in_a_cluster)  #this will give an n,768 array, where n = number of sentences
    ss1 = np.sum(bert_sent_embeddings, axis=0)
    vv1 = bert_sent_embeddings.shape[0]
    xx1 = ss1 / vv1
    BERT_Clust_Center = xx1.reshape(1, -1)
    All_BERT_Clust_Center.append(BERT_Clust_Center)

    ### Cluster center according to BoW
    bow_features = bow_vec.transform(
        list_of_sent_in_a_cluster)  # convert all the sentences in the cluster to TFIDF representation
    bow_features_array = bow_features.toarray()
    ss2 = np.sum(bow_features_array, axis=0)
    vv2 = bow_features_array.shape[0]
    xx2 = ss2 / vv2
    BOW_Clust_Center = xx2.reshape(1, -1)
    All_BOW_Clust_Center.append(BOW_Clust_Center)


All_TFIDF_Clust_Center =np.array(All_TFIDF_Clust_Center)
All_BERT_Clust_Center = np.array(All_BERT_Clust_Center)
All_BOW_Clust_Center = np.array(All_BOW_Clust_Center)

from sklearn.metrics.pairwise import euclidean_distances
def predict_by_kosi(cluster_cent, new_instance):
    dist_list=[]
    for each_cent in cluster_cent:
        each_cent = each_cent.reshape(1, -1)
        dist = euclidean_distances(each_cent, new_instance)
        # print(dist.shape)  #(1, 1)
        dist_list.append(dist)
    # print(dist_list)
    return dist_list

# '''
# # ####################################################   METHOD 1 #####################################################
# # def DerivedClusterNames_TopTFIDF(K, corpus, stop_words):
# #     filename = "/vol/ecrg-solar/kosimadukwe/StanceDetection/Ensemble_MLMA_4Class_SansBERTRandomKmeans.p"
# #     model = pickle.load(open(filename, 'rb'))  # To load saved model from local directory
# #     fs = model.labels['mixture_model']
# #     fss = fs - 1   ## tomake it zeero- indexed
# #     # K = 5
# #     clustered_sentences = [[] for ii in range(K)]
# #     for sentence_id, cluster_id in enumerate(fss):
# #         # cluster_id = cluster_id - 1  ## to make cluster id zero -indexed so it can access the clustered sentence which is zero indexed.
# #         clustered_sentences[cluster_id].append(corpus[sentence_id])
# #         ## Automating the inspection of clusters to assign correct labels before the calculation of metrics
# #
# #     ###
# #     Top TFIDF Gold Label words for each target extracted from the dataset. Done in InspectCluster_MLMA.py
# #     ###
# #
# #     Origin = ['shithole' 'spic' 'ching' 'chong' 'nigger' 'country' 'countries' 'negro' 'mongoloid' 'immigrants']
# #     Gender = ['cunt', 'twat', 'feminazi', 'faggot', 'retarded', 'bitch', 'stupid', 'man', 'retard', 'dyke']
# #     SexualOrientation = ['faggot' 'dyke' 'bitch' 'cunt' 'never' 'retarded' 'retard' 'hate' 'man' 'someone']
# #     Disability = ['retard' 'retarded' 'mongoloid' 'mongy' 'mongol' 'man' 'stupid' 'never''make' 'word']
# #
# #     # Origin = ['shithole', 'ching', 'chong', 'nigger', 'spic', 'country', 'countries', 'negro', 'immigrants', 'mongoloid']
# #     # Gender = ['cunt', 'twat', 'feminazi', 'faggot', 'retarded', 'bitch', 'stupid', 'man', 'retard', 'dyke']
# #     # SexualOrientation = ['faggot', 'dyke', 'ass', 'fucking', 'fuck', 'shit', 'bitch', 'never', 'cunt', 'people']
# #     #Religion = ['raghead', 'faggot', 'cunt', 'people', 'retard', 'leftist', 'shithole', 'country', 'ass', 'white']
# #     # Disability = ['retard', 'retarded', 'mongoloid', 'fucking', 'people', 'mongy', 'fuck', 'mongol', 'shit', 'ass']
# #     #Other = ['twat', 'cunt', 'retarded', 'leftist', 'retard', 'faggot', 'fucking', 'spic', 'shithole', 'country']
# #
# #     ###
# #     Extracted the top tfid words for each cluster
# #     ###
# #     ListOfIdentifiedTargets = []
# #     for i, cluster in enumerate(clustered_sentences):
# #         vectorizer=TfidfVectorizer(ngram_range=(1, 1), max_df=0.95, min_df=0.01, stop_words=stop_words, tokenizer=LemmaTokenizer())   #, tokenizer=LemmaTokenizer()
# #         # vectorizer = TfidfVectorizer(stop_words= stop_words)
# #         vectorizer.fit_transform(corpus)
# #         X = vectorizer.transform(cluster)
# #         importance = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
# #         tfidf_feature_names = np.array(vectorizer.get_feature_names())
# #         Top10Words_array = tfidf_feature_names[importance[:10]]
# #         Top10Words = Top10Words_array.tolist()
# #         # If this cluster contains 70% or greater of the words in any of the GoldLabel TFIDf, then it will be assigned the same label/target as that
# #         OriginCount = 0
# #         GenderCount = 0
# #         SexualOrientationCount = 0
# #         #ReligionCount = 0
# #         DisabilityCount = 0
# #         #OtherCount = 0
# #
# #         j = 0
# #         while j <= 0:
# #             for eachtopword in Top10Words:
# #                 if eachtopword in Origin:
# #                     OriginCount += 1
# #                 if eachtopword in Gender:
# #                     GenderCount += 1
# #                 if eachtopword in SexualOrientation:
# #                     SexualOrientationCount += 1
# #                 # if eachtopword in Religion:
# #                 #     ReligionCount += 1
# #                 if eachtopword in Disability:
# #                     DisabilityCount += 1
# #                 # if eachtopword in Other:
# #                 #     OtherCount += 1
# #                 j += 1
# #
# #         MaxThis = [OriginCount,  GenderCount, SexualOrientationCount,  DisabilityCount,]  # ReligionCount,  OtherCount
# #         max_value = max(MaxThis)       #instead of 70% or greater, i did maximum instead.
# #         max_index = MaxThis.index(max_value)
# #
# #         if max_index == 0:
# #             # cluster_label = 0
# #             cluster_label = 'Origin'
# #             ListOfIdentifiedTargets.append(cluster_label)
# #         elif max_index == 1:
# #             # cluster_label = 1
# #             cluster_label = 'Gender'
# #             ListOfIdentifiedTargets.append(cluster_label)
# #         elif max_index == 2:
# #             # cluster_label = 2
# #             cluster_label = 'SexualOrientation'
# #             ListOfIdentifiedTargets.append(cluster_label)
# #         # elif max_index == 3:
# #         #     # cluster_label = 3
# #         #     cluster_label = 'Religion'
# #         #     ListOfIdentifiedTargets.append(cluster_label)
# #         else:
# #             # cluster_label = 4
# #             cluster_label = 'Disability'
# #             ListOfIdentifiedTargets.append(cluster_label)
# #         # else:
# #         #     # cluster_label = 4
# #         #     cluster_label = 'Other'
# #         #     ListOfIdentifiedTargets.append(cluster_label)
# #
# #     return ListOfIdentifiedTargets
#
# # print(DerivedClusterNames_TopTFIDF(4,corpus,stop_words))
# # ListOfIdentifiedTargets = DerivedClusterNames_TopTFIDF(4,corpus,stop_words)
# '''
#
# def DerivedClusterNames_OtherMethods(A0,A1,A2,A3):  #,A4,A5
#     '''
#     Top TFIDF Gold Label words for each target extracted from the dataset. Done in InspectClusters.py
#     '''
#
#     Origin = ['shithole' ,'spic' ,'ching' ,'chong' ,'nigger' ,'country' ,'countries' ,'negro', 'mongoloid' ,'immigrants']
#     Gender = ['cunt', 'twat', 'feminazi', 'faggot', 'retarded', 'bitch', 'stupid', 'man', 'retard', 'dyke']
#     SexualOrientation = ['faggot' ,'dyke' ,'bitch', 'cunt' ,'never', 'retarded' ,'retard', 'hate' ,'man', 'someone']
#     Disability = ['retard' ,'retarded', 'mongoloid', 'mongy' ,'mongol' ,'man', 'stupid', 'never' ,'make', 'word']
#
#     ### 5 classes
#     # Origin = ['shithole', 'ching', 'chong', 'nigger', 'spic', 'country', 'countries', 'negro', 'immigrants','mongoloid']
#     # Gender = ['cunt', 'twat', 'feminazi', 'faggot', 'retarded', 'bitch', 'stupid', 'man', 'retard', 'dyke']
#     # SexualOrientation = ['faggot', 'dyke', 'ass', 'fucking', 'fuck', 'shit', 'bitch', 'never', 'cunt', 'people']
#     # Religion = ['raghead', 'faggot', 'cunt', 'people', 'retard', 'leftist', 'shithole', 'country', 'ass', 'white']
#     # Disability = ['retard', 'retarded', 'mongoloid', 'fucking', 'people', 'mongy', 'fuck', 'mongol', 'shit', 'ass']
#     # Other = ['twat', 'cunt', 'retarded', 'leftist', 'retard', 'faggot', 'fucking', 'spic', 'shithole', 'country']
#
#
#     from collections import OrderedDict
#     A0 = OrderedDict(A0)
#     A1 = OrderedDict(A1)
#     A2 = OrderedDict(A2)
#     A3 = OrderedDict(A3)
#     # A4 = OrderedDict(A4)
#     # A5 = OrderedDict(A5)
#
#     clust_list = [A0, A1, A2, A3] #, A4, A5
#
#     ListOfIdentifiedTargets = []
#
#     # for eachcluster in OutputOfInspectMethod:
#     for i in clust_list:
#         # If this cluster contains 70% or greater of the words in any of the GoldLabel TFIDf, then it will be assigned the same label/target as that
#         OriginWeightList = []
#         GenderWeightList = []
#         SexualOrientationWeightList = []
#         # ReligionWeightList = []
#         DisabilityWeightList = []
#         # OtherWeightList = []
#
#         j = 0
#         while j <= 0:
#             for key, value in i.items():
#                 if key in Origin:
#                     OriginWeightList.append(value)
#                 if key in Gender:
#                     GenderWeightList.append(value)
#                 if key in SexualOrientation:
#                     SexualOrientationWeightList.append(value)
#                 # if key in Religion:
#                 #     ReligionWeightList.append(value)
#                 if key in Disability:
#                     DisabilityWeightList.append(value)
#                 # if key in Other:
#                 #     OtherWeightList.append(value)
#                 j += 1
#
#         # take a sum of each list and then the max list is selected
#
#         OriginSum = np.sum(OriginWeightList)
#         GenderSum = np.sum(GenderWeightList)
#         SexualOrientationSum = np.sum(SexualOrientationWeightList)
#         # ReligionSum = np.sum(ReligionWeightList)
#         DisabilitySum = np.sum(DisabilityWeightList)
#         # OtherSum = np.sum(OtherWeightList)
#
#         MaxThis = [OriginSum, GenderSum, SexualOrientationSum, DisabilitySum] # ReligionSum, , OtherSum
#         max_value = max(MaxThis)  # instead of 70% or greater, i did maximum instead.
#         max_index = MaxThis.index(max_value)
#
#         if max_index == 0:
#             # cluster_label = 0
#             cluster_label = 'Origin'
#             ListOfIdentifiedTargets.append(cluster_label)
#         elif max_index == 1:
#             # cluster_label = 1
#             cluster_label = 'Gender'
#             ListOfIdentifiedTargets.append(cluster_label)
#         elif max_index == 2:
#             # cluster_label = 2
#             cluster_label = 'SexualOrientation'
#             ListOfIdentifiedTargets.append(cluster_label)
#         # elif max_index == 3:
#         #     # cluster_label = 3
#         #     cluster_label = 'Religion'
#         #     ListOfIdentifiedTargets.append(cluster_label)
#         else : #max_index == 4
#             # cluster_label = 4
#             cluster_label = 'Disability'
#             ListOfIdentifiedTargets.append(cluster_label)
#         # else:
#         #     # cluster_label = 4
#         #     cluster_label = 'Other'
#         #     ListOfIdentifiedTargets.append(cluster_label)
#
#     return ListOfIdentifiedTargets
#
# '''
# Output_of_InspectCluster_ReverseEngrMethod
# '''
#  ## the output of Inspectcluster_reverseEngrMethod is a bunch of ordered diction.copy and paste it here and delete the OrderedDict srrounding
#     ## it so it can become a list of tuples.
#
# # Re0 =[('cunt', 0.5073686766214673), ('twat', 0.39546610377303165), ('stupid', 0.022059002386962227), ('bitch', 0.015526781034429907), ('never', 0.014099802587467734), ('day', 0.013532716970621332), ('make', 0.011040717676579843), ('doe', 0.010379403669011873), ('year', 0.010242856163098944), ('mongy', 0.009863607391515145)]
# # Re1 =[('faggot', 0.2230465932175667), ('chong', 0.14990358304771456), ('ching', 0.14765402883781395), ('nigger', 0.1332842247416781), ('negro', 0.09228718422124862), ('white', 0.04147126673341025), ('chinaman', 0.03417788099594855), ('mongoloid', 0.03401558966380325), ('mongol', 0.02961304941791873), ('okay', 0.02702741528463033)]
# # Re2 =[('country', 0.2789995194196777), ('shithole', 0.24021067826046655), ('immigrant', 0.10515447821486738), ('illegal', 0.08512153235777456), ('refugee', 0.07987285488574536), ('alien', 0.06926316311866113), ('trump', 0.038854854022442264), ('america', 0.02873897859295563), ('world', 0.0270758102280127), ('mongol', 0.020553691975496984)]
# # Re3 =[('spic', 0.19178699926722975), ('faggot', 0.12970628651538313), ('dyke', 0.08558605664194924), ('retard', 0.07276697514591755), ('retarded', 0.07125848991022286), ('feminazi', 0.059165517408849656), ('mongoloid', 0.053068210220870986), ('raghead', 0.037837298966971855), ('mongy', 0.035375194612699425), ('nigger', 0.027293463823299922)]
#
# #########     Ensemble_MLMA_4Class.p  ######
# # Re0 =[('retard', 0.21943072300210117), ('retarded', 0.20500887640488585), ('country', 0.1228627245517631), ('shithole', 0.10581218874803257), ('immigrant', 0.04398811906940569), ('illegal', 0.03735957805380297), ('refugee', 0.03434508801851314), ('alien', 0.030550789357025782), ('feminazi', 0.024864860319684358), ('spic', 0.022894886437673408)]
# # Re1 =[('faggot', 0.4132224879729252), ('nigger', 0.1843547333410048), ('negro', 0.11797470995727077), ('white', 0.051875296841221843), ('okay', 0.03085748169732313), ('spic', 0.026853447465794898), ('dyke', 0.02475621055252233), ('bitch', 0.020026932694319415), ('nigga', 0.019924406568767056), ('never', 0.01576806969441803)]
# # Re2 =[('chong', 0.20496254102259306), ('ching', 0.2015242817186502), ('mongoloid', 0.16243780518064366), ('spic', 0.1401663823098646), ('mongol', 0.07207111699062557), ('chinaman', 0.05270291477682569), ('dyke', 0.0510707865587071), ('mongy', 0.045792613782785305), ('day', 0.014117809607594215), ('nigger', 0.011066174645894378)]
# # Re3 =[('cunt', 0.4783647561727038), ('twat', 0.37820339016666976), ('retard', 0.03090528717555188), ('stupid', 0.020093095528370474), ('retarded', 0.01776108225827743), ('mongy', 0.016395983593051968), ('hate', 0.01400589276824674), ('bitch', 0.013156749676453004), ('day', 0.012974353657974502), ('doe', 0.01267559253208812)]
#
# #########     Ensemble_MLMA_4Class_SansBERT.p  ######
# # Re0 =[('country', 0.10125757051968248), ('shithole', 0.0869669284864107), ('spic', 0.07939974875464427), ('chong', 0.07811902156388395), ('ching', 0.07652104325576889), ('nigger', 0.07629936692862392), ('mongoloid', 0.06823875767907747), ('negro', 0.047452437823913604), ('immigrant', 0.03845656837786247), ('dyke', 0.03350983967923601)]
# # Re1 =[('twat', 0.8232841136307014), ('cunt', 0.05899235436896428), ('day', 0.021126103829205267), ('doe', 0.019116394164469863), ('stupid', 0.018904804855129274), ('make', 0.0188558529433938), ('mongy', 0.018242548306556045), ('give', 0.01820395689349809), ('never', 0.015637042444362603), ('thing', 0.010982238789151692)]
# # Re2 =[('faggot', 0.5361091234263626), ('cunt', 0.3690903629466275), ('bitch', 0.02261988577017344), ('never', 0.015207335696504476), ('stupid', 0.013970164407487724), ('hate', 0.012892424979031493), ('guy', 0.011779540008644535), ('year', 0.01132054281521663), ('man', 0.009340503657904039), ('someone', 0.00926670518248825)]
# # Re3 =[('retard', 0.47372586923653653), ('retarded', 0.4207737101524018), ('guy', 0.018368272361729998), ('nigga', 0.017786554731809352), ('make', 0.015387720999169213), ('man', 0.01305562676735955), ('stupid', 0.012861742601500624), ('never', 0.011716443293987949), ('doe', 0.010672718652692727), ('bitch', 0.010175958565913988)]
#
#
# #########     Ensemble_MLMA_4Class_SansBERTRandomKmeans.p  ######
# # Re0 =[('country', 0.09987380808381321), ('shithole', 0.08546749868492086), ('spic', 0.07803078756921937), ('chong', 0.07699886057082793), ('nigger', 0.07561073304506957), ('ching', 0.07543002469414153), ('mongoloid', 0.06706222737426579), ('negro', 0.04663429234419096), ('immigrant', 0.037793524095485524), ('dyke', 0.033138823119890255)]
# # Re1 =[('retard', 0.474072600714969), ('retarded', 0.42628283571005526), ('nigga', 0.018212155243836695), ('guy', 0.01699611824453882), ('make', 0.013439180893154065), ('man', 0.013368024616229568), ('never', 0.011996796872308698), ('stupid', 0.011642163010987219), ('doe', 0.010928097763026352), ('bitch', 0.01041945109391193)]
# # Re2 =[('faggot', 0.9070983465923477), ('bitch', 0.019703592142923027), ('never', 0.019644410828158147), ('guy', 0.01729768384673437), ('man', 0.011303824601572131), ('year', 0.011121699433915597), ('nigga', 0.009985471979094598), ('someone', 0.009644912556651414), ('thing', 0.00956694662687762), ('mean', 0.008567584365121186)]
# # Re3 =[('cunt', 0.4995529493053211), ('twat', 0.3976880887877117), ('stupid', 0.022348834219617187), ('mongy', 0.017062938857854083), ('hate', 0.014575623999497455), ('bitch', 0.013691939493800244), ('never', 0.012763199513648361), ('day', 0.012370352008838457), ('give', 0.010958806455149394), ('doe', 0.010632662388635349)]
#
# ##### iNCREASED TF AND TFIDF SIZE
#
# ######### finalclustermodel_MLMA_2022.p ######
# # Re0 =[('retard', 0.11512048783272924), ('faggot', 0.1116144273590971), ('retarded', 0.10169034664057701), ('nigga', 0.010368684982839615), ('gay', 0.009816114596967558), ('bitch', 0.008401955402237147), ('guy', 0.008049762584438491), ('man', 0.006983544929032127), ('ok', 0.00694243730421462), ('stupid', 0.0065985630276817545)]
# # Re1 =[('cunt', 0.15753898832872287), ('twat', 0.14408271609222156), ('retard', 0.01863783978565816), ('horrible', 0.012987679761367856), ('absolute', 0.012422236259087878), ('stupid', 0.01047108819367451), ('he\\', 0.010050519882861386), ('dumb', 0.009512127504838706), ('sick', 0.008787778743460385), ('hate', 0.007966365964982621)]
# # Re2 =[('ching', 0.05309428864339237), ('chong', 0.05256373003012994), ('nigger', 0.048782641150776414), ('spic', 0.042606760119356664), ('mongoloid', 0.030821169598213193), ('negro', 0.026354629345765726), ('dyke', 0.020084233994507563), ('mongol', 0.015292369649601277), ('white', 0.014476598944180285), ('mongy', 0.012952290412524632)]
# # Re3 =[('country', 0.09646351376637453), ('shithole', 0.08419000347453061), ('illegal', 0.034195861194953305), ('immigrant', 0.032025169647729064), ('alien', 0.02823738668915466), ('refugee', 0.02152216460065829), ('trump', 0.016122018547659225), ('america', 0.012413787302266171), ('\\shithole\\', 0.011386990688340983), ('american', 0.010034468939221951)]
#
#
# # ######## finalclustermodel_MLMA_2022_SansBERT.p
# # Re0 =[('faggot', 0.15729562582441833), ('retarded', 0.14748656371733906), ('nigga', 0.013057567059622351), ('bitch', 0.009526660452805664), ('gay', 0.009394611001086055), ('guy', 0.008111915721972174), ('man', 0.007461913601391237), ('hate', 0.006152056900862301), ('dick', 0.005924652157136756), ('make', 0.005555828433572749)]
# # Re1 =[('cunt', 0.1737899498353938), ('twat', 0.16321324833643452), ('horrible', 0.014131672279208548), ('absolute', 0.012844285966100396), ('stupid', 0.011497707690774576), ('dumb', 0.011086365121912845), ('he\\', 0.009733806154416406), ('sick', 0.009561831793402491), ('hate', 0.007675112491697349), ('mongy', 0.006560075167782601)]
# # Re2 =[('country', 0.03662419688500121), ('ching', 0.035313077819983024), ('chong', 0.03495999252931564), ('nigger', 0.03290481049126259), ('shithole', 0.031892780655107596), ('spic', 0.029347918329038936), ('mongoloid', 0.021648619411026082), ('negro', 0.018437732098248845), ('dyke', 0.013462686364824943), ('illegal', 0.012896067189066551)]
# # Re3 =[('retard', 0.31934022539766255), ('ok', 0.012305713660415022), ('never', 0.009212073677008707), ('stupid', 0.008333101174390503), ('gay', 0.007773801242617467), ('guy', 0.007410297514088611), ('literally', 0.006862740043613798), ('jew', 0.006515006347712095), ('gone', 0.006265874478343022), ('account', 0.00599422879289843)]
#
#
# #######  finalclustermodel_MLMA_2022_SansBERT_MC.p
# Re0 =[('country', 0.03890417583017291), ('ching', 0.03760162018895649), ('chong', 0.03722565242251986), ('nigger', 0.034896917855080194), ('shithole', 0.03395966307656247), ('spic', 0.031249875300322017), ('mongoloid', 0.023051606230936716), ('negro', 0.019632630240793276), ('dyke', 0.014335165628829184), ('illegal', 0.01381220954735972)]
# Re1 =[('mongol', 0.19758535887806689), ('mongy', 0.1610845081732954), ('horde', 0.01994475533536038), ('what\\', 0.014745154021929755), ('american', 0.011704526431276775), ('fan', 0.009365922155398929), ('pencil', 0.00933244583348439), ('many', 0.009252544107756392), ('khan', 0.009162517180202534), ('he\\', 0.009064853829374523)]
# Re2 =[('cunt', 0.17477045256253523), ('twat', 0.16232513046879612), ('horrible', 0.014131672279208548), ('absolute', 0.012844285966100396), ('stupid', 0.011497707690774576), ('dumb', 0.011086365121912845), ('he\\', 0.009733806154416406), ('sick', 0.009561831793402491), ('hate', 0.007675112491697349), ('mongy', 0.006560075167782601)]
# Re3 =[('retard', 0.11486417686449259), ('faggot', 0.10045274180320059), ('retarded', 0.09542410061326792), ('nigga', 0.009534528471114951), ('gay', 0.008805346361240205), ('bitch', 0.00790639671732516), ('guy', 0.007854164013049064), ('ok', 0.007050856773422724), ('never', 0.006830820869859324), ('man', 0.006776334588812184)]
#
#
# # print(DerivedClusterNames_ReverseEngrCenter(Output_of_InspectCluster_ReverseEngrMethod))
# ListOfIdentifiedTargets1 = DerivedClusterNames_OtherMethods(Re0,Re1,Re2,Re3) #,Re4,Re5)
# print("M1",ListOfIdentifiedTargets1)
#
# '''
# Output_of_InspectCluster_IGAttributionMethod
# '''
# ## the output of Inspectcluster_ExplanationMethodIGAttribution is a bunch of ordered dictionaries.
# # copy and paste it here and delete the OrderedDict surrounding it so it can becomes
# # a list of tuples.
#
# # #########     Ensemble_MLMA_4Class.p  ######
# # Ig0 =[('faggot', 0.14617061435440784), ('retard', 0.06275501043381354), ('chong', 0.049471332807232885), ('ching', 0.041507789865122935), ('retarded', 0.040466054627999605), ('spic', 0.03697367005815277), ('twat', 0.03661693225230831), ('nigger', 0.03490703847030399), ('mongoloid', 0.03236746171291769), ('dyke', 0.02457665714633091)]
# # Ig1 =[('retarded', 0.09074117674296601), ('retard', 0.0875580697433203), ('ching', 0.05831779137881408), ('chong', 0.03937260614172795), ('spic', 0.036150881844806616), ('twat', 0.034748827319075554), ('negro', 0.03321700141888505), ('nigger', 0.03088767759071992), ('feminazi', 0.024221696976894688), ('word', 0.020267776920294016)]
# # Ig2 =[('cunt', 0.22974844572005979), ('retarded', 0.06973535728149982), ('retard', 0.05427958158919047), ('ching', 0.05173739492219939), ('twat', 0.044655659043476764), ('chong', 0.031221677561964613), ('spic', 0.027352189347847583), ('nigger', 0.027275758193357882), ('negro', 0.023232244493858748), ('mongoloid', 0.022706417056456492)]
# # Ig3 =[('shithole', 0.3962394814443085), ('retard', 0.09317988109140206), ('ching', 0.04443770415829746), ('retarded', 0.04413681985310958), ('chong', 0.041108398483809816), ('nigger', 0.03903171674149916), ('spic', 0.03628479551422588), ('man', 0.033357842719688106), ('twat', 0.030154774606083526), ('aliens', 0.027430327977540303)]
#
# # # #########     Ensemble_MLMA_4Class_SansBERT.p  ######
# # Ig0 =[('retarded', 0.08592563018905201), ('shithole', 0.060944809683806056), ('twat', 0.05611419800400344), ('faggot', 0.055259623533451346), ('cunt', 0.05260284837878988), ('spic', 0.05128625772021726), ('retard', 0.050250501949071234), ('ching', 0.0460579282950676), ('chong', 0.038143024220924984), ('nigger', 0.029851087285627513)]
# # Ig1 =[('ching', 0.07904851973555815), ('retard', 0.05706521094924928), ('chong', 0.056703573635471356), ('nigger', 0.04387154027191754), ('shithole', 0.04312944044558175), ('cunt', 0.04170851236234216), ('faggot', 0.030321058057358498), ('mongoloid', 0.024086990672289584), ('negro', 0.023175117434394314), ('immigrants', 0.021532374027519122)]
# # Ig2 =[('shithole', 0.08485418046480929), ('retard', 0.06477206245653244), ('ching', 0.05047396843777409), ('chong', 0.041850522898802166), ('cunt', 0.03582037877198725), ('nigger', 0.03185246930160349), ('faggot', 0.03003902684959928), ('refugees', 0.024367492599036495), ('mongoloid', 0.02053840912516854), ('illegal', 0.020153593828191763)]
# # Ig3 =[('faggot', 0.08127913227144384), ('retard', 0.06379809195286723), ('cunt', 0.05735665358945578), ('chong', 0.05232219107574797), ('shithole', 0.043342364484907234), ('nigger', 0.04205518665891104), ('ching', 0.039720982178056974), ('negro', 0.025766137851652417), ('mongoloid', 0.02281712657220534), ('aliens', 0.01835208965390632)]
#
#
# # #########     Ensemble_MLMA_4Class_SansBERTRandomKmeans.p  ######
# # Ig0 =[('faggot', 0.09058799831932719), ('cunt', 0.08741050412964504), ('twat', 0.06043976446005955), ('retarded', 0.05912211389067393), ('spic', 0.052992391149126676), ('ching', 0.047367578500369385), ('illegal', 0.035532139606336756), ('nigger', 0.033091620585702496), ('chong', 0.0318499970541636), ('dyke', 0.025913171832832583)]
# # Ig1 =[('retard', 0.35375434801584127), ('retarded', 0.04962987005333378), ('ching', 0.04624031646807231), ('chong', 0.03930100119779605), ('nigger', 0.03424305108197883), ('mongoloid', 0.030912781766251136), ('immigrants', 0.01775202841719269), ('white', 0.01663275481037094), ('negro', 0.015694464781892195), ('aliens', 0.012497362983741968)]
# # Ig2 =[('retarded', 0.10151872300703889), ('chong', 0.06340518322670398), ('ching', 0.04965631762435559), ('aliens', 0.024000845103536513), ('mongoloid', 0.02387226001031563), ('nigger', 0.022488126231934434), ('immigrants', 0.016355140186915886), ('white', 0.015942282109898773), ('never', 0.014698537053620259), ('negro', 0.014657518091798856)]
# # Ig3 =[('shithole', 0.42377193779286576), ('chong', 0.05087133216478738), ('retarded', 0.043344176997466274), ('ching', 0.03881277247892725), ('nigger', 0.037077352035613735), ('mongoloid', 0.030206114904459617), ('okay', 0.030105649550339914), ('aliens', 0.02594037942151648), ('white', 0.020247059358876555), ('never', 0.014105740076339145)]
#
# ##### iNCREASED TF AND TFIDF SIZE
#
# ######### finalclustermodel_MLMA_2022.p ######
# # Ig0 =[('retard', 0.10361644320654302), ('chong', 0.03057432082325666), ('retarded', 0.020324883822608122), ('ching', 0.018033719926297258), ('cunt', 0.017815692584428548), ('negro', 0.01557437951662941), ('mongoloid', 0.01245143736327645), ('twat', 0.012190120159016358), ('spic', 0.011703617926483311), ('white', 0.010587976989585801)]
# # Ig1 =[('shithole', 0.15611309379309515), ('chong', 0.03654736407944168), ('ching', 0.02100419520789441), ('refugees', 0.017040141066774452), ('retarded', 0.015614883397783887), ('cunt', 0.01515428853208216), ('twat', 0.014096682174458875), ('mongoloid', 0.012506739956651925), ('white', 0.010617865399790658), ('make', 0.010497433961327974)]
# # Ig2 =[('retarded', 0.026347590187817696), ('cunt', 0.01864968217667915), ('twat', 0.01442426923716069), ('ching', 0.012815724245589445), ('spic', 0.012651219770385152), ('bitch', 0.010950016697355226), ('nigger', 0.009751139552365487), ('chong', 0.009402931825985696), ('mongy', 0.00833942459741742), ('mongol', 0.008189978891535079)]
# # Ig3 =[('faggot', 0.10588845098583724), ('illegal', 0.030280167554053217), ('nigger', 0.025589958318412754), ('twat', 0.018522322012818296), ('immigrants', 0.018286386100065048), ('spic', 0.01605063183685063), ('man', 0.01588609727953291), ('trump', 0.011808468707012442), ('ever', 0.011686805154203108), ('cunt', 0.010047856736440404)]
#
# ####finalclustermodel_MLMA_2022_SansBERT.p
# # Ig0 =[('retard', 0.14811105160867236), ('retarded', 0.02423565732364774), ('negro', 0.021284425964357905), ('ching', 0.01691164259559619), ('chong', 0.01501912399484663), ('dyke', 0.014712478726101963), ('immigrants', 0.014534689687981201), ('white', 0.01413125998038405), ('shithole', 0.013992888215016038), ('mongoloid', 0.011199402622499348)]
# # Ig1 =[('ching', 0.02302039601236218), ('shithole', 0.01687948871967387), ('chong', 0.016308151357646555), ('retarded', 0.014717097158109363), ('leftist', 0.012223456633853883), ('doesn', 0.01152461333557101), ('ok', 0.010977058941046799), ('let', 0.01078149598576994), ('white', 0.010389807699770834), ('okay', 0.010287613077828764)]
# # Ig2 =[('cunt', 0.039764025377896174), ('faggot', 0.03870970948175392), ('twat', 0.028539058417563868), ('spic', 0.024912020267400078), ('retarded', 0.01908055439626403), ('ching', 0.016890300355892916), ('shithole', 0.016162849711609365), ('nigger', 0.015434710834181345), ('chong', 0.012071956796634963), ('illegal', 0.011646977566226367)]
# # Ig3 =[('chong', 0.014246754704523159), ('retarded', 0.014179145339532826), ('shithole', 0.013587234608605947), ('ching', 0.011414529336578431), ('nigger', 0.009303106964418611), ('mongoloid', 0.008433982311906417), ('care', 0.007203298845884222), ('refugees', 0.006379854066906171), ('bitch', 0.006353637473682581), ('let', 0.005850469442049239)]
#
#
# #######  finalclustermodel_MLMA_2022_SansBERT_MC.p
# Ig0 =[('retarded', 0.04206804955233131), ('cunt', 0.039870080190030396), ('ching', 0.035913936542847115), ('nigger', 0.024920570508034753), ('shithole', 0.021158638798205343), ('faggot', 0.020816634331999654), ('spic', 0.015857053250476157), ('negro', 0.013745587167067353), ('illegal', 0.012935333067433233), ('chong', 0.012566843451997083)]
# Ig1 =[('mongol', 0.13847362448551986), ('chong', 0.019174138921006976), ('wit', 0.018700354708810776), ('shithole', 0.015478186536873675), ('aliens', 0.014542855232468743), ('faggot', 0.011999338191922356), ('ve', 0.011477567363449176), ('kind', 0.00935731350985888), ('ate', 0.00928717790289117), ('fights', 0.009031117040024041)]
# Ig2 =[('shithole', 0.0185442780686292), ('twat', 0.015498576383290242), ('faggot', 0.014116493643683638), ('trying', 0.01280658994078178), ('chong', 0.008591244560797405), ('mongoloid', 0.008151679334043762), ('bitch', 0.0075248709631955115), ('second', 0.007219876171835483), ('wants', 0.0072005537250272), ('without', 0.006715765998890544)]
# Ig3 =[('retard', 0.09746785661875854), ('faggot', 0.020295102944989915), ('shithole', 0.01780886771349057), ('twat', 0.012534234154202724), ('chong', 0.007588853027181724), ('president', 0.007034139256222115), ('spic', 0.006909506784048975), ('mongoloid', 0.006871780386346919), ('never', 0.006064578259019716), ('refugees', 0.006016325654319872)]
#
#
# ListOfIdentifiedTargets2 = DerivedClusterNames_OtherMethods(Ig0,Ig1,Ig2,Ig3) #,Ig4,Ig5
# print("M2",ListOfIdentifiedTargets2)
#
#
# '''
# Output_of_TopicModelling_with_Top2Vec_Method
# '''
#
# # Ensemble_MLMA_4Class_All.p
# # TV0=[('retarded', 0.62348557), ('retard', 0.54185045), ('shithole', 0.4445773), ('trump', 0.37274554), ('spic', 0.35730398), ('illegal', 0.30954075), ('immigrants', 0.23722115), ('refugees', 0.22716221), ('aliens',0.22494093), ('feminazi', 0.2059044), ('country', 0.1677975), ('countries', 0.14891897)]
# # TV1=[('faggot', 0.8362974), ('nigger', 0.37348154), ('negro', 0.20220256), ('okay', 0.16856527), ('white', 0.11181371)]
# # TV2=[('ching', 0.66268796), ('chong', 0.62502193), ('spic', 0.5095651), ('chinaman', 0.49240923), ('mongoloid', 0.47283876), ('mongol', 0.46038377), ('dyke', 0.27768707)]
# # TV3=[('cunt', 0.66592497), ('twat', 0.5061062)]
#
#
# # Ensemble_MLMA_4Class_SansBERT
# # TV0=[('countries', 0.5555068), ('country', 0.52863574), ('refugees', 0.3625134), ('immigrants', 0.36123437), ('shithole', 0.30825102), ('illegal', 0.27567738), ('nigger', 0.2712993), ('aliens', 0.25034142), ('chinaman', 0.22559233), ('mongol', 0.20471093), ('spic', 0.19037943), ('trump', 0.18078683), ('mongoloid', 0.17646933), ('mongy', 0.14869365), ('negro', 0.14869344), ('feminazi', 0.14686713), ('ching', 0.12589608), ('dyke', 0.09918106), ('okay', 0.09496795), ('chong', 0.09311895), ('white', 0.078460485)]
# # TV1=[('twat', 0.5166405)]
# # TV2=[('faggot', 0.81366885), ('cunt', 0.6226381)]
# # TV3=[('retarded', 0.6696243), ('retard', 0.6095145)]
#
# # Ensemble_MLMA_4Class_SansBERTRandomKmeans.p
# # TV0= [('refugees', 0.53609157), ('immigrants', 0.53411704), ('countries', 0.44600374), ('country', 0.4068308), ('aliens', 0.3449617), ('illegal', 0.3192184), ('nigger', 0.2871472), ('faggot', 0.26940128), ('shithole', 0.23867176), ('chinaman', 0.19567545), ('mongol', 0.16482066), ('spic', 0.16222481), ('negro', 0.14278884), ('mongoloid', 0.14030086), ('trump', 0.13814618), ('feminazi', 0.13407704), ('mongy', 0.10831715), ('ching', 0.07857536), ('okay', 0.07255708), ('dyke', 0.0679311), ('white', 0.06321385), ('chong', 0.039245855)]
# # TV1=[('retarded', 0.62575936), ('retard', 0.5746751)]
# # TV2=[('faggot', 0.84818494)]
# # TV3=[('cunt', 0.6726511), ('twat', 0.5092827)]
#
#
# ##### iNCREASED TF AND TFIDF SIZE
#
# ######### finalclustermodel_MLMA_2022.p ######
# # TV0=[('retarded', 0.7052125), ('retard', 0.6264192), ('faggot', 0.46590889)]
# # TV1=[('cunt', 0.64871615), ('twat', 0.48913822), ('retard', 0.30153352)]
# # TV2=[('nigger', 0.5949724), ('ching', 0.54177), ('chong', 0.5070909), ('spic', 0.4936098), ('negro', 0.46220112), ('mongoloid', 0.45435432), ('mongol', 0.42392558), ('chinaman', 0.4189504), ('mongy', 0.4154557), ('white', 0.304671), ('dyke', 0.28682768), ('feminazi', 0.26506394), ('okay', 0.26069134)]
# # TV3=[('countries', 0.54423654), ('country', 0.52422655), ('refugees', 0.35850516), ('immigrants', 0.35705182), ('shithole', 0.31925857), ('illegal', 0.27861816), ('aliens', 0.25402862), ('trump', 0.19416443)]
#
# ####finalclustermodel_MLMA_2022_SansBERT.p
# # TV0=[('retarded', 0.65039515), ('faggot', 0.48599923)]
# # TV1=[('cunt', 0.67467725), ('twat', 0.51200765)]
# # TV2=[('refugees', 0.53645265), ('immigrants', 0.5340506), ('countries', 0.445616), ('country', 0.40625703), ('aliens', 0.34501916), ('illegal', 0.3190316), ('nigger', 0.2862001), ('shithole', 0.23831907), ('chinaman', 0.19566025), ('mongol', 0.16391332), ('spic', 0.16176715), ('negro', 0.14184685), ('mongoloid', 0.13922146), ('trump', 0.13753237), ('feminazi', 0.13342945), ('mongy', 0.10757261), ('ching', 0.07849912), ('okay', 0.072632484), ('dyke', 0.06771314), ('white', 0.062622584), ('chong', 0.039273366)]
# # TV3=[('retard', 0.7029683)]
#
# #######  finalclustermodel_MLMA_2022_SansBERT_MC.p
# TV0=[('refugees', 0.53210247), ('immigrants', 0.52760553), ('countries', 0.44943672), ('country', 0.40942824), ('aliens', 0.34330142), ('illegal', 0.31716642), ('nigger', 0.30066085), ('shithole', 0.23931229), ('chinaman', 0.19710147), ('spic', 0.1625313), ('negro', 0.16056332), ('mongoloid', 0.14843298), ('trump', 0.13965696), ('feminazi', 0.13440642), ('ching', 0.08081701), ('okay', 0.07378872), ('dyke', 0.06932825), ('white', 0.06548087), ('chong', 0.042475935)]
# TV1=[('mongol', 0.5896776), ('mongy', 0.28917402)]
# TV2=[('cunt', 0.67598987), ('twat', 0.51313186)]
# TV3=[('retarded', 0.6283648), ('retard', 0.57047427), ('faggot', 0.43240252)]
#
# ListOfIdentifiedTargets4 = DerivedClusterNames_OtherMethods(TV0,TV1,TV2,TV3)
# print("M4",ListOfIdentifiedTargets4)
#
# ##METHOD 3 contd:
# ListOfIdentifiedTargets3 = []
# for each_label in target_label_list:
#     if each_label == 0:  # cluster_label = 0
#         cluster_label = 'Origin'
#         ListOfIdentifiedTargets3.append(cluster_label)
#         # print('Class')
#     elif each_label == 1:  # cluster_label = 1
#         cluster_label = 'Gender'
#         ListOfIdentifiedTargets3.append(cluster_label)
#         # print('Disability')
#     elif each_label == 2:  # cluster_label = 2
#         cluster_label = 'SexualOrientation'
#         ListOfIdentifiedTargets3.append(cluster_label)
#     else:
#         # cluster_label = 3
#         cluster_label = 'Disability'
#         ListOfIdentifiedTargets3.append(cluster_label)
#         # print('SexualOrientation')
#
# print("M3",ListOfIdentifiedTargets3)
#
# # '''
# # FINAL TARGET LIST
# # '''
# # Final_Target_List= []
# # for i in range(4):
# #     intermediate_list = []
# #     intermediate_list.append(ListOfIdentifiedTargets1[i])
# #     intermediate_list.append(ListOfIdentifiedTargets2[i])
# #     intermediate_list.append(ListOfIdentifiedTargets3[i])
# #     final_label = max(intermediate_list)
# #     Final_Target_List.append(final_label)
# #
# # print("Final Target List",Final_Target_List)
# #


############PREDICTION AFTER THE FINAL TARGET LIST OF LABELS ARE CHOSEN #####

# Final_Target_List = ["Origin", "Disability", "SexOrien", "Gender"]
''' 
# Since i didnt take the max, instead i just used the best prforming label method 
# for prediction, the  Final_Target_List will now be made up of either
# ListOfIdentifiedTargets1, ListOfIdentifiedTargets2, ListOfIdentifiedTargets3 or ListOfIdentifiedTargets4
'''

Final_Target_List= ['Origin', 'Disability', 'Gender', 'SexOrien']
'''
 ### these are labels from the Majority Voting and Elimination technique done in my paper
'''



### Prep new instance
#HatEval_HateClass.csv
#Davidson_HateClass
# Kurrek_HateClass.csv
'''
load new data to be predicted
'''
df1 = pd.read_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/HatEval_HateClass.csv", sep=",")   #/vol/ecrg-solar/kosimadukwe/StanceDetection/SampleDavidsonClass0.csv
df1=df1.dropna(axis=0, how='any',)
df1['tweet'] = df1['tweet'].astype(str)
newdata = df1['tweet'].values.tolist()

O_count = 0
D_count = 0
G_count = 0
S_count = 0

O_target =[]
D_target =[]
G_target =[]
S_target =[]

for newsent in newdata:
    newsent = [newsent]
    TFIDFfeatures = tfidf_vec.transform(newsent)
    newsentTFIDF =TFIDFfeatures.toarray()
    newsentBERT = embedder.encode(newsent)
    BOWfeatures = bow_vec.transform(newsent)
    newsentBoW = BOWfeatures.toarray()

    All_Target_List = []

    pred_prob1 = predict_by_kosi(All_TFIDF_Clust_Center,newsentTFIDF)    ## this will contain the distance for all the cluster centers from the new instance
    list_of_list_of_cluster_distance1= [arr.tolist() for arr in pred_prob1]  # the next 5 linesd is just to conver that to a list
    list_of_cluster_distance1=[]    #this will contain how far the new sentence is fom the existing cluster
    for each_clust_dist1 in list_of_list_of_cluster_distance1:
        each_clust_dist1 = each_clust_dist1[0]
        each_clust_dist1= each_clust_dist1[0]
        list_of_cluster_distance1.append(each_clust_dist1)
    minimum_distance1 = np.min(list_of_cluster_distance1)   # get the smallest distance because that is the one closest to the cluster
    Predicted_cluster1 = list_of_cluster_distance1.index(minimum_distance1)  #the index of that value will indicate what cluster is being referred to assuming order is maintained
    # print("The new instance belongs to cluster according to TFDIF: ",Predicted_cluster1)
    All_Target_List.append(Predicted_cluster1)

    # pred_prob2 = predict_by_kosi(All_BERT_Clust_Center,newsentBERT)    ## this will contain the distance for all the cluster centers from the new instance
    # list_of_list_of_cluster_distance2 = [arr.tolist() for arr in pred_prob2]  # the next 5 linesd is just to conver that to a list
    # list_of_cluster_distance2 =[]
    # for each_clust_dist2 in list_of_list_of_cluster_distance2:
    #     each_clust_dist2 = each_clust_dist2[0]
    #     each_clust_dist2 = each_clust_dist2[0]
    #     list_of_cluster_distance2.append(each_clust_dist2)
    # minimum_distance2 = np.min(list_of_cluster_distance2)   # get the smalled distance
    # Predicted_cluster2 = list_of_cluster_distance2.index(minimum_distance2)
    # print("The new instance belongs to cluster according to BERT:  ", Predicted_cluster2)
    # All_Target_List.append(Predicted_cluster2)

    pred_prob3 = predict_by_kosi(All_BOW_Clust_Center,newsentBoW)    ## this will contain the distance for all the cluster centers from the new instance
    list_of_list_of_cluster_distance3 = [arr.tolist() for arr in pred_prob3]  # the next 5 linesd is just to conver that to a list
    list_of_cluster_distance3 = []
    for each_clust_dist3 in list_of_list_of_cluster_distance3:
        each_clust_dist3 = each_clust_dist3[0]
        each_clust_dist3 = each_clust_dist3[0]
        list_of_cluster_distance3.append(each_clust_dist3)
    minimum_distance3 = np.min(list_of_cluster_distance3)   # get the smalled distance
    Predicted_cluster3= list_of_cluster_distance3.index(minimum_distance3)
    # print("The new instance belongs to cluster according to BOW:  ", Predicted_cluster3)
    All_Target_List.append(Predicted_cluster3)

    if All_Target_List[0] == All_Target_List[1]: # used to remove instances where the different representation methods did not have the same  prediction
        Predicted_Target_Label = All_Target_List[0]   #max(All_Target_List)

        if Predicted_Target_Label == 0:
            cluster_label = Final_Target_List[0]
            O_count += 1
            O_target.append(newsent)
        elif Predicted_Target_Label == 1:
            cluster_label = Final_Target_List[1]
            D_count += 1
            D_target.append(newsent)
        elif Predicted_Target_Label == 2:
            cluster_label = Final_Target_List[2]
            S_count += 1
            S_target.append(newsent)
        # elif Predicted_Target_Label == 3:
        #     cluster_label = ListOfIdentifiedTargets[3]
        else:
            cluster_label = Final_Target_List[3]
            G_count += 1
            G_target.append(newsent)

        # print("Predicted Cluster Label : ", cluster_label)
    else:
        continue

print("O_count", O_count)
print("D_count", D_count)
print("S_count", S_count)
print("G_count", G_count)

total = O_count + D_count + S_count + G_count
print("O_pt",(O_count/total) * 100)
print("D_pt",(D_count/total) * 100)
print("S_pt",(S_count/total) * 100)
print("G_pt",(G_count/total) * 100)


print("O_target : ", O_target)
print("D_target : ", D_target)
print("S_target : ", S_target)
print("G_target : ", G_target)