##25th July, 20222
##WRITING THIS CODE TO ADD SOMETHING TO CHAPTER 6 OF MY THESIS BASED ON SHARONS COMMENT

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
# finalclustermodel_MLMA_2022_SansBERT.p
# finalclustermodel_MLMA_2022_SansBERT_MC.p
# /vol/ecrg-solar/kosimadukwe/StanceDetection/Initial Results 11th October, 2021/Single Model/TF_MWE.p
filename = "/vol/ecrg-solar/kosimadukwe/StanceDetection/Initial Results 11th October, 2021/With Increased Tf and TFIDF/ensemble models/finalclustermodel_MLMA_2022.p" #/vol/ecrg-solar/kosimadukwe/StanceDetection/Initial Results 11th October, 2021/Ensemble_HL_5_SansBertrandomkmeans_IGAttrib/finalclustermodel_MixMod_HL5_sans_BERTrandomkmeans.p"
model = pickle.load(open(filename, 'rb')) #To load saved model from local directory



'''
load the data
'''
df1 = pd.read_csv("/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/MLMA_4class.csv", sep=",")
df1=df1.dropna(axis=0, how='any',)
df1['tweet_clean'] = df1['tweet_clean'].astype(str)
corpus = df1['tweet_clean'].values.tolist()
labels = df1['target'].values.tolist()


#GET THE DATA IN EACH CLUSTER
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

def searchlabels(indexes, labellist):
    listss= []
    for i in indexes:
        sent= labellist[i]
        listss.append(sent)
        # print("sent")
    return listss

row_index_of_instances = get_cluster_row_index(model)   #pass in the model here
ListofListsForAllClusters =[]

for each_cluster_in_centroid in row_index_of_instances.keys():   # getting the row indexes for each cluster and converting it to a list
    Row_index_List= list(row_index_of_instances[each_cluster_in_centroid])
    list_of_sent_in_a_cluster = search(Row_index_List, corpus)  # get the sentences belonging to those indexes
    list_of_labels_in_a_cluster = searchlabels(Row_index_List, labels)  # get the labels belonging to those indexes
    print("No. of sents in a cluster: ",len(list_of_sent_in_a_cluster))
    print("No. of labels in a cluster: ",len(list_of_labels_in_a_cluster))
    print("No. of sents in a cluster should be EQUAL TO No. of labels in a cluster")
    # print(list_of_sent_in_a_cluster)
    # ListForACluster = []   #this list contains the sentence in each cluster
    # ListForACluster.append(list_of_sent_in_a_cluster)
    # ListofListsForAllClusters.append(list_of_sent_in_a_cluster)

    ListForACluster = []   #this list contains the sentence in each cluster
    ListForACluster.append(list_of_labels_in_a_cluster)
    ListofListsForAllClusters.append(list_of_labels_in_a_cluster)

#Denominators of each cluster
DenoCluster0 = len(ListofListsForAllClusters[0])
DenoCluster1 = len(ListofListsForAllClusters[1])
DenoCluster2 = len(ListofListsForAllClusters[2])
DenoCluster3 = len(ListofListsForAllClusters[3])
# DenoCluster4 = len(ListofListsForAllClusters[4])


## Loop through one of these lists
## Take each sentence and find its match in the data
## Get the label of the match from the data
## Load into a label list

#get the unique count of each label in the list.
ListofDicts= []
from collections import Counter
for i , each_clusters_label_list in enumerate(ListofListsForAllClusters):
    _count = Counter(each_clusters_label_list) #_count is a dictionary
    print("Labels in Cluster " + str(i) + "and their counts: ", _count)
    ListofDicts.append(_count)


#Map the labels intergers to Target names by checking the data and the data section of your thesis
"""
For MLMA Dataset, 
Class 0 is Origin
Class 1 is Gender
Class 2 is Sex Orientation
Class 3 is Disability

"""
print("Class 0 = Origin, Class 1 = Gender, Class 2 = Sex Orientation, Class 3 = Disability")

#Get the label attached to that cluster from your thesis
'''

TF_MWE_.p
Method 1 "SexOrien", "Origin", "SexOrien", "SexOrien"
Method 2 "Gender", "Origin" ,"SexOrien" ,"Origin"
Method 3 "Origin" ,"Gender" ,"SexOrien", "Disability"
Method 4 "Origin", "Gender", "Gender", "Gender"

EBT T −All
Method 1 "Gender" ,"SexOrien" ,"Origin" ,"Gender"
Method 2 "Gender" ,"Gender","Gender", "Origin"
Method 3 "Origin" ,"Origin" ,"Origin" ,"Origin"
Method 4 "Gender", "Gender" ,"Origin", "Gender"

ET T −All
Method 1 "Origin" ,"Gender" ,"SexOrien" ,"Disability"
Method 2 "Gender", "Origin" ,"Origin" ,"Origin"
Method 3 "Origin" ,"Gender" ,"SexOrien" ,"Disability"
Method 4 "Origin" ,"Gender" ,"Gender" ,"Gender"

ET T −M C
Method 1 "Origin", "Disability", "SexOrien" ,"Gender"
Method 2 "Gender","Disability" ,"Origin", "Origin"
Method 3 "Origin" ,"Disability" ,"SexOrien", "Gender"
Method 4 "Origin", "Gender", "Gender" ,"Gender"

'''
LabelsFromLabellingMethod = ["Gender" ,"SexOrien" ,"Origin" ,"Gender"]   ##Switch this for each labelling method
print("LabelsFromLabellingMethod", LabelsFromLabellingMethod)
# this piece of code above converts the string labels to their corresponding interger
for i, val in enumerate(LabelsFromLabellingMethod):
    if val == "Class":
        LabelsFromLabellingMethod[i] = 0
    elif val == "Disability":
        LabelsFromLabellingMethod[i] = 1
    elif val == "Ethnicity":
        LabelsFromLabellingMethod[i] = 2
    elif val == "Gender":
        LabelsFromLabellingMethod[i] = 3
    else:
        LabelsFromLabellingMethod[i] = 4
print("Interger Values of Labels from Labelling Method: ",LabelsFromLabellingMethod)

NumeratorForAllClusters= []
for j,each_interger_label in enumerate(LabelsFromLabellingMethod):
    ClusterUniqueLabelCount = ListofDicts[j]
    num = ClusterUniqueLabelCount.get(each_interger_label, 0) # 0: if a key is missing, i.e if that label is not present in the cluster, they return a zero count
    NumeratorForAllClusters.append(num)

#Calculate the fraction: Numerator is the number of instances in the cluster that have the same label as the label assigned by the labelling algorithm or the max unique count of labels. I think ideally they should be the same. It would be interesting to see if a cluster was not labelled with its most frequent label.
# Denominator is the total number of instances in that cluster (Len of the cluster list or summation of the uniquecounts. Ideally, they should be equal)
print("Correctness of Cluster 0: ", NumeratorForAllClusters[0]/DenoCluster0)
print("Correctness of Cluster 1: ", NumeratorForAllClusters[1]/DenoCluster1)
print("Correctness of Cluster 2: ", NumeratorForAllClusters[2]/DenoCluster2)
print("Correctness of Cluster 3: ", NumeratorForAllClusters[3]/DenoCluster3)
# print("Correctness of Cluster 4: ", NumeratorForAllClusters[4]/DenoCluster4)