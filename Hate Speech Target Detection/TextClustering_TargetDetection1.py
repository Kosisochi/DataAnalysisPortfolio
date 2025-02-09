
# This is second secon of the Classifcation -Clustering experiment.
# The first part is in TextClassification_TargetDetection1.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
import numpy as np

# for reproducibility
random_state = 12

#import the output from running TextClassification_TargetDetection1.py
df= pd.read_csv("/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/HateLingo_Pred.csv")

#Get the sentences where the contents for columns "class" and "pred_class" are equal
tweet_list= []
true_class =[]
pred_class=[]
for i in range(len(df)):
    if df['class'][i] == df['pred_class'][i]:
        tweet_list.append(df['comment'][i])
        true_class.append(df['class'][i])
        pred_class.append(df['pred_class'][i])
d = {'comment':tweet_list,'class':true_class,"pred_class":pred_class}
df1=pd.DataFrame(d)


## Do clustering on those sentences.
#Load a pretrained model.
w2v_model =gensim.models.KeyedVectors.load_word2vec_format('/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/NewTextClassification/GoogleNews-vectors-negative300.txt', binary=False)


sent_vectors = [] # the avg-w2v for each sentence/review is stored in this train
for sent in tweet_list: # for each review/sentence
    sent_vec = np.zeros(300) # as word vectors are of zero length
    cnt_words =0 # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
sent_vectors = np.array(sent_vectors)
sent_vectors = np.nan_to_num(sent_vectors)


model2 = KMeans(n_clusters = 2)
model2.fit(sent_vectors)

# word_cluster_pred=model2.predict(sent_vectors)
word_cluster_pred_2=model2.labels_
word_cluster_center=model2.cluster_centers_

# Giving Labels/assigning a cluster to each point/text
df1['AVG-W2V Clus Label']=''
df1['AVG-W2V Clus Label'] = model2.labels_

#save the new pdf to file for further analysis
cols= ['comment','class', 'pred_class', "AVG-W2V Clus Label"]
export_csv_file= df1.to_csv('/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/HateLingo_Cluster.csv', columns = cols)

# How many points belong to each cluster ->
object_count= df1.groupby(['AVG-W2V Clus Label'])['comment'].count()
print("How many points belong to each cluster", object_count)

plt.bar([x for x in range(2)], df1.groupby(['AVG-W2V Clus Label'])['comment'].count(), alpha = 0.4)
plt.title('KMeans cluster points- AVG-W2V')
plt.xlabel("Cluster number")
plt.ylabel("Number of points")
plt.show()

#VISUALIZATION
# reduce the features to 2D
pca = PCA(n_components=2, random_state=random_state)
reduced_features = pca.fit_transform(sent_vectors)
# reduce the cluster centers to 2D
reduced_cluster_centers = pca.transform(model2.cluster_centers_)
plt.scatter(reduced_features[:,0], reduced_features[:,1], c=word_cluster_pred_2)  # c=Cluster_Labels c=cls.predict(features)
plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
plt.legend(loc="lower left", title="AvgW2VClasses")
plt.show()

# evaluation
from sklearn.metrics import silhouette_score
print(silhouette_score(sent_vectors, labels=word_cluster_pred_2))

##Creating Word Cloud for each cluster.
from wordcloud import WordCloud
labels= [0,1]
title= ['favour', 'against']
wiki_cl=pd.DataFrame(list(zip(title,labels)),columns=['title','cluster'])

result={'cluster':word_cluster_pred_2,'wiki':tweet_list}
result=pd.DataFrame(result)
for k in range(0,2):
   s=result[result.cluster==k]
   text=s['wiki'].str.cat(sep=' ')
   text=text.lower()
   text=' '.join([word for word in text.split()])
   wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
   print('Cluster: {}'.format(k))
   titles=wiki_cl[wiki_cl.cluster==k]['title']
   print('Titles :',titles.to_string(index=False))
   plt.figure()
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.axis("off")
   plt.show()