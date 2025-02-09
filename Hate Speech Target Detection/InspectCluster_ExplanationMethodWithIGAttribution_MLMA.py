#Here, i want to use IG Attribution as a way of finding the important words in the cluster.


#Step 1: Train a classification model using the sentencces in each cluster and their arbitrarily assigned cluster labels.
#The choice of the feature and the the classification algorithm used here is very important. IG Attribution is applied to deep learning models i think
# But i want to keep the features simple like TFIDF

#A question: The words deemed as importANT to the kmeans during clustering might not be same words deemed as important to the classifcation algorithm


#Step 2: Apply IG Attribution to the saved model.


#Step 3: Find the most important woors according to the explanation model for each cluster.

import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from alibi.explainers import IntegratedGradients
from tensorflow.keras.layers import Input, Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from sklearn.model_selection import train_test_split

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
print('TF version: ', tf.__version__)   #2.3.0
print('Eager execution enabled: ', tf.executing_eagerly()) # True

#Ensemble_MLMA_4Class_SansBERTRandomKmeans.p
#Ensemble_MLMA_4Class_All.p

'''
LOADING A MODEL
'''
filename = "/vol/ecrg-solar/kosimadukwe/StanceDetection/finalclustermodel_MLMA_2022_SansBERT_MC.p"
model = pickle.load(open(filename, 'rb')) #To load saved model from local directory
fs= model.labels['mixture_model']

'''
load the data   # If you dont have the cluster labels saved to file, you can run this section to do so
'''
df1 = pd.read_csv("/home/kosimadukwe/Downloads/MLMA_hate_speech-master/hate_speech_mlma/MLMA_4class.csv", sep=",")
df1 = df1.dropna(axis=0, how='any',)
df1['tweet_clean'] = df1['tweet_clean'].astype(str)
corpus = df1['tweet_clean'].values.tolist()

class_col = df1['target'].values
d = {'tweet_clean':corpus, "class":class_col}
dfinal= pd.DataFrame(d)
fss = fs -1 # to make it zero indexed
dfinal['Clus_Label'] = fss
print("Dfinal Clus Label value Counts: ",dfinal['Clus_Label'].value_counts())

#save the new df to file for further analysis
cols= ['tweet_clean',"class", "Clus_Label"]
export_csv_file= dfinal.to_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster_forIGAttrib_4class_new.csv", columns = cols)


'''
load the data   #if you have the cluster labels already saved to file, then use this section.
'''
K= 4
maxlen = 100
df2 = pd.read_csv("/vol/ecrg-solar/kosimadukwe/StanceDetection/MLMA_Cluster_forIGAttrib_4class_new.csv")
#Goldlabel_ClusterLabel_DF

df2 = df2.dropna()
df2['tweet_clean'] = df2['tweet_clean'].astype(str)
tweet_list = df2['tweet_clean'].values.tolist()
y = df2['Clus_Label'].values

#train_val split
train_x, val_x, train_y, val_y = train_test_split(tweet_list,y, random_state=12, shuffle=True, stratify=y, test_size=0.10)

#using the train data for all the experiments here
x_train = train_x
y_train = train_y
# the train data (without the validation data)will  be used to train the model

# x_test =  tweet_list
# y_test =  y
# #the train data (with the validation data) will be used for the feature attribution side because we need a larger source
# # of data in other to generate a larger vocabulary. Also, this data set is the one that the augmentation will be performed
# # on, so the over lap is neccessary
#
# # test_labels = y_test.copy()
# # train_labels = y_train.copy()
# # print(len(x_train), 'train sequences')
# # print(len(x_test), 'test sequences')
# # #y_train, y_test, val_y= to_categorical(y_train), to_categorical(y_test), to_categorical(val_y)

df3 = df2.loc[df2['Clus_Label'] == 3]  #0 to 3                   #Select cluster by cluster by uisng label index 0 to 5
x_test = df3['tweet_clean'].values.tolist()


#tokenize, convert to sequence and pad sequences
print('Pad sequences (samples x time)')
tokenizer= Tokenizer()
tokenizer.fit_on_texts(x_train)
x_train  = tokenizer.texts_to_sequences(x_train)
x_train  = pad_sequences(x_train, maxlen = maxlen, padding = 'post')
print('x_train shape:', x_train.shape)

val_x = tokenizer.texts_to_sequences(val_x)
val_x = pad_sequences(val_x, maxlen = maxlen, padding = 'post')
print('val_x shape:', val_x.shape)

x_test  = tokenizer.texts_to_sequences(x_test)
x_test  = pad_sequences(x_test, maxlen = maxlen, padding = 'post')
print('x_test shape:', x_test.shape)

word_index = tokenizer.word_index
reverse_index = {value: key for (key, value) in word_index.items()}

max_features = len(word_index) +1
num_words= max_features


#A sample review from the test set. Note that unknown words are replaced with ‘UNK’
def decode_sentence(x, reverse_index):
    # the `-3` offset is due to the special tokens used by keras
    # see https://stackoverflow.com/questions/42821330/restore-original-text-from-keras-s-imdb-dataset
    return " ".join([reverse_index.get(i - 3, 'UNK') for i in x])

print("Decoded sentence: ",decode_sentence(x_test[1], reverse_index))

batch_size = 32
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250

load_model = True
save_model = False
filepath = '/vol/ecrg-solar/kosimadukwe/StanceDetection/'

################################### W2V embedding Stuff ##################################
embeddings_index = {}
embed_file = open(os.path.join('','/vol/ecrg-solar/kosimadukwe/NewTextClassification/GoogleNews-vectors-negative300.txt'), encoding = 'utf-8')
for line in embed_file:                          #for every line in the saved embedding,
    values = line.split()               #split that line and load it into 'values'

    word = values[0]                    #pick the first value in values ie the value with index 0 and load it into 'word'
    coefs = np.asarray(values[1:])      #create an array of the remaining values in values from index 1 to the end and load it into 'coefs'
    embeddings_index[word] = coefs      #then load word and coefs in to the dictionary 'embeddings_index'
embed_file.close()

vocab_size = num_words + 1 # 3000000  #this should be equal to the length of word_index.items
print("Vocab Size",vocab_size)
embedding_matrix = np.zeros((vocab_size, 300))

for word, index in  word_index.items():   #word_index is a dictionary containg the words after tokenization and their index. the index is assigned based on the frequency of occurence of each word. Therefore, the word with index 1 will be the word with the highest frequency of occurence.
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word) #this gets the vector of the word
        if embedding_vector is not None:             #if that vector isn't empty
            embedding_matrix[index] = embedding_vector  # add the the index and the vector to the matrix
            ##########################################################################################################################################################################

if load_model:
    model = tf.keras.models.load_model(os.path.join(filepath, 'Ensemble_MLMA_4Class_All_SansBERT_MC_CNN_NEW2022.h5'))  #ClusterResult_ClassificationModel_LSTM.h5
    model.summary()
else:
    print('Build model...')


# ######LSTM with W2V model ###################
#     model = Sequential()
#     model.add(Embedding(vocab_size, 300, input_length=maxlen, weights=[embedding_matrix], trainable=True))
#     model.add(Bidirectional(LSTM(100, dropout=0.2)))
#     model.add(Dense(5, activation='sigmoid'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
#     model.fit(x_train, y_train, epochs=60, verbose=True, batch_size=64,
#                      validation_data=(val_x, val_y))
#     if save_model:
#         if not os.path.exists(filepath):
#             os.makedirs(filepath)
#         model.save(os.path.join(filepath, 'ClusterResult_ClassificationModel_LSTM.h5'))

    #################CNN with random embeddings model #####################
    inputs = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = Embedding(max_features,
                                   embedding_dims)(inputs)
    out = Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1)(embedded_sequences)
    out = Dropout(0.4)(out)
    out = GlobalMaxPooling1D()(out)
    out = Dense(hidden_dims,
                activation='relu')(out)
    out = Dropout(0.4)(out)
    outputs = Dense(4, activation='sigmoid')(out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=256,
              epochs=8,
              validation_data=(val_x, val_y))
    if save_model:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        model.save(os.path.join(filepath, 'Ensemble_MLMA_4Class_All_SansBERT_MC_CNN_NEW2022.h5'))

n_steps = 200
method = "gausslegendre"
internal_batch_size = 100
nb_samples = 100#x_test.shape[0]
ig  = IntegratedGradients(model,
                          layer=model.layers[1],
                          n_steps=n_steps,
                          method=method,
                          internal_batch_size=internal_batch_size)
#For CNN, layer=model.layers[1] layer 1.  For LSTM layer 0

x_test_sample = x_test[1500:1516] #x_test[:nb_samples]  x_test[0:200]
predictions = model(x_test_sample).numpy().argmax(axis=1)   #Can this take only one sentence at a time or multiple??????
explanation = ig.explain(x_test_sample,
                         baselines=None,
                         target=predictions)

# Metadata from the explanation object
print("Metadata from Explanation Object: ",explanation.meta)

# Data fields from the explanation object
print("Explanation Data Keys: ",explanation.data.keys())

# Get attributions values from the explanation object
attrs = explanation.attributions[0]
print('Attributions shape 1:', attrs.shape)

#Sum attributions
attrs = attrs.sum(axis=1)    #axis=2  ####################################################### you changed something d=for bilistm 9th Sept 2021. it wasnt changed in the bilstm for DA work.  it might be wrong here
print('Attributions shape: 2', attrs.shape)

#We can visualize the attributions for the text instance by mapping the values of the attributions onto a matplotlib colormap.
# Below we define some utility functions for doing this.
from IPython.display import display, HTML
def  hlstr(string, color='white'):
    """
    Return HTML markup highlighting text with the desired color.
    """
    return f"<mark style=background-color:{color}>{string} </mark>"

def colorize(attrs, cmap='PiYG'):
    """
    Compute hex colors based on the attributions for a single instance.
    Uses a diverging colorscale by default and normalizes and scales
    the colormap so that colors are consistent with the attributions.
    """
    import matplotlib as mpl
    from matplotlib import colors
    from matplotlib import cm
    cmap_bound = np.abs(attrs).max()
    norm = colors.Normalize(vmin=-cmap_bound, vmax=cmap_bound)
    cmap = cm.get_cmap(cmap)

    # now compute hex values of colors
    colors = list(map(lambda x: mpl.colors.rgb2hex(cmap(norm(x))), attrs))
    return colors


#Below we visualize the attribution values (highlighted in the text) having the highest positive
# attributions. Words with high positive attribution are highlighted in shades of green and words
# with negative attribution in shades of pink. Stronger shading corresponds to higher attribution values.
# Positive attributions can be interpreted as increase in probability of the predicted class
# (“Positive sentiment”) while negative attributions correspond to decrease in probability of the
# predicted class.

#Visualize attributions
# for i in range(len(x_test_sample)):  #len(x_test)
#     # i = 1
#     x_i = x_test_sample[i]
#     attrs_i = attrs[i]
#     pred = predictions[i]
#     pred_dict = {0: 'Hate', 1: 'Offensive', 2: 'Neither'}
#
#     print('Predicted label =  {}: {}'.format(pred, pred_dict[pred]))
#     words = decode_sentence(x_i, reverse_index).split()
#     colors = colorize(attrs_i)
#     data = "".join(list(map(hlstr, words, colors)))
#
#     with open("data0lstm200steps.html", "a") as file:
#         file.write(data)
#         file.write("<br>")
#         file.write(str(attrs_i))
#         file.write("<br>")
# #
# # HTML("".join(list(map(hlstr, words, colors))))
# # plt.show()


#Drop unimportant words
with open("/vol/ecrg-solar/kosimadukwe/StanceDetection/Ensemble_MLMA_4Class_IGAttrib_Class3_NEW2022_SansBERT_MC"
          ".txt", "a+") as f: #AttributionWithIG_Class2_dropb
    for x in range(len(x_test_sample)):
        attrs_i = attrs[x]  # attrs is a m *n ndarray where m is the number of samples in the test set and n is the length of the sentence or embedding dim (not sure)

        #####median threshold edit
        attrs_i2 = []
        for i in range(len(attrs_i)):
            if attrs_i[i] > 0 or attrs_i[i] < 0:
                attrs_i2.append(attrs_i[i])
        median_threshold = np.mean(attrs_i2)
        ######median threshold edit

        decoded_sample = decode_sentence(x_test_sample[x], reverse_index)  #converting the sequence back to words
        decoded_sample_tokens = decoded_sample.lower().split()
        try:
            for index in range(len(decoded_sample_tokens[x])):
                if attrs_i[index]  <= median_threshold:   #< 0:
                    decoded_sample_tokens[index] = ""
        except IndexError:
            pass
        new_sentence = " ".join(decoded_sample_tokens)
        new_sentence  =new_sentence.replace('unk', '')
        f.write(new_sentence + '\n')