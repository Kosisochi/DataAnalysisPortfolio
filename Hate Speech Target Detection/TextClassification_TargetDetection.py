# Text classification for target detection using HateLingo data and Ethos Multilabel data


#Creating the HateLingo 5 Targets datasets with labels   Class = 0, Disability = 1, Ethnicity = 2, Gender = 3, SexualOrientation = 4
#import pandas as pd
# df1 = pd.read_csv("/home/kosimadukwe/PycharmProjects/DataExperiment/Twitter_KPB_Datasets/Class.csv", sep=",")
# df1.drop_duplicates(subset="tweet",keep = False, inplace = True)
# label = [0 for x in range(len(df1))]
# df1["class"] = label
# export_csv_file1 = df1.to_csv('Class.csv')  #/local/scratch/Hate Speech Datasets/SemEval2020-SOLID/TrainTweets_new.csv


'''
This uses the 4 class Hate lingo for training and Ethos for testing
'''


import os

import numpy as np
import pandas as pd
import tensorflow
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
# np.random.seed(12)
# rn.seed(12)
# tf.random.set_seed(12)
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


df1 = pd.read_csv("/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/HateLingo_Reduced_Train.csv", sep=",")
df1['tweet_clean'] = df1['tweet_clean'].astype(str)
train_x = df1['tweet_clean'].values.tolist()
train_y = df1['class'].values

# #train_test split
# train_x, test_x, train_y, test_y = train_test_split(tweet_list,y, random_state=12, shuffle=True, stratify=y, test_size=0.30)
df2 = pd.read_csv("/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/Ethos_Reduced.csv", sep=",")
df2['comment'] = df2['comment'].astype(str)
test_x = df2['comment'].values.tolist()
test_y = df2['class'].values


df3 = pd.read_csv("/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/HateLingo_Reduced_Val.csv", sep=",")
df3['tweet_clean'] = df3['tweet_clean'].astype(str)
val_x= df3['tweet_clean'].values.tolist()
val_y= df3['class'].values
# #train_val split
# train_x, val_x, train_y, val_y = train_test_split(train_x ,train_y, random_state=12, shuffle=True, stratify=y, test_size=0.15)


#tokenize, convert to sequence and pad sequences
tokenizer= Tokenizer()
tokenizer.fit_on_texts(train_x)
train_sequences = tokenizer.texts_to_sequences(train_x)
train_sequences = pad_sequences(train_sequences, maxlen = 50, padding = 'post')
val_sequences = tokenizer.texts_to_sequences(val_x)
val_sequences = pad_sequences(val_sequences, maxlen = 50, padding = 'post')
test_sequences = tokenizer.texts_to_sequences(test_x)
test_sequences = pad_sequences(test_sequences, maxlen =50, padding = 'post')
word_index = tokenizer.word_index
num_words = len(word_index)


embeddings_index = {}
embed_file = open(os.path.join('','/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/NewTextClassification/GoogleNews-vectors-negative300.txt'), encoding = 'utf-8') #/home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/results/Emo_Refined_GloVe2.txt #/home/kosimadukwe/Downloads/Emotional Embeddings/em-glove.6B.300d-20epoch   /home/kosimadukwe/PycharmProjects/PSOforFS/Emotional Embeddings/CounterFitting/results/counter_fitted_vectors.txt
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
            embedding_matrix[index] = embedding_vector  #add the the index and the vector to the matrix


accuracy_scores = []
recall_scores = []
precision_scores=[]
f1_scores =[]
AverageF1_forPnN = []

for i in range(1):
    B = [2]#, 3, 4, 5, 6] #, 7, 8, 9, 10, 11] #, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    tensorflow.random.set_seed(B[i])
    #t2=time.time()
    model = Sequential()
    model.add(Embedding(vocab_size, 300, input_length=50, weights=[embedding_matrix], trainable=True))
    #model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(200, dropout=0.2)))
    #model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adagrad', metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', baseline=0.3, patience=450, verbose=1)
    mc = ModelCheckpoint('/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/Results/HateLingoEthos4Target.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    hist = model.fit(train_sequences, train_y, epochs=500, verbose=False, batch_size=100, validation_data=(val_sequences, val_y),callbacks=[es, mc])

    # load the saved model
    saved_model = load_model('/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/Results/HateLingoEthos4Target.h5')


    pred_prob = saved_model.predict(test_sequences)  # predict probabilities for the test set
    #print(pred_prob)
    trainresults = saved_model.evaluate(train_sequences, train_y)
    print('Train Set Evaluation results:', trainresults)
    testresults = saved_model.evaluate(test_sequences, test_y)
    print('Test Set Evaluation results:', testresults)

    y_pred = np.argmax(pred_prob, axis = 1)
    df2['pred_class'] = y_pred
    cols=['comment','class','pred_class']
    export_csv_file = df2.to_csv(
        '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/Ethos_Reduced_Pred.csv', columns=cols)

    #print('Class Prediction', y_pred)

    #calculate Precision, recall and F1
    #accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(test_y, y_pred)
    accuracy_scores.append(accuracy)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(test_y, y_pred, average='macro')
    precision_scores.append(precision)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test_y, y_pred, average='macro')
    recall_scores.append(recall)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(test_y, y_pred, average='macro')
    f1_scores.append(f1)
    print('F1 score: %f' % f1)
    cnf_matrix = confusion_matrix(test_y, y_pred, labels=[0,1,2,3])
    print(cnf_matrix)
    target_names = ['class 0','class 1', 'class 2','class 3']
    print(classification_report(test_y, y_pred, target_names=target_names))


#print(accuracy_scores)
#print("Time it takes :", round(time.time()-t2, 3), 's')
mean_accuracy = np.mean(np.array(accuracy_scores))
mean_precision = np.mean(np.array(precision_scores))
mean_recall = np.mean(np.array(recall_scores))
mean_f1 = np.mean(np.array(f1_scores))
mean_af1 = np.mean(np.array(AverageF1_forPnN))

std_dev_accuracy = np.std(np.array(accuracy_scores), axis=0, dtype=np.float64)
std_dev_precision = np.std(np.array(precision_scores), axis=0, dtype=np.float64)
std_dev_recall = np.std(np.array(recall_scores), axis=0, dtype=np.float64)
std_dev_f1 = np.std(np.array(f1_scores), axis=0, dtype=np.float64)
std_dev_af1 = np.std(np.array(AverageF1_forPnN), axis=0, dtype=np.float64)

print("Mean Accuracy",mean_accuracy,std_dev_accuracy)
print("Mean Precision",mean_precision, std_dev_precision)
print("Mean Recall",mean_recall, std_dev_recall)
print("Mean F1",mean_f1, std_dev_f1)

print("Accuracy list", accuracy_scores)
print("Precision list", precision_scores)
print("Recall list", recall_scores)
print("F1 list", f1_scores)

