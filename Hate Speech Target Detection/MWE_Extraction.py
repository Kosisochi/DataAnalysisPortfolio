### Adapting the code from Zampieri et all (Improving automatic HSD with MWE features)
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import csv
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
# from keras.utils.np_utils import to_categorical


from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

'''
UTILS

'''


def read_founta(path_file):
    reader = csv.reader(open(path_file, 'r', encoding='UTF-8'), delimiter=',')
    TWEET = 0
    TARGET = 1
    tweets = []
    targets = []
    for row in reader:
        tweets.append(row[TWEET])
        targets.append(row[TARGET])
    vocab = {
        "<pad>": 0,
        "<unk>": 1
    }
    for tweet in tweets:
        for word in tweet.split():
            vocab[word] = len(vocab)

    targets_tokenize = []
    for target in targets:
        targets_tokenize.append(target)

    return tweets, targets_tokenize, vocab


def plot_history_acc(history, path):
    import matplotlib.pyplot as plt
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(path)
    plt.close()


def plot_history_loss(history, path):
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.savefig(path)
    plt.close()


def write_prediction_founta(path_file, prediction_file, prediction):
    r"""

    :param path_file:
    :param prediction:
    :return:
    """
    # path_file_write = path_file.split('.')[0] + '-ref.tsv'
    writer = csv.writer(open(prediction_file, 'w', encoding='utf-8'), delimiter='\t')
    # writer_ref = csv.writer(open(reference_file, 'w', encoding='utf-8'), delimiter='\t')
    reader = csv.reader(open(path_file, 'r', encoding='utf-8'), delimiter='\t')
    count_prediction = 0
    for row in reader:
        writer.writerow([row[0], prediction[count_prediction]])    #prediction is the output of this function which reurns a list containin normal, abusive and hateful
        # writer_ref.writerow([row[0], row[2], row[3], row[4]])
        count_prediction += 1
    assert (count_prediction == len(prediction))


def prediction_to_class_softmax(prediction):
    r"""
    :param prediction: list of probabilities.
    :return class_prediction: list of classes.
    """
    import numpy as np
    class_prediction = []
    for prob in prediction:
        prob_max = np.argmax(prob)
        if prob_max == 0:
            class_prediction.append('hateful')
        if prob_max == 1:
            class_prediction.append('abusive')
        if prob_max == 2:
            class_prediction.append('none')
    return class_prediction


def load_embeddings(path_file, size=512):
    reader = csv.reader(open(path_file, 'r', encoding="utf-8"), delimiter="\t")
    embeddings = []
    for row in reader:
        vector = np.zeros(size)
        assert len(row) == size
        for v_index in range(len(row)):
            vector[v_index] = float(row[v_index])
        embeddings.append(vector)

    return np.array(embeddings)


def one_hot_weight(vocab):
    one_hot = []
    for k, v in vocab.items():
        if '<pad>' == 0:
            one_hot.append(np.zeros(len(vocab)))
        else:
            vector = np.zeros(len(vocab))
            vector[v - 1] = 1
            one_hot.append(vector)
    return np.array(one_hot)


def reconstruct_sentence(tweet):
    tweets_reconstruct = []
    tweet_reconstruct = ""
    for word in tweet:
        tweet_reconstruct += word + " "
    return tweet_reconstruct


'''

EXTRACT MWE FEATURES

'''
import spacy_udpipe

spacy_udpipe.download('en')
nlp = spacy_udpipe.load('en')


def read_lexicon(path_file):
    import csv
    lexicon = []
    file = csv.reader(open(path_file, 'r', encoding='utf-8'), delimiter="\t")
    first_row = True
    len_lexicon = 0
    for row in file:
        if first_row:
            len_lexicon = int(row[0])
            first_row = False
        else:
            lexicon.append((row[0], row[1], row[2]))
    assert (len(lexicon) == len_lexicon)
    return lexicon


def lemmatize_tweet(tweet):
    parsing = nlp(tweet)
    tweet_lemmatize = ''
    for word in parsing:
        tweet_lemmatize += word.lemma_ + ' '
    return tweet_lemmatize


def read_mwes(path, vocab_mwe, size):
    import csv
    import numpy as np
    reader = csv.reader(open(path, 'r', encoding='UTF-8'), delimiter='\t')
    mwe_features = []
    for row in reader:
        vector = []
        mwes = []
        for index_word in range(len(row)):
            if row[index_word] != '':
                mwes.append(row[index_word])
        for pad in range(size - len(mwes)):
            vector.append(vocab_mwe['<pad>'])
        for mwe in mwes:
            vector.append(vocab_mwe[mwe])

        mwe_features.append(np.array(vector))

    return np.array(mwe_features)


def load_vocab_mwe(path):
    file_vocab_mwe = open(path, 'r', encoding=('utf-8'))
    vocab = {}
    for line in file_vocab_mwe.readlines():
        vocab[line.split()[0]] = int(line.split()[1])
    return vocab


def write_vector(path, vector_mwe):
    import csv
    writer = csv.writer(open(path, 'w', encoding="utf-8"), delimiter="\t")
    for vector in vector_mwe:
        writer.writerow(vector.tolist())


def load_vector(path, size):
    import csv
    import numpy as np
    vectors = []
    reader = csv.reader(open(path, 'r', encoding="utf-8"), delimiter="\t")
    for row in reader:
        assert len(row) == size
        vector = np.zeros(size)
        for v_index in range(len(row)):
            vector[v_index] = float(row[v_index])
        vectors.append(vector)

    return np.array(vectors)


####   VARIABLES ####
embeddings = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/NewTextClassification/GoogleNews-vectors-negative300.txt'
train = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/TrainFounta.csv'
dev = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/DevFounta.csv'
test = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/TestFounta.csv'
# mwe_embeddings = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/lexicons/lexicon_mweall.voc'
mwe_embeddings = '/home/kosimadukwe/Downloads/Stance Detection/MWE for HSD/MWE-HSD-main/lexicons/lexicon_mweall.txt'
mwe_one_hot = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/lexicons/lexicon_mweall.voc'
prediction_file = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/MWE_Pred.csv'
path_lexicon = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/lexicons/lexicon_mweall.txt'
epochs = 100
patience = 5
max_sentence_length = 280
batch_size = 100
model_name = '/am/vuwstocoisnrin1.vuw.ac.nz/ecrg-solar/kosimadukwe/StanceDetection/MWE/MWE_Model'

X_train_no_tokenize, Y_train, vocab_train = read_founta(train)
X_dev_no_tokenize, Y_dev, vocab_dev = read_founta(dev)

# path = train
# annotated_only_mwe_features(path_lexicon, path)

#########
#########
#########  FEATURES
#########
#########

X_train = load_embeddings(train.split(".csv")[0] + ".usembed", size=512)
X_dev = load_embeddings(dev.split(".csv")[0] + ".usembed", size=512)

X_TRAIN = [X_train]
X_DEV = [X_dev]
features_spec = []
features_input = []

### ONE HOT  FEATURES ####
vocab_mwe = load_vocab_mwe(path=mwe_one_hot)
train_mwe_features = load_vector(
    train.split(".csv")[0] + '.mwe.' + mwe_one_hot.split("/")[-1].split(".voc")[0],
    size=max_sentence_length)  # TrainFounta.mwe.lexicon_mweall  #this is the one hot encoding of the train set
dev_mwe_features = load_vector(
    dev.split(".csv")[0] + '.mwe.' + mwe_one_hot.split("/")[-1].split(".voc")[0],
    size=max_sentence_length)  # DevFounta.mwe.lexicon_mweall   #this is the one hot encoding of the dev set
X_TRAIN.append(train_mwe_features)
X_DEV.append(dev_mwe_features)

mwe_categories_spec = {'name': 'MWE_One_hot',
                       'output_dim': len(vocab_mwe),
                       'weights': one_hot_weight(vocab_mwe),
                       'trainable': False,
                       'input_dim': len(vocab_mwe),
                       'initializer': 'uniform',
                       'mask_zero': False}

####### EMBEDDING FEATURES ####
lexicon_mwe = read_lexicon(mwe_embeddings)
vocab_mwe = {'<pad>': 0}
for mwe in lexicon_mwe:
    for word in mwe[0].split():
        if word not in vocab_mwe:
            vocab_mwe[word] = len(vocab_mwe)
from gensim.models import KeyedVectors, fasttext
import numpy as np

matrix = []
embed = KeyedVectors.load_word2vec_format(embeddings, binary=False)
for word, key in vocab_mwe.items():
    if word in embed.vocab:
        matrix.append(embed[word])
    else:
        matrix.append(np.zeros(300))
print(len(vocab_mwe), len(matrix))
print(np.array(matrix).shape)
print(np.vstack(matrix).shape)
np_matrix = np.vstack(matrix)
tf_matrix = tf.convert_to_tensor(np_matrix, dtype=tf.int64)
# print("is tf matrix a tensor: ", tf.is_tensor(tf_matrix))  #True
train_mwe_features = read_mwes(
    train.split(".csv")[0] + '.mwe.' + mwe_embeddings.split("/")[-1], vocab_mwe,
    size=15)  # need to generate this  txt file
dev_mwe_features = read_mwes(
    dev.split(".csv")[0] + '.mwe.' + mwe_embeddings.split("/")[-1], vocab_mwe,
    size=15)
X_TRAIN.append(train_mwe_features)
X_DEV.append(dev_mwe_features)

mwe_embeddings_spec = {'name': 'MWEs_embeddings',
                       'output_dim': 300,
                       'weights': tf_matrix,
                       'trainable': False,
                       'input_dim': len(vocab_mwe),
                       'initializer': 'uniform',
                       'mask_zero': False}


# np.array(matrix)
# 'weights': np.vstack(matrix)

# DO CLASSIFICATION
def use_mwe_embeddings_w2v(mwe_categories_features, mwe_embeddings_features, sentence_len):
    try:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    inputs = [Input(512, name='use_embed'),  # USE features
              Input(shape=(sentence_len,), name=mwe_categories_features["name"]),  # MWE categories features
              Input(shape=(15,), name=mwe_embeddings_features["name"])]  # MWE embeddings features

    # USE branch
    dense_use = Dense(256, activation='relu')(inputs[0])
    print("is dense_use a Tensor : ", tf.is_tensor(dense_use))

    # MWE categories branch
    embeddings_one_hot = Embedding(input_dim=mwe_categories_features['input_dim'],
                                   output_dim=mwe_categories_features['output_dim'],
                                   input_length=sentence_len,
                                   weights=[mwe_categories_features['weights']],
                                   embeddings_initializer=mwe_categories_features['initializer'],
                                   trainable=mwe_categories_features['trainable'],
                                   mask_zero=mwe_categories_features['mask_zero'])(
        inputs[1])
    cnn_one_hot = Conv1D(32, 3, activation='relu')(embeddings_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(16, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    cnn_one_hot = Conv1D(8, 3, activation='relu')(cnn_one_hot)
    cnn_one_hot = MaxPooling1D()(cnn_one_hot)
    output_cnn_one_hot = Flatten()(cnn_one_hot)

    # print("is output_cnn_one_hot a Tensor : ", tf.is_tensor(output_cnn_one_hot)) #True

    # MWE embeddings branch
    embeddings_embeddings = Embedding(input_dim=mwe_embeddings_features['input_dim'],
                                      output_dim=mwe_embeddings_features['output_dim'],
                                      input_length=sentence_len,
                                      weights=[mwe_embeddings_features['weights']],
                                      embeddings_initializer=mwe_embeddings_features['initializer'],
                                      trainable=mwe_embeddings_features['trainable'],
                                      mask_zero=mwe_embeddings_features['mask_zero'])(inputs[2])
    # print('embeddingEmbedding Dtype',embeddings_embeddings.dtype)  #<dtype: 'float32'>
    # print('embeddingEmbedding Shape', embeddings_embeddings.shape)   #(None, 15, 300)
    # print("is EmbEMb a Tensor : ",tf.is_tensor(embeddings_embeddings))  #True
    # print(tf.is_tensor(x))

    lstm_mwe_embed = LSTM(192)(embeddings_embeddings)
    # print("is lstm_mwe_embed  a Tensor : ", tf.is_tensor(lstm_mwe_embed))  #True

    concat = Concatenate()([dense_use, output_cnn_one_hot, lstm_mwe_embed])
    print('Concat Shape: ', concat.shape)

    dense = Dense(256, activation="relu")(concat)

    output = Dense(3, activation='sigmoid')(dense)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])
    model.summary()
    return model


model = use_mwe_embeddings_w2v(mwe_categories_spec, mwe_embeddings_spec, max_sentence_length)
Y_train = np_utils.to_categorical(Y_train)
Y_dev = np_utils.to_categorical(Y_dev)

print("Training...")
checkpoint = ModelCheckpoint(model_name + '.h5', monitor='val_loss', verbose=1,
                             save_best_only=True,
                             mode='min')
earlyStopping = EarlyStopping(monitor="val_loss", patience=patience, verbose=1, mode="min")
callbacks_list = [checkpoint, earlyStopping]
history = model.fit(X_TRAIN, Y_train, batch_size=batch_size,
                    validation_data=(X_DEV, Y_dev), epochs=epochs,
                    callbacks=callbacks_list)

plot_history_acc(history, model_name + '-acc.png')
plot_history_loss(history, model_name + '-loss.png')

#######################################             #######################################
#######################################             #######################################
#######################################            #######################################
####################################### TESTING     #######################################
#######################################            #######################################
#######################################           #######################################
#######################################            #######################################

X_test_no_tokenize, Y_test, vocab_test = read_founta(test)
model = load_model(model_name + ".h5")
model.summary()

# FEATURES
X_test = load_embeddings(test.split(".csv")[0] + ".usembed", size=512)
X_TEST = [X_test]

###added by me to convert test data to one hot
vocab_mwe = load_vocab_mwe(path=mwe_one_hot)
test_mwe_features = load_vector(
    test.split(".csv")[0] + '.mwe.' + mwe_one_hot.split("/")[-1].split(".voc")[0],
    size=max_sentence_length)  # TestFounta.mwe.lexicon_mweall  #this is the one hot encoding of the train set
X_TEST.append(test_mwe_features)
############################################33

lexicon_mwe = read_lexicon(mwe_embeddings)
vocab_mwe1 = {'<pad>': 0}
for mwe in lexicon_mwe:
    for word in mwe[0].split():
        if word not in vocab_mwe1:
            vocab_mwe1[word] = len(vocab_mwe1)
mwe_features_embeddings = read_mwes(test.split(".csv")[0] + '.mwe.' + mwe_embeddings.split("/")[-1], vocab_mwe1, size=max_sentence_length)  #TestFounta.mwe.lexicon_mweall.txt
X_TEST.append(mwe_features_embeddings)


Y_pred = model.predict(X_TEST)
write_prediction_founta(test, prediction_file, prediction_to_class_softmax(Y_pred))


print("Y Test: ", Y_test)
Y_pred = np.argmax(Y_pred, axis = 1)
print("Y Pred: ", Y_pred)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, Y_pred, average='macro')
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test,Y_pred, average='macro')
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, Y_pred, average='macro')
print('F1 score: %f' % f1)