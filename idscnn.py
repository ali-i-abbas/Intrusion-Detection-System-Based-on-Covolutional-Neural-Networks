# preprocessing code is from https://github.com/CynthiaKoopman/Network-Intrusion-Detection/blob/master/DecisionTree_IDS.ipynb


import time
import random

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn import preprocessing
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


# taken with modification from scikit-learn ConfusionMatrixDisplay class
def plot(confusion_matrix, display_labels, title, include_values=True, cmap='viridis',
         xticks_rotation='horizontal', values_format=None, ax=None):
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    fig, ax = plt.subplots(figsize=(7, 7))

    cm = confusion_matrix
    n_classes = cm.shape[0]
    im_ = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    text_ = None

    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)

    if include_values:
        text_ = np.empty_like(cm, dtype=object)
        if values_format is None:
            values_format = '0.2f'

        # print text with appropriate color depending on background
        thresh = (cm.max() - cm.min()) / 2.
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text_[i, j] = ax.text(j, i,
                                  format(cm[i, j], values_format),
                                  ha="center", va="center",
                                  color=color)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(im_, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=display_labels,
           yticklabels=display_labels
           )

    ax.set_xlabel(xlabel="Predicted label", fontsize=14, labelpad=2)
    ax.set_ylabel(ylabel="True label", fontsize=14, labelpad=2)

    ax.set_ylim((n_classes - 0.5, -0.5))
    ax.set_title(title, fontsize=18, pad=3)
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)


def preprocess_data(filename, title='Dataset'):
    print('Preprocessing of the ', title)
    print()

    # attach the column names to the dataset
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

    X = pd.read_csv(filename, header=None, names=col_names)

    print('Dimensions of the ', title, ': ', X.shape)
    print()

    print('Label Distribution of the ', title, ': ')
    print(X['label'].value_counts())
    print()

    # insert code to get a list of categorical columns into a variable, categorical_columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    # Get the categorical values into a 2D numpy array
    df_categorical_values = X[categorical_columns]

    # protocol type
    unique_protocol = sorted(X.protocol_type.unique())
    string1 = 'Protocol_type_'
    unique_protocol2 = [string1 + x for x in unique_protocol]
    # service
    unique_service = sorted(X.service.unique())
    string2 = 'service_'
    unique_service2 = [string2 + x for x in unique_service]
    # flag
    unique_flag = sorted(X.flag.unique())
    string3 = 'flag_'
    unique_flag2 = [string3 + x for x in unique_flag]
    # put together
    dumcols = unique_protocol2 + unique_service2 + unique_flag2
    print(dumcols)
    print()

    df_categorical_values_enc = df_categorical_values.apply(LabelEncoder().fit_transform)

    enc = OneHotEncoder()
    df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
    df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(), columns=dumcols)

    X = X.join(df_cat_data)
    X.drop('flag', axis=1, inplace=True)
    X.drop('protocol_type', axis=1, inplace=True)
    X.drop('service', axis=1, inplace=True)

    # take label column
    labeldf = X['label']
    # change the label column
    labeldf = labeldf.replace(
        {'normal.': 0, 'neptune.': 1, 'back.': 1, 'land.': 1, 'pod.': 1, 'smurf.': 1, 'teardrop.': 1, 'mailbomb.': 1,
         'apache2.': 1,
         'processtable.': 1, 'udpstorm.': 1, 'worm.': 1,
         'ipsweep.': 1, 'nmap.': 1, 'portsweep.': 1, 'satan.': 1, 'mscan.': 1, 'saint.': 1
            , 'ftp_write.': 1, 'guess_passwd.': 1, 'imap.': 1, 'multihop.': 1, 'phf.': 1, 'spy.': 1, 'warezclient.': 1,
         'warezmaster.': 1, 'sendmail.': 1, 'named.': 1, 'snmpgetattack.': 1, 'snmpguess.': 1, 'xlock.': 1,
         'xsnoop.': 1,
         'httptunnel.': 1,
         'buffer_overflow.': 1, 'loadmodule.': 1, 'perl.': 1, 'rootkit.': 1, 'ps.': 1, 'sqlattack.': 1, 'xterm.': 1})
    # put the new label column back
    X['label'] = labeldf

    # Split dataframes into X & Y
    # assign X as a dataframe of feautures and Y as a series of class labels
    size0 = X[X['label'] == 0].shape[0]
    size1 = X[X['label'] == 1].shape[0]
    # print(size0,'-',size1)
    selSize = min(size0, size1)
    index0 = np.where(X['label'] == 0)[0]
    random.shuffle(index0)
    index1 = np.where(X['label'] == 1)[0]
    random.shuffle(index1)
    index = np.concatenate((index0[:selSize], index1[:selSize]))
    random.shuffle(index)
    Y = X.label[index]
    X = X.iloc[index].drop('label', 1)

    #Y = X.label
    #X = X.drop('label', 1)

    # print(Y[Y==0].size)
    # print(Y[Y==1].size)

    scaler1 = preprocessing.MinMaxScaler().fit(X)
    X = scaler1.transform(X)

    print('Dimensions of X:', X.shape)
    print()
    print('Label Distribution for Binary Classification (0 normal, 1 attack): ')
    print(labeldf.value_counts())
    print()

    Y_ohe = tf.keras.utils.to_categorical(Y, 2)
    print('Dimensions of Y after One-Hot Encoding:', Y_ohe.shape)
    print()

    # Pad X with zero columns such that it can be reshaped into dim * dim square
    dim = 32
    pad = np.zeros((X.shape[0], np.int64(np.square(dim)) - X.shape[1]), dtype=np.int64)
    X = np.append(X, pad, axis=1)
    print('Dimensions of X after Padding:', X.shape)
    print()
    X = X.reshape((X.shape[0], dim, dim, 1))
    print('Dimensions of X after Reshaping to Square:', X.shape)
    print()
    print('------------------------------------------------------------------------------------------------------\n')

    return X, Y_ohe, Y


start_time = time.time()

# Read and preprocess train dataset
# X_train, Y_train_ohe, Y_train = preprocess_data('kddcup.data_10_percent_corrected', 'Train Dataset')
X_train, Y_train_ohe, Y_train = preprocess_data('kddcup.data.corrected', 'Train Dataset')


# X_train, X_test, Y_train, Y_test, Y_train_ohe, Y_test_ohe = train_test_split(X_train, Y_train, Y_train_ohe, test_size=0.4)
# Y_test = np.argmax(Y_test_ohe, axis=1, out=None)

# Read and preprocess test dataset
X_test, Y_test_ohe, Y_test = preprocess_data('corrected', 'Test Dataset')

#X_test, X_test0, Y_test_ohe, Y_test_ohe0, Y_test, Y_test0 = train_test_split(X_test, Y_test_ohe, Y_test, test_size=0.4)

# class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

print('Dimensions of the X_train:', X_train.shape)
print('Dimensions of the X_test:', X_test.shape)
print('Dimensions of the Y_train_ohe:', Y_train_ohe.shape)
print('Dimensions of the Y_test_ohe:', Y_test_ohe.shape)
print()

# LeNet-5 CNN
model = tf.keras.Sequential()
# convolutional layer 1
model.add(tf.keras.layers.Conv2D(filters=6,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 activation='relu',
                                 input_shape=(X_train.shape[1], X_train.shape[2], 1)))
# average pooling layer 1
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2)))

#model.add(Dropout(0.5))

# convolutional layer 2
model.add(tf.keras.layers.Conv2D(filters=16,
                                 kernel_size=(5, 5),
                                 strides=(1, 1),
                                 activation='relu'))
# average pooling layer 2
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                           strides=(2, 2)))
model.add(tf.keras.layers.Flatten())
# fully connected
#model.add(Dropout(0.5))
model.add(tf.keras.layers.Dense(units=120,
                                activation='relu'))
#model.add(Dropout(0.5))
model.add(tf.keras.layers.Flatten())
# fully connected
model.add(tf.keras.layers.Dense(units=84, activation='relu'))
#model.add(Dropout(0.5))
# output layer
model.add(tf.keras.layers.Dense(units=Y_train_ohe.shape[1], activation='softmax'))


class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, X_test, Y_test):
        super().__init__()
        self.model = model
        self.X_test = X_test
        self.Y_test = Y_test

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) % 10 != 0:
            return

        Y_test_pred = model.predict_classes(self.X_test)
        cm1 = confusion_matrix(self.Y_test, Y_test_pred).astype(np.float)

        print('\n-------------------------------------------------------------------------')
        print('Confusion matrix in epoch %d' % (epoch + 1))
        print(cm1)

        # accuracy: (tp + tn) / (tp + fp + tn + fn)
        print('\nAccuracy: %f' % ((cm1[1, 1] + cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1] + cm1[1, 0] + cm1[1, 1])))
        # recall: tp / (tp + fn)
        print('Recall (Detection Rate): %f' % (cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])))
        # false alarm rate (FAR): fp / (fp + tn)
        print('False alarm rate (FAR): %f' % (cm1[0, 1] / (cm1[0, 1] + cm1[0, 0])))
        print('-------------------------------------------------------------------------')


performance_cbk = PerformanceVisualizationCallback(model=model, X_test=X_test, Y_test=Y_test)

callbacks_list = [performance_cbk,
                  tf.keras.callbacks.ModelCheckpoint('cnn.h5', monitor='val_accuracy'),
                  tf.keras.callbacks.CSVLogger(filename='cnn.log', append=True)
                  ]

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.SGD(lr=0.1, momentum=0.0, decay=0.0),
              metrics=['accuracy'])

model.summary()

epochs = 30
batch_size = 128
#batch_size = 2048

# train the model
history = model.fit(
    x=X_train,
    y=Y_train_ohe,
    epochs=epochs, batch_size=batch_size, validation_data=(
        X_test, Y_test_ohe),
    verbose=2, callbacks=callbacks_list)

print()
print("--- %s seconds ---" % (time.time() - start_time))
print()

np.savez_compressed('cnn_history.npz', tr_acc=history.history['accuracy'], tr_loss=history.history['loss'],
                    val_acc=history.history['val_accuracy'], val_loss=history.history['val_loss'])

loss_train = min(history.history['loss'])
accuracy_train = max(history.history['accuracy'])

print('Log Loss and Accuracy on Train Dataset:')
print("Loss: {}".format(loss_train))
print("Accuracy: {}".format(accuracy_train))
print()

loss_test = min(history.history['val_loss'])
accuracy_test = max(history.history['val_accuracy'])

print('\nLog Loss and Accuracy on Test Dataset:')
print("Loss: {}".format(loss_test))
print("Accuracy: {}".format(accuracy_test))
print()

plt.clf()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='testing accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.ylim(ymin=0.6, ymax=1.1)
plt.savefig("cnn_accuracy.png", type="png", dpi=300)

plt.clf()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='testing loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig("cnn_loss.png", type="png", dpi=300)

plt.clf()

Y_train_pred = model.predict_classes(X_train)
cm1 = confusion_matrix(Y_train, Y_train_pred).astype(np.float)

print('\n-------------------------------------------------------------------------')
print('Training Dataset Metrics')
print('\nConfusion matrix')
print('[TN, FP]')
print('[FN, TP]')
print(cm1)

# accuracy: (tp + tn) / (tp + fp + tn + fn)
print('\nAccuracy: %f' % ((cm1[1, 1] + cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1] + cm1[1, 0] + cm1[1, 1])))
# recall: tp / (tp + fn)
print('Recall (Detection Rate): %f' % (cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])))
# false alarm rate (FAR): fp / (fp + tn)
print('False alarm rate (FAR): %f' % (cm1[0, 1] / (cm1[0, 1] + cm1[0, 0])))

Y_test_pred = model.predict_classes(X_test)
cm1 = confusion_matrix(Y_test, Y_test_pred).astype(np.float)

# print(Y_test[Y_test==0].size)
# print(Y_test_pred[Y_test_pred==0].size)
# print(Y_test[Y_test==1].size)
# print(Y_test_pred[Y_test_pred==1].size)

print('\n-------------------------------------------------------------------------')
print('Testing Dataset Metrics')
print('\nConfusion matrix')
print('[TN, FP]')
print('[FN, TP]')
print(cm1)

# accuracy: (tp + tn) / (tp + fp + tn + fn)
print('\nAccuracy: %f' % ((cm1[1, 1] + cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1] + cm1[1, 0] + cm1[1, 1])))
# recall: tp / (tp + fn)
print('Recall (Detection Rate): %f' % (cm1[1, 1] / (cm1[1, 1] + cm1[1, 0])))
# false alarm rate (FAR): fp / (fp + tn)
print('False alarm rate (FAR): %f' % (cm1[0, 1] / (cm1[0, 1] + cm1[0, 0])))

cm2 = confusion_matrix(Y_test, Y_test_pred, normalize='true')

print('\nNormalized confusion matrix\n')
print(cm1)

# save the confusion matrices
# np.savez_compressed('cnn_cm.npz', cm1=cm1, cm2=cm2)

title = "Normalized confusion matrix"
plot(confusion_matrix=cm2, display_labels=['Normal', 'Attack'], title=title, include_values=True, cmap=plt.cm.Blues,
     xticks_rotation='vertical')

plt.savefig("cnn_confusion_matrix.png", type="png", dpi=300)
