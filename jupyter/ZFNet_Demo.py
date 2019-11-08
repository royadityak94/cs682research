import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv3D, Flatten, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import metrics
import matplotlib.pyplot as plt
from rectified_adam import RAdam
import numpy as np
import sys
from utilities.pythonDB import writeToDB, deleteExistingPrimaryKeyDB

def get_random_data(data1, data2, low, high, max_samples=100):
    _, H1, W1, C1 = data1.shape
    _, N = data2.shape
    suff_data1 = np.zeros((max_samples, H1, W1, C1))
    suff_data2 = np.zeros((max_samples, N))
    shuffles = np.random.randint(low, high+1, max_samples)
    for idx in range(shuffles.shape[0]):
        suff_data1[idx] = data1[idx, :, :, :]
        suff_data2[idx] = data2[idx, :]
    return suff_data1, suff_data2

class MetricsAfterEachEpoch(tf.keras.callbacks.Callback):
    def on_train_begin(self, scores={}):
        self.loss, self.accuracy, self.mae, self.mse = [], [], [], []
    def on_train_end(self, logs={}):
        return self.loss, self.accuracy, self.mae, self.loss
    def on_epoch_end(self, epoch, scores):
        self.loss.append(scores.get('loss'))
        self.accuracy.append(scores.get('acc'))
        self.mae.append(scores.get('mean_absolute_error'))
        self.loss.append(scores.get('mean_squared_error'))
        return
    
def ZFNetClassifier(activation_func):
    # Creating a ZFNet Classifier
    model = Sequential()

    #Instantiating Layer 1
    model.add(Conv2D(96, kernel_size=(7, 7), strides=(2, 2), activation=activation_func, padding='same'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1'))

    # #Instantiating Layer 2
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), activation=activation_func, padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='pool2'))

    # #Instantiating Layer 3
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same'))

    # #Instantiating Layer 4
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same'))

    # #Instantiating Layer 5
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same'))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool3'))

    model.add(Flatten())

    #Instantiating Layer 6
    model.add(Dense(4096, activation=activation_func)) 

    # #Instantiating Layer 8
    model.add(Dense(4096, activation=activation_func))

    #Output Layer
    model.add(Dense(10, activation='softmax'))
    
    return model

def ZFNetEvaluator(curr_optimizer, activation_func, label=None, curr_epochs=25):
    model = ZFNetClassifier(activation_func)
    model.compile(loss="categorical_crossentropy", optimizer=curr_optimizer, validation_data=(x_val, y_val),
                  metrics=["accuracy", "mae", "mse"])
    history = model.fit(x_train, y_train, epochs=curr_epochs, callbacks=[MetricsAfterEachEpoch()], batch_size=2048).history
    score = model.evaluate(x_test, y_test, batch_size=512)
    #Saving the model
    if label == 'RAdam':
        model.save("../saved_models/{}_{}_{}.hdf5".format('RAdam', activation_func, "ZFNet"))
    else: 
        model.save("../saved_models/{}_{}_{}.hdf5".format(curr_optimizer, activation_func, "ZFNet"))
    return score, history


if __name__ == '__main__':
    MAX_TRAINING = 25000
    MAX_VALIDATION = 5000
    MAX_TESTING = 4000
    correct_class = {1 : 'airplane', 2 : 'automobile', 3 : 'bird', 4 : 'cat', 5 : \
                     'deer', 6 : 'dog', 7 : 'frog', 8 : 'horse', 9 : 'ship', 10 : 'truck'}
    
    #Loading CIFAR-10 dataset
    cifar = tf.keras.datasets.cifar10 
    (x_train, y_train), (x_test, y_test) = cifar.load_data()

    # Keeping the data between 0 and 1
    x_train_max, x_test_max = x_train / 255.0, x_test / 255.0
    y_train_max = to_categorical(y_train, num_classes=10)
    y_test_max = to_categorical(y_test, num_classes=10)
    
    train_cutoff = int(x_train_max.shape[0] - MAX_VALIDATION*1.5)

    x_train, y_train = get_random_data(x_train_max, y_train_max, 0, train_cutoff, MAX_TRAINING)
    x_val, y_val = get_random_data(x_train_max, y_train_max, train_cutoff+1, x_train_max.shape[0], MAX_VALIDATION)
    x_test, y_test = get_random_data(x_test_max, y_test_max, 0, x_test_max.shape[0], MAX_TESTING)

    print ("Number of Training Samples: X={}, Y={}".format(x_train.shape[0], y_train.shape[0]))
    print ("Number of Validation Samples: X={}, Y={}".format(x_val.shape[0], y_val.shape[0]))
    print ("Number of Test Samples: X={}, Y={}".format(x_test.shape[0], y_test.shape[0]))
    
    testing_optimizers = ['RAdam', 'Adam', 'NAdam', 'SGD']
    testing_activations = ['tanh', 'relu']
    loss, accuracy, mae, mse = {}, {}, {}, {}
    train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict = {}, {}, {}, {}

    for optimizer in testing_optimizers:
        for activation in testing_activations:

            if optimizer == 'RAdam':
                score_returned, history = ZFNetEvaluator(RAdam(lr=0.0005), activation, label='RAdam')
            else:
                score_returned, history = ZFNetEvaluator(optimizer, activation)

            #Test Side Stats
            loss[(optimizer, activation)] = score_returned[0]
            accuracy[(optimizer, activation)] = score_returned[1]
            mae[(optimizer, activation)] = score_returned[2]
            mse[(optimizer, activation)] = score_returned[3]

            #Training Side Stats
            train_loss_dict[(optimizer, activation)] = history.get('loss')
            train_accuracy_dict[(optimizer, activation)] = history.get('acc')
            train_mae_dict[(optimizer, activation)] = history.get('mean_absolute_error')
            train_mse_dict[(optimizer, activation)] = history.get('mean_squared_error')

            bg = (loss, accuracy, mae, mse, train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict)

            #Delete Existing Primary Keys and then write to DB
            deleteExistingPrimaryKeyDB(optimizer, activation, 'ZFNet', 'milestone-experiments', 'benchmark_testing')

            writeToDB(optimizer, activation, 'ZFNet', bg, 'milestone-experiments', 'benchmark_testing')
            
            
    print (loss, accuracy)