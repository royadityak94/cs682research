import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv3D, Flatten, MaxPool2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import metrics
import matplotlib.pyplot as plt
from utilities.rectified_adam import RAdam
import numpy as np
import sys
from utilities.pythonDB import writeToDB, deleteExistingPrimaryKeyDB, recordsExists
from utilities.data_preprocessors import get_random_data, MetricsAfterEachEpoch
from keras import regularizers
from keras.constraints import unit_norm

global MAX_EPOCHS, MAX_BATCH_SIZE, architecture, label, keywords
MAX_EPOCHS, MAX_BATCH_SIZE = 15, 1024
architecture = 'ZFNet'
label, keywords = 'noisy_bl-experiments', 'benchmark_testing'
    
def model_extractor(activation_func, weight_decay=1e-4):
    # Creating a ZFNet Classifier
    model = Sequential()

    #Instantiating Layer 1
    model.add(Conv2D(96, kernel_size=(7, 7), strides=(2, 2), activation=activation_func, padding='same', 
                    kernel_constraint=unit_norm(), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2), name='pool1'))
    model.add(BatchNormalization())

    # #Instantiating Layer 2
    model.add(Conv2D(256, kernel_size=(5, 5), strides=(2, 2), activation=activation_func, padding='same', 
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='pool2'))
    model.add(BatchNormalization())

    # #Instantiating Layer 3
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay)))

    # #Instantiating Layer 4
    model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same', 
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(BatchNormalization())
        
    # #Instantiating Layer 5
    model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation=activation_func, padding='same', 
                    kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), name='pool3'))
    model.add(BatchNormalization())
    
    model.add(Flatten())

    #Instantiating Layer 6
    model.add(Dense(4096, activation=activation_func)) 

    #Instantiating Layer 8
    model.add(Dense(4096, activation=activation_func))

    #Output Layer
    model.add(Dense(10, activation='softmax'))
    
    return model

def diff_optimizer(curr_optimizer, activation_func, label=None, curr_epochs=MAX_EPOCHS):
    model = model_extractor(activation_func)
    model.compile(loss="categorical_crossentropy", optimizer=curr_optimizer, validation_data=(x_val, y_val),
                  metrics=["accuracy", "mae", "mse"])
    history = model.fit(x_train, y_train, epochs=curr_epochs, batch_size=1024).history
    score = model.evaluate(x_test, y_test, batch_size=MAX_BATCH_SIZE)
    
    #Saving the model
    if label == 'RAdam':
        model.save("../../saved_models_advanced/{}_{}_{}.hdf5".format('RAdam', activation_func, architecture))
    else: 
        model.save("../../saved_models_advanced/{}_{}_{}.hdf5".format(curr_optimizer, activation_func, architecture))
    return score, history


if __name__ == '__main__':
    MAX_TRAINING = 30000
    MAX_VALIDATION = 5000
    MAX_TESTING = 4000
    
    base = 'noisy_dataset/noisy/cifar_10_{}.npy'
    x_train_max, y_train = np.load(base.format('x_train')), np.load(base.format('y_train'))
    x_test_max, y_test = np.load(base.format('x_test')), np.load(base.format('y_test'))
    y_train_max, y_test_max = to_categorical(y_train, num_classes=10), to_categorical(y_test, num_classes=10)
    train_cutoff = int(x_train_max.shape[0] - MAX_VALIDATION*4)

    x_train, y_train = get_random_data(x_train_max, y_train_max, 0, train_cutoff, MAX_TRAINING)
    x_val, y_val = get_random_data(x_train_max, y_train_max, train_cutoff+1, x_train_max.shape[0], MAX_VALIDATION)
    x_test, y_test = get_random_data(x_test_max, y_test_max, 0, x_test_max.shape[0], MAX_TESTING)

    print ("Number of Training Samples: X={}, Y={}".format(x_train.shape[0], y_train.shape[0]))
    print ("Number of Validation Samples: X={}, Y={}".format(x_val.shape[0], y_val.shape[0]))
    print ("Number of Test Samples: X={}, Y={}".format(x_test.shape[0], y_test.shape[0]))
    
    testing_optimizers = ['RAdam', 'Adam', 'NAdam', 'SGD']
    testing_activations = ['tanh', 'relu', 'selu']

    for optimizer in testing_optimizers:
        for activation in testing_activations:
            
            if recordsExists((architecture, label, optimizer, activation)):
                print ("Continuing for Optimizer = {} and Activation = {}".format(optimizer, activation))
                continue
                
            print ("Optimizer-{}, Activation-{}".format(optimizer, activation))
            
            loss, accuracy, mae, mse = {}, {}, {}, {}
            train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict = {}, {}, {}, {}
            
            if optimizer == 'RAdam':
                score_returned, history = diff_optimizer(RAdam(lr=0.005), activation, label='RAdam')
            else:
                score_returned, history = diff_optimizer(optimizer, activation)

            
            bg = (score_returned[0], score_returned[1], score_returned[2], score_returned[3], history.get('loss'),\
              history.get('accuracy'), history.get('mae'), history.get('mae'))
        
            #Delete Existing Primary Keys and then write to DB
            deleteExistingPrimaryKeyDB(optimizer, activation, architecture, label, keywords)
            writeToDB(optimizer, activation, architecture, bg, label, keywords)
    
    print ("Completed Execution for architecture={}".format(architecture))