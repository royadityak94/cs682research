import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv3D, Flatten, MaxPool2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import metrics
import matplotlib.pyplot as plt
from utilities.rectified_adam import RAdam
import numpy as np
import sys
from utilities.pythonDB import writeToDB, deleteExistingPrimaryKeyDB, recordsExists
from utilities.resnet import ResNet18, ResNet34
from utilities.data_preprocessors import get_random_data, MetricsAfterEachEpoch

global MAX_EPOCHS, MAX_BATCH_SIZE, architecture, label, keywords
MAX_EPOCHS, MAX_BATCH_SIZE = 15, 1024
architecture = 'ResNet18'
label, keywords = 'noisy_bl-experiments', 'benchmark_testing'
    
def diff_optimizer(curr_optimizer, model, activation_func, label=None, curr_epochs=MAX_EPOCHS):
    model.compile(loss="categorical_crossentropy", optimizer=curr_optimizer, validation_data=(x_val, y_val),
                  metrics=["accuracy", "mae", "mse"])
    history = model.fit(x_train, y_train, epochs=curr_epochs, batch_size=1024).history
    score = model.evaluate(x_test, y_test, batch_size=MAX_BATCH_SIZE)
    
    #Saving the model
    if label == 'RAdam':
        model.save("../../saved_models_noisy/{}_{}_{}.hdf5".format('RAdam', activation_func, architecture))
    else: 
        model.save("../../saved_models_noisy/{}_{}_{}.hdf5".format(curr_optimizer, activation_func, architecture))
    return score, history

if __name__ == '__main__':
    MAX_TRAINING = 20000
    MAX_VALIDATION = 2000
    MAX_TESTING = 5000
    
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
    
    img_rows, img_cols, img_channels = x_train[0].shape
    nb_classes = y_test[0].shape[0]

    for activation in testing_activations:
        for optimizer in testing_optimizers:
            model = ResNet18((img_rows, img_cols, img_channels), nb_classes, activation)

            if recordsExists((architecture, label, optimizer, activation)):
                print ("Continuing for Optimizer = {} and Activation = {}".format(optimizer, activation))
                continue

            loss, accuracy, mae, mse = {}, {}, {}, {}
            train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict = {}, {}, {}, {}

            if optimizer == 'RAdam':
                score_returned, history = diff_optimizer(RAdam(lr=0.005), model, activation, label='RAdam')
            else:
                score_returned, history = diff_optimizer(optimizer, model, activation)

            bg = (score_returned[0], score_returned[1], score_returned[2], score_returned[3], history.get('loss'),\
              history.get('accuracy'), history.get('mae'), history.get('mae'))

            #Delete Existing Primary Keys and then write to DB
            deleteExistingPrimaryKeyDB(optimizer, activation, architecture, label, keywords)
            writeToDB(optimizer, activation, architecture, bg, label, keywords)
    
    print ("Completed Execution for architecture={}".format(architecture))