import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Conv3D, Flatten, MaxPool2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.python.keras import metrics
import matplotlib.pyplot as plt
from rectified_adam import RAdam
import numpy as np
import sys
from utilities.pythonDB import writeToDB, deleteExistingPrimaryKeyDB
from resnet import ResNet18, ResNet34

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

MAX_TRAINING = 7500
MAX_VALIDATION = 2000
MAX_TESTING = 2000
correct_class = {1 : 'airplane', 2 : 'automobile', 3 : 'bird', 4 : 'cat', 5 : \
                 'deer', 6 : 'dog', 7 : 'frog', 8 : 'horse', 9 : 'ship', 10 : 'truck'}
architecture = 'ResNet34'
label, keywords = 'milestone-experiments', 'benchmark_testing'

#Loading CIFAR-10 dataset
cifar = tf.keras.datasets.cifar10 
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train_max, y_train_max = get_random_data(x_train, y_train, 0, MAX_TRAINING*3, MAX_TRAINING)
x_val_max, y_val_max = get_random_data(x_train, y_train, MAX_TRAINING*3+1, x_train.shape[0], MAX_VALIDATION)
x_test_max, y_test_max = get_random_data(x_test, y_test, 0, x_test.shape[0], MAX_TESTING)

# Keeping the data between 0 and 1 
x_train, x_test = x_train_max / 255.0, x_test_max / 255.0
x_val, y_val = x_val_max/255.0, y_val_max/255.0
y_train = to_categorical(y_train_max, num_classes=10)
y_test = to_categorical(y_test_max, num_classes=10)

print ("Number of Training Samples: X={}, Y={}".format(x_train.shape[0], y_train.shape[0]))
print ("Number of Validation Samples: X={}, Y={}".format(x_val.shape[0], y_val.shape[0]))
print ("Number of Test Samples: X={}, Y={}".format(x_test.shape[0], y_test.shape[0]))

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
    
def ResNetEvaluator(curr_optimizer, x_train, y_test, curr_activation, label=None, curr_epochs=20):
    img_rows, img_cols, img_channels = x_train[0].shape
    nb_classes = y_test[0].shape[0]
    
    model = ResNet34((img_rows, img_cols, img_channels), nb_classes)
    
    model.compile(loss="categorical_crossentropy", optimizer=curr_optimizer, validation_data=(x_val, y_val),
                  metrics=["accuracy", "mae", "mse"])
    history = model.fit(x_train, y_train, epochs=curr_epochs, callbacks=[MetricsAfterEachEpoch()], batch_size=2048).history
    score = model.evaluate(x_test, y_test, batch_size=512)
    #Saving the model
    if label == 'RAdam':
        model.save("../saved_models/{}_{}_{}.hdf5".format('RAdam', curr_activation, architecture))
    else: 
        model.save("../saved_models/{}_{}_{}.hdf5".format(curr_optimizer, curr_activation, architecture))
    return score, history

testing_optimizers = ['RAdam', 'Adam', 'NAdam', 'SGD']
testing_activations = ['relu']

for optimizer in testing_optimizers:
    for activation in testing_activations:
        loss, accuracy, mae, mse = {}, {}, {}, {}
        train_loss_dict, train_accuracy_dict, train_mae_dict, train_mse_dict = {}, {}, {}, {}
        
        
        if optimizer == 'RAdam':
            score_returned, history = ResNetEvaluator(RAdam(lr=0.003), x_train, y_test, activation, label='RAdam')
        else:
            score_returned, history = ResNetEvaluator(optimizer, x_train, y_test, activation)
                
        bg = (score_returned[0], score_returned[1], score_returned[2], score_returned[3], history.get('loss'),\
              history.get('accuracy'), history.get('mae'), history.get('mae'))
        #Delete Existing Primary Keys and then write to DB
        deleteExistingPrimaryKeyDB(optimizer, activation, architecture, label, keywords)
        writeToDB(optimizer, activation, architecture, bg, label, keywords)
        
print ("Final: Loss={}, Accuracy={}".format(loss, accuracy))