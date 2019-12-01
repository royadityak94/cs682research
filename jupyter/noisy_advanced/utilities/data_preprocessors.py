import numpy as np
from scipy.ndimage import rotate
from skimage.transform import swirl, AffineTransform, warp
from skimage.util import random_noise
from PIL import Image
from skimage import exposure
import tensorflow as tf
from scipy.ndimage.filters import median_filter

def flip_vertical_np(im):
    return np.flipud(im)

def flip_horizontal_np(im):
    return np.fliplr(im)

def rotate_np(im, angle=None):
    sign = -1 if np.random.uniform(low=0, high=1) < 0.5 else 1
    if angle is None:
        angle = np.random.uniform(low=5, high=45) * -1
    angle *= sign
    return rotate(im, angle, reshape=False, mode='nearest')

def flip_rotate(im, choice=None):
    if choice is None: 
        choice = 'vertical' if np.random.uniform(low=0, high=1) < 0.5 else 'horizontal'
    im = flip_vertical_np(im) if choice == 'vertical' else flip_horizontal_np(im)
    return rotate_np(im)

def perform_swirl_transformation(im, angle=None):
    sign = -1 if np.random.uniform(low=0, high=1) < 0.5 else 1
    if angle is None:
        angle = np.random.uniform(low=5, high=45) * -1
    angle *= sign
    return swirl(im, strength=.5, rotation=angle)

def perform_random_affine_transform(im):
    # populating random values
    scale_x, scale_y = np.random.uniform(1.0, 1.35), np.random.uniform(1.0, 1.3)
    tf_x, tf_y = np.random.uniform(0.0, 0.1), np.random.uniform(0.0, 0.1)
    shear = np.random.uniform(0.0, 0.15)
    tf_inv_matrix = AffineTransform(scale=(scale_x, scale_y), shear=shear, translation=(tf_x, tf_y)).inverse
    
    if np.random.uniform(low=0, high=1) < 0.15:
        im = perform_swirl_transformation(im)
    return warp(im, tf_inv_matrix, order=0)

def mixed_transformations(im):
    # Flip Rotate + Swirl + Affine Transformation
    choice = 'vertical' if np.random.uniform(low=0, high=1) < 0.5 else 'horizontal'
    im = flip_vertical_np(im) if choice == 'vertical' else flip_horizontal_np(im)
    return perform_random_affine_transform(perform_swirl_transformation(im))

def get_present_brightness(im):
    im_ = Image.fromarray(im.astype('uint8'), 'RGB').convert('L')
    hist = im_.histogram()
    brightness = len(hist)
    for s in range(len(hist)):
        brightness += (hist[s]/sum(hist)) * (s - len(hist))
    if brightness >= 255:
        return 1
    else:
        return brightness/len(hist)

def add_gaussian_noise(im):
    ''' Adds Gaussian Noise to the Image '''
    final_img = np.zeros_like(im, np.float64)
    _, _, C = im.shape
    for i in range(C):
        var = np.random.randint(2, 10) / 1e4
        noise = random_noise(im[:, :, i], mode='gaussian', var=var)
        final_img[:, :, i] = np.array(255*noise)
    return final_img.astype(np.uint8)

def add_sp_noise(im):
    ''' Adds Salt and Pepper Noise to the Image '''
    final_img = np.zeros_like(im, np.float64)
    _, _, C = im.shape
    for i in range(C):
        amt = np.random.randint(1, 15) / 1e4
        noise = random_noise(im[:, :, i], mode='s&p', amount=amt)
        final_img[:, :, i] = np.array(255*noise)
    return final_img.astype(np.uint8)

def add_poisson_noise(im):
    ''' Adds Poisson Noise to the Image '''
    final_img = np.zeros_like(im, np.float64)
    _, _, C = im.shape
    for i in range(C):
        noise = random_noise(im[:, :, i], mode='poisson')
        final_img[:, :, i] = np.array(255*noise)
    return final_img.astype(np.uint8)

def add_multiplicative_noise(im):
    ''' Adds Multiplicative Noise to the Image '''
    final_img = np.zeros_like(im, np.float64)
    _, _, C = im.shape
    for i in range(C):
        var = np.random.randint(2, 30) / 1e4
        noise = random_noise(im[:, :, i], mode='speckle', var=var)
        final_img[:, :, i] = np.array(255*noise)  
    return final_img.astype(np.uint8)

def random_image_eraser(im):
    ''' Utility for random image masking '''
    im_ = im.copy()
    H, W, C = im_.shape
    point = np.random.randint(int(H/5), int(H/2))
    w_ = np.random.randint(int(W/5), int(W/2))
    h_ = np.random.randint(int(H/5), int(H/2))
    im_[point:point + h_, point:point + w_, :] = np.random.uniform(0, 255)
    return im_.astype(np.uint8)

def correct_low_visibility(im):
    ''' Brightness Correction '''
    if get_present_brightness(im) > 0.4:
        return im
    im_ = im.copy()
    random_brightness = np.random.uniform(.5, 1.5)
    im_[:, :, 2] = im_[:, :, 2] * random_brightness
    im_[:, :, 2][im_[:, :, 2] > 255] = 255
    return (im_*255).astype(np.uint8)

def gamma_correction(im):
    ''' Gamma Correction in the Image '''
    gamma = np.random.uniform(.25, .65)
    return exposure.adjust_gamma(im, gamma, gain=0.9).astype(np.uint8)

def median_filtering(im, k=2):
    ''' Performs Median filtering on the image'''
    corrected = median_filter(im, size=(k, k, k))
    return (corrected*255).astype(np.uint8)

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

def fetch_selected_data_10(x_data, y_data):
    new_x_data, new_y_data = np.zeros_like(x_data), np.zeros_like(y_data)
    keep_list = [1, 9, 2, 6, 8]
    curr_x, curr_y = 0, 0
    for i in range(y_data.shape[0]):
        idx = y_data[i][0]
        if idx not in keep_list:
            continue
        new_x_data[curr_x], new_y_data[curr_y]  = x_data[i], y_data[i]
        curr_x, curr_y = curr_x+1, curr_y+1
    return new_x_data[:curr_x], new_y_data[:curr_y]

def fetch_selected_data_100(x_data, y_data):
    new_x_data, new_y_data = np.zeros_like(x_data), np.zeros_like(y_data)
    keep_list = [13, 58, 81, 89]
    curr_x, curr_y = 0, 0
    for i in range(y_data.shape[0]):
        idx = y_data[i][0]
        if idx not in keep_list:
            continue
        # Automobiles
        if idx in [13, 81, 89]:
            new_x_data[curr_x], new_y_data[curr_y]  = x_data[i], np.array([1])
        # Truck
        else:
            new_x_data[curr_x], new_y_data[curr_y]  = x_data[i], np.array([9])
        curr_x, curr_y = curr_x+1, curr_y+1
    return new_x_data[:curr_x], new_y_data[:curr_y]

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
       
def compute_top_k_accuracy(model, x_test, y_test, k=1):
    all_scores = model.predict_proba(x_test)
    total, correct_prediction = 0, 0
    for scores in all_scores:
        if y_test[total] in np.argsort(scores)[::-1][:k]: 
            correct_prediction += 1
        total += 1
    return (correct_prediction/total)

def averaged_top_k_accuracy(model, x_test, y_test, k=1, epochs=1, batchsize=2000):
    sum_accuracy = 0.0
    for curr_epoch in range(epochs):
        x_test_sample, y_test_sample = get_random_data(x_test, y_test, 0, x_test.shape[0], batchsize)
        sum_accuracy += compute_top_k_accuracy(model, x_test_sample, y_test_sample, k=k)
    return (sum_accuracy/epochs)