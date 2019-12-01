import tensorflow as tf
import numpy as np
from utilities.pythonDB import writeToDB, deleteExistingPrimaryKeyDB
from utilities.data_preprocessors import flip_vertical_np, flip_horizontal_np, rotate_np, flip_rotate, \
perform_swirl_transformation, perform_random_affine_transform, mixed_transformations, add_gaussian_noise, add_sp_noise, \
add_poisson_noise, add_multiplicative_noise, random_image_eraser, correct_low_visibility, gamma_correction, median_filtering
from tensorflow.keras.utils import to_categorical

def get_random_data(data1, data2, low, high, max_samples=1000):
    _, H1, W1, C1 = data1.shape
    _, N = data2.shape
    suff_data1 = np.zeros((max_samples, H1, W1, C1))
    suff_data2 = np.zeros((max_samples, N))
    shuffles = np.random.randint(low, high+1, max_samples)
    for idx in range(shuffles.shape[0]):
        suff_data1[idx] = data1[idx, :, :, :]
        suff_data2[idx] = data2[idx, :]
    return suff_data1, suff_data2

def fetch_selected_data(x_data, y_data):
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

def get_random_shuffle_set(data1, data2, ratios=None, alt=False):
    if ratios is None:
        ratios = {"flip_vertical_np":.1, 'flip_horizontal_np':.1, 'rotate_np':.15, 'flip_rotate':.2, \
                  'perform_swirl_transformation':.1, 'perform_random_affine_transform':.15, 'mixed_transformations':.2}
    _, H1, W1, C1 = data1.shape
    N, _ = data2.shape
    total = int(np.round(np.sum([ratios.get(k) for k in ratios.keys()]) * N, 1))
    new_N = total if total != N else N
    
    shuffles = np.random.randint(0, N, new_N)
    low = 0
    x_train_ret, y_train_ret, img_idx  = np.zeros((new_N, H1, W1, C1)), np.zeros((new_N, 1)), 0

    for key in ratios.keys():
        high = int(ratios.get(key) * N)
        data = shuffles[low:low+high]
        low += high
        for idxs in data:
            x_train_ret[img_idx] = data1[idxs]
            if alt:
                x_train_ret[img_idx] = TransformDataset().return_function_alt(key, data1[idxs])
            else:
                x_train_ret[img_idx] = TransformDataset().return_function(key, data1[idxs])
            y_train_ret[img_idx] = data2[idxs]
            img_idx += 1
    return x_train_ret, y_train_ret

class TransformDataset(object):
    def scale(self, X, x_min=0, x_max=1):
        nom = (X-X.min(axis=0))*(x_max-x_min)
        denom = X.max(axis=0) - X.min(axis=0)
        denom[denom==0] = 1
        X_new = x_min + nom/denom
        return (X_new)
    def return_function(self, name, im):
        return getattr(self, 'if_' + name)((correct_low_visibility(im)))
    def return_function_alt(self, name, im):
        return getattr(self, 'if_' + name)((correct_low_visibility(im)))
    def if_flip_vertical_np(self, im):
        return self.scale(flip_vertical_np(im))
    def if_flip_horizontal_np(self, im):
        return self.scale(flip_horizontal_np(im))
    def if_rotate_np(self, im):
        return self.scale(rotate_np(im))
    def if_flip_rotate(self, im):
        return self.scale(flip_rotate(im))
    def if_perform_swirl_transformation(self, im):
        return self.scale(perform_swirl_transformation(im))
    def if_perform_random_affine_transform(self, im):
        return self.scale(perform_random_affine_transform(im))
    def if_mixed_transformations(self, im):
        return self.scale(mixed_transformations(im))
    def if_add_gaussian_noise(self, im):
        return self.scale(add_gaussian_noise(im))
    def if_add_sp_noise(self, im):
        return self.scale(add_sp_noise(im))
    def if_add_poisson_noise(self, im):
        return self.scale(add_poisson_noise(im))
    def if_add_multiplicative_noise(self, im):
        return self.scale(add_multiplicative_noise(im))
    def if_random_image_eraser(self, im):
        return self.scale(random_image_eraser(im))
    def if_correct_low_visibility(self, im):
        return self.scale(correct_low_visibility(im))
    def if_gamma_correction(self, im):
        return self.scale(gamma_correction(im))
    def if_median_filtering(self, im):
        return self.scale(median_filtering(im))
    
def return_augmented_dataset(x_train_i, y_train_i, MAX_SAMPLES):
    
    cifar = tf.keras.datasets.cifar100
    (x_train_o, y_train_o), (x_test_o, y_test_o) = cifar.load_data()

    x_train_o1, y_train_o1 = fetch_selected_data(x_train_o, y_train_o)
    x_test_o1, y_test_o1 = fetch_selected_data(x_test_o, y_test_o)
    x_train_np, y_train_np = np.concatenate((x_train_o1, x_test_o1), axis=0), np.concatenate((y_train_o1, y_test_o1), axis=0)
    y_train_np = to_categorical(y_train_np, num_classes=10)
    
    if MAX_SAMPLES == -1:
        MAX_SAMPLES = x_train_i.shape[0] + y_train_np.shape[0]
    
    ratio1 = {"flip_vertical_np":.25, "flip_horizontal_np":.25, "rotate_np":.15, "flip_rotate":.25, 
              'perform_swirl_transformation':.1}
    ratio2 = {'perform_random_affine_transform':.45, 'mixed_transformations':.45}
    ratio3 = {'random_image_eraser':1}
    ratio4 = {"add_gaussian_noise": 0.2, "add_sp_noise": 0.4, "add_poisson_noise":.2, 'add_multiplicative_noise':.2}
    ratio5 = {'gamma_correction':1}
    ratio6 = {"flip_rotate":1}
    
    # CIFAR 100 + CIFAR 10
    x_train, y_train = np.concatenate((x_train_i, x_train_np), axis=0), np.concatenate((y_train_i, y_train_np), axis=0)
    
    if MAX_SAMPLES < int(x_train.shape[0]/5) and MAX_SAMPLES != -1:
        x_train, y_train = x_train[:(5*MAX_SAMPLES)], y_train[:(5*MAX_SAMPLES)]
    
    # Augmented Dataset
    nw_x_train, nw_y_train = get_random_data(x_train, y_train, 0, x_train.shape[0], int(MAX_SAMPLES/3))
    x_train_r1, y_train_r1 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio1)
    x_train_r2, y_train_r2 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio2)
    x_train_r3, y_train_r3 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio3)
    x_train_r4, y_train_r4 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio4)
    x_train_r5, y_train_r5 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio5)
    x_train_r6, y_train_r6 = get_random_shuffle_set(nw_x_train, nw_y_train, ratio6)
    
    final_x_train = np.concatenate((x_train, x_train_r1, x_train_r2, x_train_r3, x_train_r4, x_train_r5, x_train_r6), axis=0)
    final_y_train = np.concatenate((y_train, y_train_r1, y_train_r2, y_train_r3, y_train_r4, y_train_r5, y_train_r6), axis=0)
    
    x_train_shuffled, y_train_shuffled = get_random_data(final_x_train, final_y_train, 0, \
                                                         final_x_train.shape[0], max_samples=MAX_SAMPLES)
    return x_train_shuffled, y_train_shuffled