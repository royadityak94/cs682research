import tensorflow as tf
import numpy as np
from utilities.data_preprocessors import get_random_data, fetch_selected_data_10, fetch_selected_data_100
from utilities.build_augmentor import get_random_shuffle_set

def create_custom_noisy_set1(x_train_i, y_train_i):
    ratio1 = {'mixed_transformations':1}
    ratio2 = {'random_image_eraser':1}
    ratio3 = {"add_gaussian_noise": 0.2, "add_sp_noise": 0.4, "add_poisson_noise":.2, 'add_multiplicative_noise':.2}
    
    x_train_r1, y_train_r1 = get_random_shuffle_set(x_train_i, y_train_i, ratio1)
    x_train_r2, y_train_r2 = get_random_shuffle_set(x_train_i, y_train_i, ratio2)
    x_train_r3, y_train_r3 = get_random_shuffle_set(x_train_i, y_train_i, ratio3)
    
    final_x_train = np.concatenate((x_train_r1, x_train_r2, x_train_r3), axis=0)
    final_y_train = np.concatenate((y_train_r1, y_train_r2, y_train_r3), axis=0)
    
    x_train_shuffled, y_train_shuffled = get_random_data(final_x_train, final_y_train, 0, \
                                                         final_x_train.shape[0], max_samples=final_x_train.shape[0])
    return x_train_shuffled, y_train_shuffled

def create_custom_noisy_set2(x_train_i, y_train_i):
    ratio1 = {'perform_swirl_transformation':.25, 'perform_random_affine_transform':.55}
    ratio2 = {'flip_rotate':.45, 'rotate_np':.55}
    
    x_train_r1, y_train_r1 = get_random_shuffle_set(x_train_i, y_train_i, ratio1)
    x_train_r2, y_train_r2 = get_random_shuffle_set(x_train_i, y_train_i, ratio2)
    
    final_x_train = np.concatenate((x_train_r1, x_train_r2), axis=0)
    final_y_train = np.concatenate((y_train_r1, y_train_r2), axis=0)
    
    x_train_shuffled, y_train_shuffled = get_random_data(final_x_train, final_y_train, 0, \
                                                         final_x_train.shape[0], max_samples=final_x_train.shape[0])
    return x_train_shuffled, y_train_shuffled

if __name__ == '__main__':
    cifar = tf.keras.datasets.cifar10 
    (x_train, y_train), (x_test, y_test) = cifar.load_data()
    x_train_sub, y_train_sub = fetch_selected_data_10(x_train, y_train)
    x_test_sub, y_test_sub = fetch_selected_data_10(x_test, y_test)

    cifar100 = tf.keras.datasets.cifar100
    (x_train_o, y_train_o), (x_test_o, y_test_o) = cifar100.load_data()
    x_train_o1, y_train_o1 = fetch_selected_data_100(x_train_o, y_train_o)
    x_test_o1, y_test_o1 = fetch_selected_data_100(x_test_o, y_test_o)

    x_train_np, x_test_sub = np.concatenate((x_train_sub, x_train_o1, x_test_o1), axis=0)/255.0, x_test_sub/255.0
    y_train_np = np.concatenate((y_train_sub, y_train_o1, y_test_o1), axis=0)
    x_train_np_sub, y_train_np_sub = get_random_data(x_train_np, y_train_np, 0, x_train_np.shape[0], 10000)
    
    x_train_noisy1, y_train_noisy1 = create_custom_noisy_set1(x_train_np_sub, y_train_np_sub)
    x_train_noisy2, y_train_noisy2 = create_custom_noisy_set2(x_train_np_sub, y_train_np_sub)
    x_train_noisy = np.concatenate((x_train_noisy1, x_train_noisy2), axis=0)
    y_train_noisy = np.concatenate((y_train_noisy1, y_train_noisy2), axis=0)
    x_test_noisy, y_test_noisy = create_custom_noisy_set1(x_test_sub, y_test_sub)
    #x_test_less_noisy, y_test_less_noisy = get_random_shuffle_set(x_test_noisy, y_test_noisy, {'median_filtering':1}, True)
    
    np.save('noisy_dataset/noisy/cifar_10_x_train.npy', x_train_noisy)
    np.save('noisy_dataset/noisy/cifar_10_y_train.npy', y_train_noisy)
    np.save('noisy_dataset/noisy/cifar_10_x_test.npy', x_test_noisy)
    np.save('noisy_dataset/noisy/cifar_10_y_test.npy', y_test_noisy)