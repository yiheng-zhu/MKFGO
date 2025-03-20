import numpy as np
import random
import math
from config import *
from common_methods import read_name_list, read_length_from_record_file

def read_data(pssm_feature_dir, hc_ss_feature_dir, hc_interpro_feature_dir, name_list_file, one_hot_label_file, sequence_length_file):  # read feature, label, name

    print("Loading feature.......")

    name_list = read_name_list(name_list_file)

    feature_array1 = []
    feature_array2 = []
    feature_array3 = []

    count = 0

    for name in name_list:

        feature_file1 = os.path.join(pssm_feature_dir, name + ".npy")
        feature_file2 = os.path.join(hc_ss_feature_dir, name + ".npy")
        feature_file3 = os.path.join(hc_interpro_feature_dir, name + ".npy")

        feature1 = np.load(feature_file1)
        feature2 = np.load(feature_file2)
        feature3 = np.load(feature_file3)

        feature_array1.append(feature1)
        feature_array2.append(feature2)
        feature_array3.append(feature3)

        count = count + 1

        print("Loading the " + str(count) + "-th feature.......")

    feature_array1 = np.array(feature_array1)
    feature_array2 = np.array(feature_array2)
    feature_array3 = np.array(feature_array3)

    print("Loading label.......")

    label = np.loadtxt(one_hot_label_file)
    label = np.array(label)
    if label.ndim == 1:
        label = label[np.newaxis, :]


    print("Loading length.......")

    sequence_length_dict = read_length_from_record_file(sequence_length_file)

    return feature_array1, feature_array2, feature_array3, label, np.array(name_list), sequence_length_dict

def create_batch(feature_array1, feature_array2, feature_array3, label, name, batch_size, is_shuffle=True):  # create batch

    number = len(name)
    index = [i for i in range(number)]

    if (is_shuffle):
        random.shuffle(index)

    batch_number = math.ceil(float(number) / batch_size)

    data_list = []
    for i in range(batch_number):

        start = i * batch_size
        end = (i + 1) * batch_size
        if (end > number):
            end = number

        current_index = sorted(index[start:end])

        current_feature1 = feature_array1[current_index]
        current_feature2 = feature_array2[current_index]
        current_feature3 = feature_array3[current_index]
        current_label = label[current_index]
        current_name = name[current_index]

        data_list.append((current_feature1, current_feature2, current_feature3, current_label, current_name))

    return data_list


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.array([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.array([pad(x, max_len) for x in inputs])

    return output

def create_padding(name_list, sequence_length_dict, mask_dim):

    padding_array1 = []
    padding_array2 = []
    length_array = []

    for name in name_list:

        m = sequence_length_dict[name]

        if (m > sequence_cut_off):

            length_array.append(sequence_cut_off)
            padding_array1.append(np.ones([sequence_cut_off, mask_dim]))
            padding_array2.append([0 for i in range(sequence_cut_off)])

        else:

            length_array.append(m)

            x = np.ones([m, mask_dim])
            x_padded = np.pad(x, ((0, sequence_cut_off - m), (0, 0)), mode='constant', constant_values=0)
            padding_array1.append(x_padded)

            l = [0 for i in range(m)]
            l.extend([1 for i in range(sequence_cut_off - m)])

            padding_array2.append(l)

    padding_array1 = np.array(padding_array1)
    padding_array2 = np.array(padding_array2)
    length_array = np.array(length_array)

    return  length_array, padding_array1, padding_array2