import numpy as np
import torch
import random
import math
import sys
import time

max_length = 1024


def read_date_from_information_file(information_file):
    f = open(information_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    return line_set[len(line_set) - 2]


def read_data(workdir, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, type, data_type):  # read feature, label, name

    print("Loading feature.......")

    f = open(workdir + "/" + type + "/" + data_type + "_gene_list", "r")
    text = f.read()
    f.close()

    name_list = text.splitlines()

    feature_array1 = []
    feature_array2 = []
    feature_array3 = []

    count = 0

    for name in name_list:

        feature_file1 = feature_dir1 + "/" + name + ".npy"
        feature_file2 = feature_dir2 + "/" + name + ".npy"
        feature_file3 = feature_dir3 + "/" + name + ".npy"


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

    label_file = workdir + "/" + type + "/" + data_type + "_label_one_hot"
    label = np.loadtxt(label_file)

    print("Loading length.......")

    f = open(sequence_length_file, "r")
    text = f.read()
    f.close()
    sequence_length_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        sequence_length_dict[values[0]] = int(values[1])

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

        if (m > max_length):

            length_array.append(max_length)
            padding_array1.append(np.ones([max_length, mask_dim]))
            padding_array2.append([0 for i in range(max_length)])

        else:

            length_array.append(m)

            x = np.ones([m, mask_dim])
            x_padded = np.pad(x, ((0, max_length - m), (0, 0)), mode='constant', constant_values=0)
            padding_array1.append(x_padded)

            l = [0 for i in range(m)]
            l.extend([1 for i in range(max_length - m)])

            padding_array2.append(l)

    padding_array1 = np.array(padding_array1)
    padding_array2 = np.array(padding_array2)
    length_array = np.array(length_array)

    return  length_array, padding_array1, padding_array2

if __name__ == '__main__':

    workdir = sys.argv[1]
    feature_dir1 = "/data/yihengzhu/GOA/resource/pssm_array/"
    feature_dir2 = "/data/yihengzhu/GOA/resource/ss_index/"
    feature_dir3 = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/temps/entry_array/"
    sequence_length_file = "/data/yihengzhu/GOA/resource/all_protein_length"

    type = "MF"
    data_type = "test"


    feature_array1, feature_array2, feature_array3, label, name_list, sequence_length_dict = read_data(workdir, feature_dir1, feature_dir2, feature_dir3, sequence_length_file, type, data_type)
    print(feature_array1.shape)
    print(feature_array2.shape)
    print(feature_array3.shape)
    print(label.shape)

    batch_size = 256
    mask_dim = 256

    data_list = create_batch(feature_array1, feature_array2, feature_array3, label, name_list, batch_size, True)

    count = 0
    for data in data_list:

        start_time = time.time()

        current_feature1, current_feature2,  current_feature3, current_label, current_name = data
        length_array, padding_array = create_padding(current_name, sequence_length_dict, mask_dim)
        count = count + 1

        end_time = time.time()
        run_time = end_time - start_time

        print("The " + str(count) + "-th training bacth, running time is " + str(run_time * 1000) + " ms")


        print(current_feature1[0])
        print(current_feature2[0])
        print(current_feature3[0])
        print(current_label.shape)
        print(current_name.shape)
        print(padding_array.shape)
        print(length_array.shape)

        print()
        break
