import numpy as np
import os
import random
import math

from common_methods import read_name_list
from config import plm_test_feature, plm_test_label_onehot, plm_test_name

def read_data(workdir, go_type):  # read feature, label, name

    test_feature_file = os.path.join(workdir, go_type, plm_test_feature)
    test_feature = np.loadtxt(test_feature_file)

    test_label_file = os.path.join(workdir, go_type, plm_test_label_onehot)
    test_label = np.loadtxt(test_label_file)

    test_label = np.array(test_label)
    if test_label.ndim == 1:
        test_label = test_label[np.newaxis, :]

    test_feature = np.array(test_feature)
    if test_feature.ndim == 1:
        test_feature = test_feature[np.newaxis, :]


    test_name_file = os.path.join(workdir, go_type, plm_test_name)
    test_name = read_name_list(test_name_file)


    return test_feature, test_label, np.array(test_name)

def create_batch(feature, label, name, batch_size, is_shuffle=True):  #create batch

    number = len(name)
    index = [i for i in range(number)]
    
    if(is_shuffle):
        random.shuffle(index)

    batch_number = math.ceil(float(number)/batch_size)

    data_list = []
    for i in range(batch_number):

        start = i*batch_size
        end = (i+1)*batch_size
        if(end>number):
            end = number

        current_index = sorted(index[start:end])

        current_feature = feature[current_index]
        current_label = label[current_index]
        current_name = name[current_index]

        data_list.append((current_feature, current_label,current_name))

    return data_list

