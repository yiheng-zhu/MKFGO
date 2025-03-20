import os
from common_methods import read_name_list, create_dir, read_result, read_distance, read_label
from config import post_deal_script_file, min_go_prob, hc_test_name, \
    all_model_number, hc_cross_entropy_dir, hc_round_name, hc_cross_entropy_name, \
    hc_final_cross_entropy_dir, hc_final_cross_entropy_name, \
    hc_distance_dir, hc_average_distance_dir, k_number_dict, hc_data_dir, hc_train_label, \
    hc_final_triplet_dir, hc_final_triplet_name, hc_final_combine_name, hc_final_combine_dir, hc_combine_weight_dict

import subprocess
import numpy as np

def save_cross_entropy_results(resultdir, go_type, term_list_file, method_name, test_name_list, test_predict_matrix):  # save cross entropy results

    create_dir(resultdir)
    term_list = read_name_list(term_list_file)

    for i in range(len(test_name_list)):

        sub_result_dir = os.path.join(resultdir, test_name_list[i])
        os.makedirs(sub_result_dir)
        result_file = os.path.join(sub_result_dir, method_name + "_" + go_type)

        f = open(result_file, "w")

        for j in range(len(term_list)):
            if(test_predict_matrix[i][j]>=min_go_prob):
                f.write(term_list[j] + " " + go_type[1:] + " " + str(test_predict_matrix[i][j]) + "\n")

        f.close()

    cmd = "python2 " + post_deal_script_file + " " + resultdir + " " + go_type + " " + method_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while post-dealing go terms {e}")

def create_average_cross_entropy_results(workdir, go_type):  # create average cross entropy results

    test_name_list_file = os.path.join(workdir, go_type, hc_test_name)
    test_name_list = read_name_list(test_name_list_file)

    for name in test_name_list:

        final_result_dict = dict()

        for i in range(1, all_model_number + 1):

            result_file = os.path.join(workdir, go_type, hc_cross_entropy_dir, hc_round_name + str(i), name, hc_cross_entropy_name + "_" + go_type)
            result_dict = read_result(result_file)

            for term in result_dict:
                if(term not in final_result_dict):
                    final_result_dict[term] = result_dict[term]
                else:
                    final_result_dict[term] = final_result_dict[term] + result_dict[term]


        sub_result_dir = os.path.join(workdir, go_type, hc_final_cross_entropy_dir, name)
        create_dir(sub_result_dir)
        final_result_file = os.path.join(sub_result_dir, hc_final_cross_entropy_name + "_" + go_type)

        f = open(final_result_file, "w")
        for term in final_result_dict:

            final_result_dict[term] = final_result_dict[term]/all_model_number

            if(final_result_dict[term]>=min_go_prob):
                f.write(term + " " + go_type[1] + " " + str(final_result_dict[term]) + "\n")

        f.close()

    final_result_dir = os.path.join(workdir, go_type, hc_final_cross_entropy_dir)
    cmd = "python2 " + post_deal_script_file + " " + final_result_dir + " " + go_type + " " + hc_final_cross_entropy_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while post-dealing go terms {e}")


def save_distance(distance_dir, train_predict, test_predict, train_name, test_name): # save results for triplet distance

    create_dir(distance_dir)

    for i in range(test_predict.shape[0]):
        result_list = []
        for j in range(train_predict.shape[0]):
            distance = np.sqrt(np.sum(np.square(test_predict[i] - train_predict[j])))
            result_list.append((distance, train_name[j]))

        result_list = sorted(result_list)

        distance_file = os.path.join(distance_dir, test_name[i])
        f = open(distance_file, "w")
        for value, name in result_list:
            f.write(name + " " + str(value) + "\n")
        f.flush()
        f.close()


def create_average_result_per_protein(workdir, protein_id, go_type):

    distance_list = []

    for i in range(1, all_model_number + 1):

        distance_file = os.path.join(workdir, go_type, hc_distance_dir, hc_round_name + str(i), protein_id)
        distance_dict = read_distance(distance_file)
        distance_list.append(distance_dict)

    average_distance_dict = dict()

    for target_id in distance_list[0]:
        for i in range(all_model_number):
            if(target_id not in average_distance_dict):
                average_distance_dict[target_id] = distance_list[i][target_id]
            else:
                average_distance_dict[target_id] = average_distance_dict[target_id] + distance_list[i][target_id]

    for target_id in average_distance_dict:
        average_distance_dict[target_id] = average_distance_dict[target_id] / all_model_number

    average_distance_list = [(average_distance_dict[target_id], target_id) for target_id in average_distance_dict]
    average_distance_list = sorted(average_distance_list)

    average_distance_file = os.path.join(workdir, go_type, hc_average_distance_dir, protein_id)
    f = open(average_distance_file, "w")
    for distance, target_id in average_distance_list:
        f.write(target_id+" "+str(distance)+"\n")
    f.flush()
    f.close()

def create_average_distance(workdir, go_type):

    test_name_list_file = os.path.join(workdir, go_type, hc_test_name)
    test_name_list = read_name_list(test_name_list_file)

    average_distance_dir = os.path.join(workdir, go_type, hc_average_distance_dir)
    create_dir(average_distance_dir)

    for name in test_name_list:
        create_average_result_per_protein(workdir, name, go_type)


def calculate_triplet_result_per_protein(distance_file, result_file, train_label_dict, go_type): # create single result

    f = open(distance_file, "r")
    text = f.read()
    f.close()
    line_set = text.splitlines()

    weight_dict = dict()
    all_term_list = []

    for i in range(k_number_dict[go_type]):
        values = line_set[i].strip().split()
        weight_dict[values[0]] = (float(k_number_dict[go_type] - i)) / k_number_dict[go_type]
        all_term_list.extend(train_label_dict[values[0]])

    all_term_list = list(set(all_term_list))

    result_dict = dict()

    for term in all_term_list:

        sum_weight = 0.0
        sum = 0.0

        for protein_id in weight_dict:
            if(term in train_label_dict[protein_id]):
                sum = sum + weight_dict[protein_id]
            sum_weight = sum_weight + weight_dict[protein_id]

        result_dict[term] = sum/sum_weight

    f = open(result_file, "w")
    for protein_id in result_dict:
        if (result_dict[protein_id] >= min_go_prob):
            f.write(protein_id + " " + go_type[1:] + " " + str(result_dict[protein_id]) + "\n")

def create_triplet_results(workdir, go_type):

    test_name_list_file = os.path.join(workdir, go_type, hc_test_name)
    test_name_list = read_name_list(test_name_list_file)

    train_label_file = os.path.join(hc_data_dir, go_type, hc_train_label)
    train_label_dict = read_label(train_label_file)

    final_triplet_result_dir = os.path.join(workdir, go_type, hc_final_triplet_dir)
    create_dir(final_triplet_result_dir)


    for name in test_name_list:
        distance_file = os.path.join(workdir, go_type, hc_average_distance_dir, name)
        create_dir(os.path.join(final_triplet_result_dir, name))
        triplet_result_file = os.path.join(final_triplet_result_dir, name, hc_final_triplet_name + "_" + go_type)
        calculate_triplet_result_per_protein(distance_file, triplet_result_file , train_label_dict, go_type)

    cmd = "python2 " + post_deal_script_file + " " + final_triplet_result_dir + " " + go_type + " " + hc_final_triplet_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while post-dealing go terms {e}")

def combine_single_result(result_file1, result_file2, result_file3, weight, go_type):  # combine single result

    result_dict1 = read_result(result_file1)
    result_dict2 = read_result(result_file2)

    term_list1 = result_dict1.keys()
    term_list2 = result_dict2.keys()
    result_dict3 = dict()

    all_term_list= list(set(term_list1) and set(term_list2))

    for term in all_term_list:

        if (term in result_dict1):
            value1 = result_dict1[term]
        else:
            value1 = 0

        if(term in result_dict2):
            value2 = result_dict2[term]
        else:
            value2 = 0

        result_dict3[term] = weight * value1 + (1-weight) * value2

    f = open(result_file3, "w")
    for term in result_dict3:
        if(result_dict3[term]>=min_go_prob):
            f.write(term + " " + go_type[1:] + " " + str(result_dict3[term]) + "\n")
    f.close()

def combine_results(workdir, go_type):

    final_cross_entropy_result_fir = os.path.join(workdir, go_type, hc_final_cross_entropy_dir)
    final_triplet_result_dir = os.path.join(workdir, go_type, hc_final_triplet_dir)
    final_combine_result_dir = os.path.join(workdir, go_type, hc_final_combine_dir)

    name_list = os.listdir(final_cross_entropy_result_fir)
    for name in name_list:
        create_dir(os.path.join(final_combine_result_dir, name))
        cross_entropy_result = os.path.join(final_cross_entropy_result_fir, name, hc_final_cross_entropy_name + "_" + go_type + "_new")
        triplet_result = os.path.join(final_triplet_result_dir, name, hc_final_triplet_name + "_" + go_type + "_new")
        combine_result = os.path.join(final_combine_result_dir, name, hc_final_combine_name + "_" + go_type)
        combine_single_result(cross_entropy_result, triplet_result, combine_result, hc_combine_weight_dict[go_type], go_type)

    cmd = "python2 " + post_deal_script_file + " " + final_combine_result_dir + " " + go_type + " " + hc_final_combine_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while post-dealing go terms {e}")









































