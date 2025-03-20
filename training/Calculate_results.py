import os
import numpy as np
import Evaluate_Improve_Triplet as ei
import Get_Meatures_From_T as gf

pythondir = "/data/yihengzhu/GOA/resource/benchmark/pythonfile/"

def ed_distance(vec1, vec2): # calculate distance

    return np.sqrt(np.sum(np.square(vec1 - vec2)))


def create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, data_type): # save results for triplet distance

    resultdir = workdir + "/" + type + "/resultset/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"
    os.system("rm -rf " + resultdir)
    os.makedirs(resultdir)

    test_number = test_predict.shape[0]
    train_number = train_predict.shape[0]

    for i in range(test_number):

        gene1 = test_name[i]
        vector1 = test_predict[i]

        result_list = []

        for j in range(train_number):

            gene2 = train_name[j]
            vector2 = train_predict[j]

            distance = ed_distance(vector1, vector2)
            result_list.append((distance, gene2))

        result_list = sorted(result_list)

        resultfile = resultdir + "/" + gene1
        f = open(resultfile, "w")
        for value, name in result_list:
            f.write(name + " " + str(value) + "\n")
        f.flush()
        f.close()


def calculate_result(workdir, train_predict, evaluate_predict, test_predict, train_name, evaluate_name, test_name, index, type, round): # save triplet distances for all datasets

    create_result_one(workdir, type, train_predict, evaluate_predict, train_name, evaluate_name, round, index, "evaluate")
    create_result_one(workdir, type, train_predict, test_predict, train_name, test_name, round, index, "test")

    ei.evaluate_result(workdir, type, index, round)

def read_term_list(term_file):  # read term_list

    f = open(term_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def post_deal1(workdir, type, method_name, data_type, index, round): # evaluate results and save results

    resultdir = workdir + "/" + type + "/result/"

    os.system("python2 " + pythondir + "/Find_Parents.py " + resultdir + " " + type + " " + method_name)

    test_label_file = workdir + "/" + type + "/" + data_type + "_gene_label"

    roc_dir = workdir + "/" + type + "/" + method_name + "_roc/"

    if (os.path.exists(roc_dir) == False):
        os.makedirs(roc_dir)

    rocfile = roc_dir + "/roc_" + data_type + "_" + str(index) + "_" + str(round) + ".roc"

    record_file = workdir + "/" + type + "/record"
    f = open(record_file, "a")
    r = os.popen("python2 " + pythondir + "/Evaluate_pipelines.py " + workdir + " " + test_label_file + " " + type + " " + rocfile + " " + method_name)

    info = r.readlines()
    sum_line = ""
    for line in info:
        line = line.strip("\r\n")
        sum_line = sum_line + line

    f.write(sum_line + "\n\n")
    f.flush()
    f.close()

    copy_result_dir = workdir + "/" + type + "/" + method_name + "_result/" + data_type + "/round" + str(round) + "/"
    if (os.path.exists(copy_result_dir ) == False):
        os.makedirs(copy_result_dir )

    if (os.path.exists(copy_result_dir  + "/result" + str(index))):
        os.system("rm -rf " + copy_result_dir  + "/result" + str(index))

    os.system("cp -r " + resultdir + " " + copy_result_dir  + "/result" + str(index))


def post_deal2(workdir, method_name, index, type, round):  # get measures results by T

    roc_dir = workdir + "/" + type + "/" + method_name + "_roc/"
    rocfile1 = roc_dir + "/roc_evaluate_" + str(index) + "_" + str(round) + ".roc"
    rocfile2 = roc_dir + "/roc_test_" + str(index) + "_" + str(round) + ".roc"

    measures_list = gf.get_measures_files(rocfile1, rocfile2)

    record_file = workdir + "/" + type + "/record"
    f = open(record_file, "a")
    line = type + ":" + method_name + ":  "
    for measures in measures_list:
        line = line + measures + " "
    f.write(line + "\n")
    f.write("\n")
    f.write("\n")

    f.flush()
    f.close()



def calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, data_type, method_name):  # save cross entropy results

    term_file = workdir + "/" + type + "/term_list"
    term_list = read_term_list(term_file)

    resultdir = workdir + "/" + type + "/result/"
    os.system("rm -rf "+resultdir)
    os.makedirs(resultdir)

    for i in range(len(test_name_list)):

        name = test_name_list[i]

        if(os.path.exists(resultdir + "/" + name + "/")==False):
            os.makedirs(resultdir + "/" + name)

        result_file = resultdir + "/" + name + "/" + method_name + "_" + type

        f = open(result_file, "w")

        for j in range(len(term_list)):

            if(test_predict_matrix[i][j]>=0.05):
                f.write(term_list[j] + " " + type[1:] + " " + str(test_predict_matrix[i][j]) + "\n")
        f.flush()
        f.close()

    post_deal1(workdir, type, method_name, data_type, index, round)


def write_record(workdir, type, index): # write index

    record_file = workdir + "/" + type + "/record"
    f = open(record_file, "a")
    f.write("The " + str(index) + "-th iteration:\n")
    f.flush()
    f.close()


def calculate_pred_label(workdir, type, evaluate_name_list, evaluate_predict_matrix, test_name_list, test_predict_matrix, round, index):  # save cross entropy results process

    write_record(workdir, type, index)

    method_name = "cross_entropy"

    calculate_pred_label_one(workdir, type, evaluate_name_list, evaluate_predict_matrix, round, index, "evaluate", method_name)
    calculate_pred_label_one(workdir, type, test_name_list, test_predict_matrix, round, index, "test", method_name)

    post_deal2(workdir, method_name, index, type, round)


def read_result(file): # read results

    f = open(file, "r")
    text = f.read()
    f.close()
    name_list = []

    result_dict = dict()
    for line in text.splitlines():
        values = line.strip().split()
        result_dict[values[0]] = float(values[2])
        name_list.append(values[0])

    return result_dict, name_list

def combine_result(file1, file2, file3, weight, type):  # combine results

    result_dict1, name_list1 = read_result(file1)
    result_dict2, name_list2 = read_result(file2)

    result_dict3 = dict()

    name_list1.extend(name_list2)
    name_list= list(set(name_list1))

    for name in name_list:

        if (name in result_dict1):
            value1 = result_dict1[name]
        else:
            value1 = 0

        if(name in result_dict2):
            value2 = result_dict2[name]
        else:
            value2 = 0

        value = weight * value1 + (1-weight) * value2
        result_dict3[name] = value

    f = open(file3, "w")
    for name in result_dict3:
        if(result_dict3[name]>=0.01):
            f.write(name + " " + type[1:] + " " + str(result_dict3[name]) + "\n")
    f.flush()
    f.close()


def combine_one(workdir, type, index, round, weight, method_name1, method_name2, method_name3, data_type):

    resultdir1 = workdir + "/" + type + "/" + method_name1 + "_result/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"
    resultdir2 = workdir + "/" + type + "/" + method_name2 + "_result/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"

    resultdir3 = workdir + "/" + type + "/result/"
    os.system("rm -rf "+ resultdir3)
    os.makedirs(resultdir3)

    name_list = os.listdir(resultdir1)

    for name in name_list:

        file1 = resultdir1 + "/" + name + "/" + method_name1 + "_" + type
        file2 = resultdir2 + "/" + name + "/" + method_name2 + "_" + type
        file3 = resultdir3 + "/" + name + "/" + method_name3 + "_" + type

        if(os.path.exists(resultdir3 + "/" + name + "/")==False):
            os.makedirs(resultdir3 + "/" + name + "/")

        combine_result(file1, file2, file3, weight, type)

    post_deal1(workdir, type, method_name3, data_type, index, round)


def combine(workdir, type, index, round, weight):

    write_record(workdir, type, index)

    method_name1 = "triplet"
    method_name2 = "cross_entropy"
    method_name3 = "combine"

    combine_one(workdir, type, index, round, weight, method_name1, method_name2, method_name3, "evaluate")
    combine_one(workdir, type, index, round, weight, method_name1, method_name2, method_name3, "test")
    post_deal2(workdir, method_name3, index, type, round)


















