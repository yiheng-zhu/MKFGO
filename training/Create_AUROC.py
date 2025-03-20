import sys
import os
from sklearn import metrics

def read(labelfile):  # read label

    label_dict = dict()
    f=open(labelfile,"rU")
    line_txt = f.read()
    f.close()

    all_term_list = []
    name_list = []

    for line in line_txt.splitlines():
        values = line.strip().split()
        term_list = values[1].split(",")
        label_dict[values[0]] = term_list
        all_term_list.extend(term_list)
        name_list.append(values[0])

    all_term_list = list(set(all_term_list))

    return label_dict, name_list, all_term_list

def read_result(resultfile): #read results

    result_dict = dict()
    if (os.path.exists(resultfile) == False):
        return result_dict

    f=open(resultfile,"rU")
    line_txt = f.read()
    f.close()

    for line in line_txt.splitlines():
        result_dict[line.split()[0]] = line.split()[2]

    return result_dict

def get_result(resultdir, resultfile_name):  # read result

    result_list_dict = dict()
    list_dir = os.listdir(resultdir)

    for name in list_dir:
        result_list_dict[name] = read_result(resultdir + "/" + name + "/" + resultfile_name)

    return result_list_dict

def create_auc(label_dict, result_dict, name_list, all_term_list):

    number = len(name_list)
    average_auc = 0
    for term in all_term_list:

        label_array = [0 for i in range(number)]
        score_array = [0 for i in range(number)]

        for i in range(number):

            if(term in label_dict[name_list[i]]):
                label_array[i] = 1

            if(name_list[i] in result_dict and term in result_dict[name_list[i]]):
                score_array[i] = float(result_dict[name_list[i]][term])


        fpr, tpr, thresholds = metrics.roc_curve(label_array, score_array, pos_label=1)
        auc =  metrics.auc(fpr, tpr)
        average_auc = average_auc + auc

    average_auc = average_auc/len(all_term_list)

    return average_auc







if __name__=="__main__":

    '''
    workdir = sys.argv[1]
    type_list = ["MF", "BP", "CC"]
    data_type_list = ["evaluate", "test"]



    for type in type_list:
        resultfile = "final_cross_entropy_" + type + "_new"
        for data_type in data_type_list:

            resultdir = workdir + "/" + type + "/" + data_type + "/"
            label_file = workdir + "/" + type + "/" + data_type + "_gene_label"
            label_dict, name_list, all_term_list = read(label_file)
            result_list_dict = get_result(resultdir, resultfile)
            average_auc = create_auc(label_dict, result_list_dict, name_list, all_term_list)
            print(average_auc)
    '''
    label_dict, name_list, all_term_list = read(sys.argv[1])
    result_list_dict = get_result(sys.argv[2], sys.argv[3])
    average_auc = create_auc(label_dict, result_list_dict, name_list, all_term_list)
    print(average_auc)

