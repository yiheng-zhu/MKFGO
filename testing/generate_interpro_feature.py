import subprocess
import sys
import os
import numpy as np

from config import interproscan_script_file,\
    interproscan_workspace, \
    test_sequence_filename, \
    interproscan_result_filename, \
    interproscan_temp_filedir, \
    interproscan_entry_list_file, \
    interproscan_resultdir, \
    interproscan_featuredir


from common_methods import check_file_not_exist, create_dir

def run_interpro(workdir): # run interproscan

    seq_file = os.path.join(workdir, test_sequence_filename)
    result_file = os.path.join(workdir, interproscan_workspace, interproscan_result_filename)
    temp_dir = os.path.join(workdir, interproscan_workspace, interproscan_temp_filedir)

    try:
        cmd = interproscan_script_file + " -i " + seq_file + " -f tsv -o " + result_file + " -T " + temp_dir
        subprocess.run(cmd, shell=True, capture_output=True, text=True)
        os.system("rm -rf " + temp_dir)

    except Exception as e:
        print("Generate InterPro Feature Failed!")
        print(e)
        exit(1)

def read_result_file(result_file):  # read result.file

    f = open(result_file, "r")
    text = f.read()
    f.close()

    result_dict = dict()

    for line in text.splitlines():
        values = line.split("	")
        name = values[0]
        entry = values[11]

        if(name not in result_dict):
            result_dict[name] = []

        result_dict[name].append(entry)

    for name in result_dict.keys():

        temp_list = sorted(list(set(result_dict[name])))
        if("-" in temp_list):
            temp_list.remove("-")

        result_dict[name] = temp_list

    return result_dict


def read_all_entry_list(entry_list_file):  # read all entry list

    f = open(entry_list_file, "r")
    text = f.read()
    f.close()

    all_entry_list = []
    line_set = text.splitlines()
    line_set = line_set[1:]

    for line in line_set:
        all_entry_list.append(line.split("	")[0])

    return all_entry_list

def create_entry_array(current_entry_list, all_entry_list): # create entry index

    entry_number = len(all_entry_list)
    entry_array = np.array([0 for i in range(entry_number)])
    for entry in current_entry_list:
        index = all_entry_list.index(entry)
        entry_array[index] = 1

    return entry_array

def write_file(array_list, save_file): # write

    f = open(save_file, "w")
    for array in array_list:
        f.write(str(array) + "\n")
    f.close()

def create_interpro_features(workdir): # create features

    result_file = os.path.join(workdir, interproscan_workspace, interproscan_result_filename)
    if (check_file_not_exist(result_file)): return

    result_dir = os.path.join(workdir, interproscan_workspace, interproscan_resultdir)
    feature_dir = os.path.join(workdir, interproscan_workspace, interproscan_featuredir)
    if(os.path.exists(result_dir)==False): os.makedirs(result_dir)
    if (os.path.exists(feature_dir) == False):os.makedirs(feature_dir)

    result_dict = read_result_file(result_file)
    all_entry_list = read_all_entry_list(interproscan_entry_list_file)

    for name in result_dict.keys():

        entry_name_file = os.path.join(result_dir, name)
        entry_array_file = os.path.join(feature_dir, name + ".npy")

        entry_name_list = result_dict[name]
        entry_array = create_entry_array(entry_name_list, all_entry_list)

        write_file(entry_name_list, entry_name_file)
        np.save(entry_array_file, entry_array)

def check_features(workdir): # generate features for the proteins which cannot hit the templates in InterPro database

    seq_file = os.path.join(workdir, test_sequence_filename)
    result_dir = os.path.join(workdir, interproscan_workspace, interproscan_resultdir)
    feature_dir = os.path.join(workdir, interproscan_workspace, interproscan_featuredir)

    f = open(seq_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():

        line = line.strip()
        if(line.startswith(">")):

            name = line[1:]

            entry_name_file = os.path.join(result_dir, name)
            entry_array_file = os.path.join(feature_dir, name) + ".npy"

            if(check_file_not_exist(entry_name_file)):

                f = open(entry_name_file, "w")
                f.close()

                all_entry_list = read_all_entry_list(interproscan_entry_list_file)
                entry_number = len(all_entry_list)

                entry_array = np.array([0 for i in range(entry_number)])
                np.save(entry_array_file, entry_array)



def main_process(workdir):

    create_dir(os.path.join(workdir, interproscan_workspace))
    run_interpro(workdir)
    create_interpro_features(workdir)
    check_features(workdir)

if __name__ == '__main__':

    main_process(sys.argv[1])

