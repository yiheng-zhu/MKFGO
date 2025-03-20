import os
import sys
import subprocess
import torch
import numpy as np
from common_methods import create_dir, split_sequence, read_sequence, extract_name_list
from config import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = device_type_dict[device_type]["device_id"]

def init(workdir):

    seq_dir = os.path.join(workdir, ss_workspace, ss_seq_dir)
    esm1b_feature_dir = os.path.join(workdir, ss_workspace, ss_esm1b_feature_dir)
    prottrans_feature_dir = os.path.join(workdir, ss_workspace, ss_prottrans_feature_dir)
    result_dir = os.path.join(workdir, ss_workspace, ss_result_dir)
    array_dir = os.path.join(workdir, ss_workspace, ss_array_dir)

    create_dir(seq_dir)
    create_dir(esm1b_feature_dir)
    create_dir(prottrans_feature_dir)
    create_dir(result_dir)
    create_dir(array_dir)

    name_file = os.path.join(workdir, ss_workspace, ss_name_file)
    seq_file = os.path.join(workdir, test_sequence_filename)

    split_sequence(seq_file, seq_dir)
    extract_name_list(seq_file, name_file)

    return name_file, seq_file, seq_dir, esm1b_feature_dir, prottrans_feature_dir, result_dir, array_dir

def generate_esm1b_features(seq_file, feature_dir):

    python_interpreter = os.path.join(ss_esm_env_python_dir, "python")
    esm1b_cmd = python_interpreter + " " + \
                ss_esm_fe_script_file + " " + \
                ss_esm1b_model_name + " " + \
                seq_file + " " + \
                feature_dir + \
                " --repr_layers " + str(ss_esm1b_model_layer) + \
                " --include " + ss_esm1b_token_type + \
                " --truncate"

    try:
        subprocess.run(esm1b_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running ESM-1b for feature extraction: {e}")

    sequence_dict = read_sequence(seq_file)

    name_list = os.listdir(feature_dir)
    for name in name_list:

        origin_feature_file = os.path.join(feature_dir, name)
        deal_feature_file = os.path.join(feature_dir, name.split(".")[0] + ".npy")

        feature = torch.load(origin_feature_file)
        feature = feature["representations"][ss_esm1b_model_layer].numpy()
        feature = feature.astype(np.float64)

        seq_len = len(sequence_dict[name.split(".")[0]])
        if(feature.shape[0] != seq_len):
            padding_matrix = np.zeros([seq_len - feature.shape[0], feature.shape[1]])
            feature = np.concatenate([feature, padding_matrix], axis=0)

        np.save(deal_feature_file, feature)

def generate_prottrans_features(seq_file, feature_dir):

    python_interpreter = os.path.join(ss_prottrans_env_python_dir, "python")
    prottrans_cmd = python_interpreter + " " + \
                    ss_prottrans_fe_script_file + \
                    " -i " + seq_file + \
                    " -o " + feature_dir

    try:
        subprocess.run(prottrans_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running ProTtrans for feature extraction: {e}")

    sequence_dict = read_sequence(seq_file)

    name_list = os.listdir(feature_dir)
    for name in name_list:

        origin_feature_file = os.path.join(feature_dir, name)
        deal_feature_file = os.path.join(feature_dir, name.split(".")[0] + ".npy")

        feature = torch.load(origin_feature_file)

        seq_len = len(sequence_dict[name.split(".")[0]])
        if(feature.shape[0] != seq_len):
            padding_matrix = np.zeros([seq_len - feature.shape[0], feature.shape[1]])
            feature = np.concatenate([feature, padding_matrix], axis=0)
            print(name)

        np.save(deal_feature_file, feature)

def run_ss_pred(name_file, seq_dir, esm1b_feature_dir, prottrans_feature_dir, result_dir):

    ss_pred_cmd = "python " + \
                  ss_pred_script_file + \
                  " --name_list " + name_file + \
                  "  --seq_path " + seq_dir + \
                  " --esm1b_path " + esm1b_feature_dir + \
                  " --prottrans_path " + prottrans_feature_dir + \
                  " --model_path " + ss_pred_model_dir + \
                  " --save_path " + result_dir + \
                  " --device " + device_type_dict[device_type]["device"]

    try:
        subprocess.run(ss_pred_cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running SSPred: {e}")

def create_single_ss_array(pss_file, array_file):

    f = open(pss_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()
    pss_array = np.array([0 for i in range(sequence_cut_off)])
    for i in range(1, len(line_set)):
        values = line_set[i].strip().split(",")
        ss_name = values[3]
        if (ss_name in ss_type):
            ss_index = ss_type.index(ss_name) + 1
        else:
            ss_index = ss_type_number + 1

        pss_array[i - 1] = ss_index

    np.save(array_file, pss_array)

def create_ss_array(result_dir, array_dir):

    name_list = os.listdir(result_dir)

    for name in name_list:
        pss_file = os.path.join(result_dir, name)
        array_file = os.path.join(array_dir, name.split(".")[0] + ".npy")
        create_single_ss_array(pss_file, array_file)








def main_process(workdir):

    name_file, seq_file, seq_dir, esm1b_feature_dir, prottrans_feature_dir, result_dir, array_dir = init(workdir)
    generate_esm1b_features(seq_file, esm1b_feature_dir)
    generate_prottrans_features(seq_file, prottrans_feature_dir)
    run_ss_pred(name_file, seq_dir, esm1b_feature_dir, prottrans_feature_dir, result_dir)
    create_ss_array(result_dir, array_dir)

if __name__ == "__main__":

    main_process(sys.argv[1])

