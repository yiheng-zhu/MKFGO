import download_gene_sequence as dg
import sys
import os
from config import *
from common_methods import create_dir, write_list
import subprocess
import numpy as np
from decimal import Decimal
from load_glm_model import MLP
import save_result as sh

def create_features(workdir):

    gene_sequence_dir = os.path.join(workdir, glm_gene_sequence_dir)
    human_feature_dir = os.path.join(workdir, glm_human_feature_dir)
    multi_species_feature_dir = os.path.join(workdir, glm_multi_species_feature_dir)
    create_dir(human_feature_dir)
    create_dir(multi_species_feature_dir)

    python_interpreter = os.path.join(glm_env_python_dir, "python")
    cmd = python_interpreter + " " + glm_nctrans_script_file + " " + gene_sequence_dir + " " + human_feature_dir + " " + glm_human_model_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running NCTrans-Human for feature extraction: {e}")

    python_interpreter = os.path.join(glm_env_python_dir, "python")
    cmd = python_interpreter + " " + glm_nctrans_script_file + " " + gene_sequence_dir + " " + multi_species_feature_dir + " " + glm_multi_species_model_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running NCTrans-Multi-Species for feature extraction: {e}")

def create_dataset(workdir):

    for type in go_type_list:

        test_data_dir = os.path.join(workdir, glm_workspace, type)
        create_dir(test_data_dir)

        gene_sequence_dir = os.path.join(workdir, glm_gene_sequence_dir)
        sub_name_list = os.listdir(gene_sequence_dir)
        test_name_list = [name.split(".")[0] for name in sub_name_list]
        test_name_file = os.path.join(test_data_dir, glm_test_name)
        write_list(test_name_list, test_name_file)

        test_label_onehot_file = os.path.join(test_data_dir, glm_test_label_onehot)
        test_label_onehot = np.zeros([len(test_name_list), go_label_size_small[type]])
        np.savetxt(test_label_onehot_file, test_label_onehot, fmt="%d")

        data_file = os.path.join(test_data_dir, glm_test_feature)

        f = open(data_file, "w")
        for name in test_name_list:

            feature_file = os.path.join(workdir, glm_multi_species_feature_dir, name + ".npy")
            feature_array = np.load(feature_file)
            line = ""
            for value in feature_array:
                line = line + str(Decimal(float(value)).quantize(Decimal("0.000000"))) + " "

            feature_file = os.path.join(workdir, glm_human_feature_dir, name + ".npy")
            feature_array = np.load(feature_file)
            for value in feature_array:
                line = line + str(Decimal(float(value)).quantize(Decimal("0.000000"))) + " "
            line = line.strip()
            f.write(line + "\n")
        f.close()

def prediction(workdir):

    for go_type in go_type_list:
        for round_index in range(1, all_model_number + 1):
            sub_workdir = os.path.join(workdir, glm_workspace)
            mlp = MLP(sub_workdir, go_type, round_index)
            mlp.running()

def create_average_result(workdir):

    for go_type in go_type_list:
        sub_workdir = os.path.join(workdir, glm_workspace)
        sh.create_average_cross_entropy_results(sub_workdir, go_type)


def main_process(workdir):

    print("------------Feature extraction start !------------")
    create_features(workdir)
    print("------------Feature extraction end !------------")
    print()

    print("------------Dataset creating start !------------")
    create_dataset(workdir)
    print("------------Dataset creating end !------------")
    print()

    print("------------GLM prediction start !------------")
    prediction(workdir)
    create_average_result(workdir)
    print("------------GLM prediction end !------------")
    print()


if __name__ == "__main__":

    main_process(sys.argv[1])
