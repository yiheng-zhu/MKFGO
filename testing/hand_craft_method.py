import os.path

import generate_pssm_feature as gpf
import generate_ss_feature as gsf
import generate_interpro_feature as gif
import numpy as np
import sys

from config import *
from common_methods import create_dir, extract_name_list, read_name_list, create_sequence_length
from load_hand_craft_model import LSTM
import save_result as sh

def create_data_file(workdir):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)


    for type in go_type_list:

        test_data_dir = os.path.join(workdir, hc_workspace, type)
        create_dir(test_data_dir)

        test_name_file = os.path.join(test_data_dir, hc_test_name)
        extract_name_list(test_sequence_file, test_name_file)
        test_name_list = read_name_list(test_name_file)

        test_label_onehot_file = os.path.join(test_data_dir, hc_test_label_onehot)
        test_label_onehot = np.zeros([len(test_name_list), go_label_size[type]])
        np.savetxt(test_label_onehot_file, test_label_onehot, fmt="%d")

        test_sequence_length_file = os.path.join(test_data_dir, hc_test_seq_length)
        create_sequence_length(test_sequence_file, test_sequence_length_file)

def prediction(workdir):

    for go_type in go_type_list:
        for round_index in range(1, all_model_number + 1):
            lstm = LSTM(workdir, go_type, round_index)
            lstm.running()

def create_average_result(workdir):

    for go_type in go_type_list:
        sub_workdir = os.path.join(workdir, hc_workspace)
        sh.create_average_cross_entropy_results(sub_workdir, go_type)
        sh.create_average_distance(sub_workdir, go_type)
        sh.create_triplet_results(sub_workdir, go_type)
        sh.combine_results(sub_workdir, go_type)


def main_process(workdir):


    print("------------Hand craft method start !------------")
    print()

    print("------------PSSM feature generation start !------------")
    gpf.main_process(workdir)
    print("------------PSSM feature generation end !------------")
    print()

    print("------------SS feature generation start !------------")
    gsf.main_process(workdir)
    print("------------SS feature generation end !------------")
    print()

    print("------------InterPro feature generation start !------------")
    gif.main_process(workdir)
    print("------------InterPro feature generation end !------------")
    print()

    print("------------Hand craft method end !------------")
    print()

    print("------------Test dataset generation start !------------")
    create_data_file(workdir)
    print("------------Test dataset generation end !------------")
    print()
    print("------------Hand craft prediction start !------------")
    prediction(workdir)
    create_average_result(workdir)
    print("------------Hand craft prediction end !------------")
    print()







if __name__ == "__main__":

    main_process(sys.argv[1])