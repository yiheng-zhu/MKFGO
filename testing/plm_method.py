import os
from config import *
from common_methods import *
import numpy as np
from decimal import Decimal
from load_plm_model import MLP
import save_result as sh


def create_data_file(workdir):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)


    for type in go_type_list:

        test_data_dir = os.path.join(workdir, plm_workspace, type)
        create_dir(test_data_dir)

        test_name_file = os.path.join(test_data_dir, plm_test_name)
        extract_name_list(test_sequence_file, test_name_file)
        test_name_list = read_name_list(test_name_file)

        test_label_onehot_file = os.path.join(test_data_dir, plm_test_label_onehot)
        test_label_onehot = np.zeros([len(test_name_list), go_label_size[type]])
        np.savetxt(test_label_onehot_file, test_label_onehot, fmt="%d")

        data_file = os.path.join(test_data_dir, plm_test_feature)
        f = open(data_file, "w")

        for name in test_name_list:
            feature_file = os.path.join(workdir, ss_workspace, ss_prottrans_feature_dir, name + ".npy")
            feature = np.load(feature_file)
            mean_feature = np.mean(feature, axis = 0)
            line = ""
            for value in mean_feature:
                line = line + str(Decimal(float(value)).quantize(Decimal("0.000000"))) + " "
            line = line.strip()
            f.write(line + "\n")
        f.close()

def prediction(workdir):

    for go_type in go_type_list:
        for round_index in range(1, all_model_number + 1):
            sub_workdir = os.path.join(workdir, plm_workspace)
            mlp = MLP(sub_workdir, go_type, round_index)
            mlp.running()

def create_average_result(workdir):

    for go_type in go_type_list:
        sub_workdir = os.path.join(workdir, plm_workspace)
        sh.create_average_cross_entropy_results(sub_workdir, go_type)

def main_process(workdir):

    print("------------Test dataset generation start !------------")
    create_data_file(workdir)
    print("------------Test dataset generation end !------------")
    print()
    print("------------PLM prediction start !------------")
    prediction(workdir)
    create_average_result(workdir)
    print("------------PLM prediction end !------------")
    print()



if __name__ == "__main__":

    main_process(sys.argv[1])






