import os.path
import sys

from common_methods import read_name_list_from_sequence, create_dir
from config import go_type_list, naive_method_name, naive_workspace, test_sequence_filename, hc_data_dir, naive_probability_dir

def main_process(workdir):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)
    test_name_list = read_name_list_from_sequence(test_sequence_file)

    for go_type in go_type_list:
        for name in test_name_list:
            sub_result_dir = os.path.join(workdir, naive_workspace, go_type, name)
            create_dir(sub_result_dir)
            origin_file = os.path.join(naive_probability_dir, naive_method_name + "_" + go_type)
            copy_file = os.path.join(sub_result_dir, naive_method_name + "_" + go_type + "_new")
            os.system("cp " + origin_file + " " + copy_file)


if __name__ == "__main__":

    main_process(sys.argv[1])




