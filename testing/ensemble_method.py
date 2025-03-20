import os
import subprocess
from config import *
from common_methods import read_name_list_from_sequence, read_result, create_dir, check_file_not_exist, read_ensemble_datafile, write_list
import sys
import joblib
import save_result as sh

def create_dataset(workdir, go_type, ensemble_method_list, ensemble_workspace, ensemble_resultdir_list, ensemble_postfix_list):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)
    test_name_list  = read_name_list_from_sequence(test_sequence_file)
    test_name_file = os.path.join(workdir, ensemble_workspace, go_type, ensemble_test_name)

    sub_workdir = os.path.join(workdir, ensemble_workspace, go_type)
    test_data_dir = os.path.join(sub_workdir, ensemble_test_feature)
    create_dir(test_data_dir)

    write_list(test_name_list, test_name_file)

    for name in test_name_list:

        term_list = []
        all_score_dict = []
        for i in range(len(ensemble_method_list)):

            if(ensemble_method_list[i]==ppi_workspace):
                result_file = os.path.join(workdir, ensemble_method_list[i], ensemble_resultdir_list[i], name, ensemble_postfix_list[i] + "_" + go_type + "_new")
            else:
                result_file = os.path.join(workdir, ensemble_method_list[i], go_type, ensemble_resultdir_list[i], name, ensemble_postfix_list[i] + "_" + go_type + "_new")
            score_dict = read_result(result_file)
            term_list.extend(score_dict.keys())

            all_score_dict.append(score_dict)


        term_list = list(set(term_list))

        test_data_file = os.path.join(test_data_dir, name)
        f = open(test_data_file, "w")

        for term in term_list:
            temp_line = term + " "

            for i in range(len(ensemble_method_list)):
                if(term in all_score_dict[i]):
                    temp_line = temp_line + str(float(all_score_dict[i][term]) * ensemble_weight_dict[ensemble_method_list[i]][go_type]) + " "
                else:
                    temp_line = temp_line + "0.0 "
            f.write(temp_line.strip() + "\n")

        f.close()

def prediction(workdir, go_type, round_index, ensemble_workspace):

    test_data_dir = os.path.join(workdir, ensemble_workspace, go_type, ensemble_test_feature)
    name_list = os.listdir(test_data_dir)

    model_file = os.path.join(all_data_dir, ensemble_workspace, go_type, ensemble_modeldir, str(round_index), ensemble_modelname)
    clf = joblib.load(model_file)

    result_dir = os.path.join(workdir, ensemble_workspace, go_type, ensemble_result_dir, ensemble_round_name + str(round_index))

    for name in name_list:

        test_data_file = os.path.join(test_data_dir, name)
        if(check_file_not_exist(test_data_file)):
            continue

        feature_array, term_array = read_ensemble_datafile(test_data_file)
        pro_array = clf.predict_proba(feature_array)

        sub_result_dir = os.path.join(result_dir, name)
        create_dir(sub_result_dir)

        result_file = os.path.join(sub_result_dir, ensemble_result_name + "_" + go_type)

        f = open(result_file, "w")
        for i in range(pro_array.shape[0]):
            if (pro_array[i][1] >= min_go_prob):
                f.write(term_array[i] + " " + go_type[1:] + " " + str(pro_array[i][1]) + "\n")
        f.close()

    cmd = "python2 " + post_deal_script_file + " " + result_dir + " " + go_type + " " + ensemble_result_name
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while post-dealing go terms {e}")


def create_average_result(workdir, ensemble_workspace):

    for go_type in go_type_list:
        sub_workdir = os.path.join(workdir, ensemble_workspace)
        sh.create_average_cross_entropy_results(sub_workdir, go_type)



def main_process(workdir, is_dlmgo):


    if(is_dlmgo):
        ensemble_method_list = [plm_workspace, hc_workspace, ppi_workspace, glm_workspace, naive_workspace]
        ensemble_workspace = "ensemble"
        ensemble_resultdir_list = [plm_final_result_dir, hc_final_combine_dir, "", glm_final_result_dir, ""]
        ensemble_postfix_list = [plm_final_result_name, hc_final_combine_name, ppi_result_name, glm_final_result_name, naive_method_name]
    else:
        ensemble_method_list = [plm_workspace, hc_workspace, ppi_workspace, naive_workspace]
        ensemble_workspace = "ensemble_withoutdlmgo"
        ensemble_resultdir_list = [plm_final_result_dir, hc_final_combine_dir, "", ""]
        ensemble_postfix_list = [plm_final_result_name, hc_final_combine_name, ppi_result_name, naive_method_name]

    print("------------Test dataset generation start !------------")
    for go_type in go_type_list:
        create_dataset(workdir, go_type, ensemble_method_list, ensemble_workspace, ensemble_resultdir_list, ensemble_postfix_list)
    print("------------Test dataset generation end !------------")
    print()
    print("------------ensemble prediction start !------------")
    for go_type in go_type_list:
        for round_index in range(1, all_model_number + 1):
            prediction(workdir, go_type, round_index, ensemble_workspace)
    create_average_result(workdir, ensemble_workspace)
    print("------------ensemble prediction end !------------")





if __name__ == "__main__":

    main_process(sys.argv[1], eval(sys.argv[2]))