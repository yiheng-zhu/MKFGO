import os
import Get_Meatures_From_T as gf
import sys

pythondir = "/data/yihengzhu/GOA/resource/benchmark/pythonfile/"


def evaluate_result_one(workdir, type, index, record_file, round, data_type, method_name, number):

    # remove result dir
    resultdir = workdir + "/Predict_Label/" + type + "/Label/"
    os.system("rm -rf "+ resultdir)
    os.makedirs(resultdir)

    # copy results
    target_dir = workdir + "/" + type + "/resultset/" + data_type + "/round" + str(round) + "/result" + str(index) + "/"
    os.system("cp " + target_dir + "/* " + resultdir + "/")

    # create roc dir
    roc_dir = workdir + "/" + type + "/"+method_name+"_roc/"
    if (os.path.exists(roc_dir) == False):
        os.makedirs(roc_dir)
    rocfile = roc_dir + "/roc_" + data_type + "_" + str(index) + "_" + str(round) + ".roc"


    test_gene_file = workdir + "/" + type + "/" + data_type + "_gene_list"
    test_label_file = workdir + "/" + type + "/" + data_type + "_gene_label"

    temp_result_dir = workdir + "/" + type + "/result/"
    os.system("rm -rf " + temp_result_dir)

    # create triplet results
    os.system("python2 " +pythondir + "/Run_pipelines_rank.py " + workdir + " " + test_gene_file + " " + str(number) + " " + type + " " + method_name)

    # evaluate triplet results
    r = os.popen("python2 " + pythondir + "/Evaluate_pipelines.py " + workdir + " "+ test_label_file+" " + type+" " + rocfile + " " + method_name)

    # get triplet results
    info = r.readlines()
    sum_line = ""
    for line in info:
        line = line.strip("\r\n")
        sum_line = sum_line + line

    # write triplet results
    f = open(record_file, "a")
    f.write(sum_line + "\n\n")
    f.flush()
    f.close()

    # copy triplet results
    triplet_result_dir = workdir+"/"+type+"/"+method_name+"_result/"+data_type+"/round"+str(round)+"/"
    if(os.path.exists(triplet_result_dir)==False):
        os.makedirs(triplet_result_dir)

    if(os.path.exists(triplet_result_dir + "/result" + str(index))):
        os.system("rm -rf " + triplet_result_dir + "/result" + str(index))

    os.system("cp -r "+ temp_result_dir +" " + triplet_result_dir + "/result" + str(index))


def evaluate_result(workdir, type, index, round):

    record_file = workdir + "/" +type + "/record1"
    f = open(record_file, "a")
    f.write("The " + str(index) + "-th iteration:\n")
    f.flush()
    f.close()

    method_name = "triplet"
    number = 30

    evaluate_result_one(workdir, type, index, record_file, round, "evaluate", method_name, number)
    evaluate_result_one(workdir, type, index, record_file, round, "test", method_name, number)

    roc_dir = workdir + "/" + type + "/" + method_name + "_roc/"
    rocfile1 = roc_dir + "/roc_evaluate_" + str(index) + "_" + str(round) + ".roc"
    rocfile2 = roc_dir + "/roc_test_" + str(index) + "_" + str(round) + ".roc"

    measures_list = gf.get_measures_files(rocfile1, rocfile2)
    f = open(record_file, "a")
    line = type + ":" + method_name + ":  "
    for measures in measures_list:
        line = line + measures + " "
    f.write(line + "\n")
    f.write("\n")
    f.write("\n")

    f.flush()
    f.close()

if __name__=="__main__":

    workdir = sys.argv[1]
    type = sys.argv[2]
    index = 630
    for round in range(1, 11):
         evaluate_result(workdir, type, index, round)













