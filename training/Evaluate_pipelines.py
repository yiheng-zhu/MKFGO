import sys
import Evaluation as ev
import Get_Meatures_From_T as gf
import os

def evaluate(labelfile, result_dir, rocfile, type, pipeline_list):

    print(type + ":")

    for pipeline in pipeline_list:

        e = ev.evaluation(labelfile, result_dir, pipeline + "_" + type + "_new", rocfile)
        e.process()
        train_aupr = e.get_aupr()

        measures_list = gf.get_measures_files(rocfile, rocfile)
        line = ""
        for measures in measures_list:
            line = line + measures + " "
        line = line + "AUPR=" + str(train_aupr) + " "
        print(pipeline + ":  " + line)
        print("\n")
        print("\n")

if __name__=="__main__":

    dir = sys.argv[1]
    label_file = sys.argv[2]
    type_list = [sys.argv[3]]
    rocfile = sys.argv[4]

    pipeline_list = [sys.argv[5]]
    

    for type in type_list:

        result_dir = dir + "/" + type + "/result/"

        evaluate(label_file, result_dir, rocfile, type, pipeline_list)


