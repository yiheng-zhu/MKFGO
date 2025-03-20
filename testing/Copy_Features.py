import os
import sys

def read_name_list(sequence_file):

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    name_list = []

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name_list.append(line[1:])

    return name_list

def create_length_file(origin_file, deal_file, name_list):

    f = open(origin_file, "r")
    text = f.read()
    f.close()

    f = open(deal_file, "w")
    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        if(values[0] in name_list):
            f.write(line + "\n")
    f.close()

def copy_features(origin_dir, copy_dir, name_list, postfix):

    for name in name_list:
        os.system("cp " + origin_dir + "/" + name + "." + postfix + " " + copy_dir + "/" + name + "." + postfix)


def main_process(origindir, copydir, sequence_file):

    name_list = read_name_list(sequence_file)
    create_length_file(origindir + "/all_protein_length", copydir + "/protein_length", name_list)
    copy_features(origindir + "/pssm_array/", copydir + "/pssm_array/", name_list, "npy")
    copy_features(origindir + "/ss_index/", copydir + "/ss_index/", name_list, "npy")
    copy_features("/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/temps/entry_array/", copydir + "/entry_array/", name_list, "npy")

main_process(sys.argv[1], sys.argv[2], sys.argv[3])


