import os.path
import sys

from common_methods import read_name_list_from_sequence
sequence_info_dir = "/data/yihengzhu/GOA/resource/sequence_info/"

def read_date_from_information_file(information_file):

    f = open(information_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    return line_set[len(line_set)-2]


def copy_gene_sequence(test_sequence_file, origindir, copydir):

    test_name_list = read_name_list_from_sequence(test_sequence_file)
    for name in test_name_list:
        seq_info_file = sequence_info_dir + "/" + name
        gene_id = read_date_from_information_file(seq_info_file)

        if(gene_id!="None"):
            gene_sequence_file = os.path.join(origindir, gene_id + ".fasta")
            if(os.path.exists(gene_sequence_file)):
                copy_file = os.path.join(copydir, name + ".fasta")
                os.system("cp " + gene_sequence_file + " " + copy_file)

def check_different(sequence_dir1, sequence_dir2):

    name_list1 = os.listdir(sequence_dir1)
    name_list2 = os.listdir(sequence_dir2)

    name_list = list(set(name_list1) & set(name_list2))
    count = 0

    for name in name_list:
        f = open(sequence_dir1 + "/" + name, "r")
        text1 = f.read()
        f.close()

        f = open(sequence_dir2 + "/" + name, "r")
        text2 = f.read()
        f.close()

        if(text1!=text2):
            print(name)
            count = count + 1

    print(count)
    print(len(name_list))

check_different(sys.argv[1], sys.argv[2])



