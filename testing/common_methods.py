import os
import sys
import numpy as np
from config import sequence_cut_off, ppi_seq_name


def check_file_not_exist(file): # check file

    if (os.path.exists(file) == False or os.path.getsize(file) == 0):
        print(file + " is not exist")
        return True

    return False

def create_dir(temp_dir): # create dir

    os.system("rm -rf " + temp_dir)
    os.makedirs(temp_dir)


def read_sequence(sequence_file):  # read sequence

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        line = line.strip()
        if (line.startswith(">")):
            name = line[1:]
        else:
            sequence_dict[name] = line

    return sequence_dict


def find_longest_sequence(sequence_file):  # find the protein with the longest sequence

    sequence_dict = dict()

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    max_len = 0
    min_len = 100000000

    max_sequence_name = ""
    min_sequence_name = ""

    for line in text.splitlines():
        line = line.strip()
        if (line.startswith(">")):
            name = line
        else:
            sequence_dict[name] = line

            if(len(line)>max_len):
                max_len = len(line)
                max_sequence_name = name

            if(len(line)<min_len):
                min_len = len(line)
                min_sequence_name = name

    print(max_sequence_name)
    print(sequence_dict[max_sequence_name])

    print(min_sequence_name)
    print(sequence_dict[min_sequence_name])

def split_sequence(sequence_file, sequence_dir): #split sequence

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    for line in line_set:
        line = line.strip()
        if (line.startswith(">")):
            name = line[1:]
        else:

            if(len(line)>sequence_cut_off):
                line = line[0:sequence_cut_off]

            sub_sequence_file = os.path.join(sequence_dir, name + ".fasta")
            f = open(sub_sequence_file, "w")
            f.write(">" + name + "\n" + line + "\n")
            f.close()

def deal_sequence(origin_file, deal_file):  # cut off the sequence whose length<1024

    sequence_dict = read_sequence(origin_file)
    f = open(deal_file, "w")

    for name in sequence_dict:
        sequence = sequence_dict[name]
        if(len(sequence)>sequence_cut_off):
            sequence = sequence[0: sequence_cut_off]
        f.write(">" + name + "\n" + sequence + "\n")

    f.close()

def extract_name_list(sequence_file, name_file): # extract name list from sequence file

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    f = open(name_file, "w")

    for line in text.splitlines():
        line = line.strip()
        if (line.startswith(">")):
            f.write(line[1:] + "\n")
    f.close()

def read_name_list(name_list_file):  # read name list

    f = open(name_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def read_length_from_record_file(record_file): # read sequence length from the record file

    f = open(record_file, "r")
    text = f.read()
    f.close()
    sequence_length_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        sequence_length_dict[values[0]] = int(values[1])

    return sequence_length_dict

def write_list(value_list, save_file):

    f = open(save_file, "w")
    for value in value_list:
        f.write(value + "\n")
    f.close()

def create_sequence_length(sequence_file, length_file): # create sequence length

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    f = open(length_file, "w")

    for line in text.splitlines():
        line = line.strip()
        if(line.startswith(">")):
            name = line[1:]
        else:
            f.write(name + " " + str(len(line)) + "\n")
    f.close()

def read_result(result_file):   # read results

    if (os.path.exists(result_file) == False):
        return dict()

    f = open(result_file, "r")
    text = f.read()
    f.close()

    result_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        result_dict[values[0]] = float(values[2])

    return result_dict

def read_distance(distance_file):  # read result file

    f = open(distance_file, "r")
    text = f.read()
    f.close()

    distance_dict = dict()

    for line in text.splitlines():

        values = line.strip().split()
        distance_dict[values[0]] = float(values[1])

    return distance_dict

def read_label(label_file):  # read go dict

    f = open(label_file, "r")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict

def read_name_list_from_sequence(sequence_file): # read name list from sequence file

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    name_list = []

    for line in text.splitlines():
        line = line.strip()
        if (line.startswith(">")):
            name_list.append(line[1:])
    return name_list

def read_ensemble_datafile(data_file):


    pro_array = []
    term_array = []

    f = open(data_file, "r")
    text = f.read()
    f.close()

    for line in text.splitlines():
        values = line.strip().split()

        temp_pro = []
        for i in range(1, len(values)):
            temp_pro.append(float(values[i]))

        pro_array.append(temp_pro)
        term_array.append(values[0])

    pro_array = np.array(pro_array)
    term_array = np.array(term_array)

    return pro_array, term_array


def split_sequence_another(sequence_file, sequence_dir): #split sequence

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    for line in line_set:
        line = line.strip()
        if (line.startswith(">")):
            name = line[1:]
        else:

            if(len(line)>sequence_cut_off):
                line = line[0:sequence_cut_off]

            current_dir = os.path.join(sequence_dir, name)
            if(os.path.exists(current_dir)==False):
                os.makedirs(current_dir)

            sub_sequence_file = os.path.join(current_dir, ppi_seq_name)
            f = open(sub_sequence_file, "w")
            f.write(">" + name + "\n" + line + "\n")
            f.close()







if __name__ == "__main__":
    deal_sequence(sys.argv[1], sys.argv[2])



