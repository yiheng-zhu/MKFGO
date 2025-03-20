import subprocess
import sys
from decimal import Decimal
import math
import os
import numpy as np
import threading

from config import test_sequence_filename,\
    blast_script_file, \
    blast_database, \
    blast_sequence_dir, \
    blast_original_pssm_dir, \
    blast_log_pssm_dir, \
    blast_pssm_array_dir,\
    blast_workspace,\
    blast_iterations, \
    blast_e_value, \
    pssm_col_num, \
    sequence_cut_off

from common_methods import check_file_not_exist, create_dir, split_sequence

def run_psi_blast(query_file, output_file): # run blast

    cmd = [
        blast_script_file,
        "-query", query_file,
        "-db", blast_database,
        "-num_iterations", str(blast_iterations),
        "-evalue", str(blast_e_value),
        "-out_ascii_pssm", output_file,
        "-outfmt", "0"  # output format 0 for the traditional BLAST output
    ]

    # Run the PSI-BLAST command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running PSI-BLAST: {e}")


def normalize_pssm(origin_pssm_file, log_pssm_file): # normalization

    pssm_matrix = read_pssm(origin_pssm_file)
    f = open(log_pssm_file, "w")
    for row in pssm_matrix:
        line = ""
        for element in row:
            value = 1.0/(1.0 + math.exp(-element))
            line = line + str(Decimal(value).quantize(Decimal("0.000000"))) + " "
        line = line.strip()
        f.write(line + "\n")
    f.close()

def create_pssm_for_proteins_without_templates(logpssm_file, seq_len): # create the pssm for the proteins which cannot hit templates in UniProt database

    f = open(logpssm_file, "w")
    for i in range(seq_len):
        line = ""
        for j in range(pssm_col_num):
            line = line + str(Decimal(0).quantize(Decimal("0.000000"))) + " "
        f.write(line.strip() + "\n")
    f.close()

def read_pssm(pssm_file): # read

    with open(pssm_file) as f:
        lines = f.readlines()

    # Identify the start of the PSSM matrix
    header_end = 3
    data_start = 3
    while len(lines[header_end].strip())>0:
        header_end += 1

    data_lines = lines[data_start:header_end + 1]

    pssm_matrix = []
    for line in data_lines:
        if not line.strip():
            continue
        parts = line.split()
        scores = [float(x) for x in parts[2:22]]
        pssm_matrix.append(scores)

    return pssm_matrix

def read_seq_len(sequence_file): # read sequence length

    f = open(sequence_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    return int(len(line_set[1].strip()))

def run_single_pssm(query_file, origin_pssm_file, log_pssm_file): # single run

    run_psi_blast(query_file, origin_pssm_file)
    seq_len = read_seq_len(query_file)

    try:
        normalize_pssm(origin_pssm_file, log_pssm_file)
    except Exception as e:
        print(e)
        create_pssm_for_proteins_without_templates(log_pssm_file, seq_len)


def pad(x, max_len):

    PAD = 0
    if np.shape(x)[0] > max_len:
        raise ValueError("not max_len")

    s = np.shape(x)[1]
    x_padded = np.pad(x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD)

    return x_padded[:, :s]

def create_pssm_array(log_pssm_file, pssm_array_file):

    feature = np.loadtxt(log_pssm_file)
    [m, n] = feature.shape

    if (m >= sequence_cut_off):
        feature = feature[0:sequence_cut_off, :]

    else:
        feature = pad(feature, sequence_cut_off)

    np.save(pssm_array_file, feature)


def main_process(workdir):

    seq_file = os.path.join(workdir, test_sequence_filename)
    sequence_dir = os.path.join(workdir, blast_workspace, blast_sequence_dir)
    original_pssm_dir = os.path.join(workdir, blast_workspace, blast_original_pssm_dir)
    log_pssm_dir = os.path.join(workdir, blast_workspace, blast_log_pssm_dir)
    pssm_array_dir = os.path.join(workdir, blast_workspace, blast_pssm_array_dir)

    create_dir(sequence_dir)
    create_dir(original_pssm_dir)
    create_dir(log_pssm_dir)
    create_dir(pssm_array_dir)

    split_sequence(seq_file, sequence_dir)

    name_list = os.listdir(sequence_dir)

    for name in name_list:
        query_file = os.path.join(sequence_dir, name)
        origin_pssm_file = os.path.join(original_pssm_dir, name.split(".")[0] + ".pssm")
        log_pssm_file = os.path.join(log_pssm_dir, name.split(".")[0] + ".pssm")
        pssm_array_file = os.path.join(pssm_array_dir, name.split(".")[0] + ".npy")
        run_single_pssm(query_file, origin_pssm_file, log_pssm_file)
        create_pssm_array(log_pssm_file, pssm_array_file)

if __name__ == "__main__":

    main_process(sys.argv[1])