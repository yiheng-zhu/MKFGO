import os
import sys
import blast2msa_new as bm
import requests
import subprocess
from decimal import Decimal

from config import test_sequence_filename, ppi_workspace, ppi_seq_name, ppi_blast_xml_file, \
    ppi_blast_msa_file, ppi_blast_file, ppi_sequence_database_file, ppi_template_name, \
    ppi_list_file_name, ppi_output_format, string_api_url, ppi_method_type, max_ppi_number,\
    ppi_sequence_file, go_type_list, go_database_dir, go_database_name, go_template_name, \
    ppi_homology_cut_off, go_term_name, go_term_dir, ppi_go_cut_off, min_go_prob, ppi_py_file, ppi_result_name

from common_methods import split_sequence_another, check_file_not_exist

def blast_first_search(workdir): # blast search to find the most homologous protein in STRING database

    seq_file = os.path.join(workdir, ppi_seq_name)
    if(check_file_not_exist(seq_file)):return

    xml_file = os.path.join(workdir, ppi_blast_xml_file)
    cmd = ppi_blast_file + " -query " + seq_file + " -db " + ppi_sequence_database_file+ " -outfmt 5 -out " + xml_file
    os.system(cmd)

def blast_second_search(workdir, go_database_file): # blast search to find the most homologous protein in GOA database

    seq_file = os.path.join(workdir, ppi_seq_name)
    if(check_file_not_exist(seq_file)):return

    xml_file = os.path.join(workdir, ppi_blast_xml_file)
    cmd = ppi_blast_file + " -query " + seq_file + " -db " + go_database_file + " -outfmt 5  -out " + xml_file + " -evalue 0.1"
    os.system(cmd)

def extract_msa(workdir): # extract msa from blast.xml

    xml_file = os.path.join(workdir, ppi_blast_xml_file)
    seq_file = os.path.join(workdir, ppi_seq_name)
    msa_file = os.path.join(workdir, ppi_blast_msa_file)

    if (check_file_not_exist(xml_file)): return
    bm.run_extract_msa(seq_file, xml_file, msa_file)

def extract_ppi_template(workdir): # extract the most homologous protein as the PPI template

    msa_file = os.path.join(workdir, ppi_blast_msa_file)
    if (check_file_not_exist(msa_file)): return

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    line_set = text.splitlines()

    template_id = line_set[0].strip().split("\t")[0][1:]
    score = round(eval(line_set[0].strip().split("\t")[2]),3)

    template_file = os.path.join(workdir, ppi_template_name)
    f = open(template_file, "w")
    f.write(template_id + " " + str(score))
    f.close()


def download_ppi(workdir):  # download the PPIs for the PPI template from the STRING database

    ppi_template_file = os.path.join(workdir, ppi_template_name)
    ppi_file = os.path.join(workdir, ppi_list_file_name)

    if (check_file_not_exist(ppi_template_file)): return

    f = open(ppi_template_file, "r")
    text = f.read()
    f.close()

    ppi_id = text.split()[0]
    my_genes = [ppi_id]

    request_url = "/".join([string_api_url, ppi_output_format, ppi_method_type])

    params = {
        "identifiers": "%0d".join(my_genes),  # your protein
        "species": ppi_id.split(".")[0],  # species NCBI identifier
        "limit": 100000,
        "caller_identity": "www.awesome_app.org",  # your app name
        "network_type": ["functional"]
    }

    response = requests.post(request_url, data=params)
    f = open(ppi_file, "w")

    count = 0

    for line in response.text.strip().split("\n"):

        line = line.strip()
        if (len(line) == 0):
            continue

        l = line.split("\t")
        query_ensp = l[0]
        partner_ensp = l[1]
        combined_score = l[5]

        f.write(" ".join([query_ensp, partner_ensp, combined_score]))
        f.write("\n")

        count = count + 1
        if (count >= max_ppi_number):
            break

    f.close()

def extract_sequence(command): # extract the sequences for download PPIs

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout.strip()
        else:
            output = result.stderr.strip()

        return output

    except Exception as e:
        print(e)

        return ""

def extract_go_template(workdir): # extract the most homologous protein as the GO template

    msa_file = os.path.join(workdir, ppi_blast_msa_file)
    if (check_file_not_exist(msa_file)): return

    f = open(msa_file, "r")
    text = f.read()
    f.close()

    go_template_file = os.path.join(workdir, go_template_name)
    f = open(go_template_file, "w")

    for line in text.splitlines():
       if(line.startswith(">")==False):
           continue

       values = line.strip().split("\t")
       template_id = values[0][1:]
       score = round(eval(values[2]), 3)
       f.write(template_id + " " + str(score) + "\n")

    f.close()

def go_homlogy_search(workdir): # search go termplates for each PPI

    ppi_file = os.path.join(workdir, ppi_list_file_name)
    if (check_file_not_exist(ppi_file)): return

    f = open(ppi_file, "r")
    text = f.read()
    f.close()

    ppi_sequence_dir = os.path.join(workdir, ppi_workspace)
    os.system("rm -rf " + ppi_sequence_dir)
    os.makedirs(ppi_sequence_dir)

    for line in text.splitlines():
        values = line.strip().split()
        command = "grep -A 1 " + values[1] + " " + ppi_sequence_file
        result = extract_sequence(command)
        if(len(result)>0 and result.startswith(">")):

            sub_ppidir = os.path.join(ppi_sequence_dir, values[1])
            os.makedirs(sub_ppidir)
            seq_file = os.path.join(sub_ppidir, ppi_seq_name)
            f = open(seq_file, "w")
            f.write(result+"\n")
            f.close()

            for type in go_type_list:
                tempdir = os.path.join(sub_ppidir, type)
                os.makedirs(tempdir)
                os.system("cp " + seq_file + " " + tempdir)
                go_database_file = os.path.join(go_database_dir, type, go_database_name)
                blast_second_search(tempdir, go_database_file)
                extract_msa(tempdir)
                extract_go_template(tempdir)

def read_ppi_socres(ppi_file): # read ppi scores

    f = open(ppi_file, "r")
    text = f.read()
    f.close()

    score_dict = dict()
    for line in text.splitlines():

        values = line.strip().split()
        score_dict[values[1]] = float(values[2])

    return score_dict

def read_go(gofile):  # read GO Terms

    f = open(gofile, "rU")
    text = f.read()
    f.close()

    go_dict = dict()

    for line in text.splitlines():
        line = line.strip()
        values = line.split()
        go_dict[values[0]] = values[1].split(",")

    return go_dict

def go_annotation(workdir): # GO annotation

    ppi_template_file = os.path.join(workdir, ppi_template_name)
    if (check_file_not_exist(ppi_template_file)): return

    f = open(ppi_template_file, "r")
    text = f.read()
    f.close()

    max_seq_id = float(text.strip().split()[1])

    if(max_seq_id<ppi_homology_cut_off):
        return

    donwload_ppi_file = os.path.join(workdir, ppi_list_file_name)
    if (check_file_not_exist(donwload_ppi_file)): return
    score_dict = read_ppi_socres(donwload_ppi_file)

    for type in go_type_list:

        go_term_file = os.path.join(go_term_dir, type + "_" + go_term_name)
        go_dict = read_go(go_term_file)

        weight_dict = dict()
        label_dict = dict()
        term_list = []

        for ppi_id in score_dict:
            go_template_file = os.path.join(workdir, ppi_workspace, ppi_id, type, go_template_name)
            if (check_file_not_exist(go_template_file)): continue

            weight_dict[ppi_id] = dict()
            label_dict[ppi_id] = dict()

            f = open(go_template_file, "r")
            text = f.read()
            f.close()

            for line in text.splitlines():
                values = line.strip().split()
                homology_id = values[0]
                homology_score = float(values[1])

                if(homology_score<ppi_go_cut_off):
                    continue

                weight_dict[ppi_id][homology_id] = score_dict[ppi_id] * homology_score
                label_dict[ppi_id][homology_id] = go_dict[homology_id]
                term_list.extend(go_dict[homology_id])

        term_list = list(set(term_list))

        result_dict = dict()
        for term in term_list:

            sum1 = 0.0
            sum2 = 0.0

            for ppi_id in weight_dict:
                for homology_id in weight_dict[ppi_id]:
                    if(term in label_dict[ppi_id][homology_id]):
                        sum1 = sum1 + weight_dict[ppi_id][homology_id]
                    sum2 = sum2 + weight_dict[ppi_id][homology_id]

            result_dict[term] = sum1/sum2

        result_list = [(result_dict[term], term) for term in result_dict]
        result_list = sorted(result_list, reverse=True)

        resultfile = os.path.join(workdir, ppi_result_name + "_" + type)

        f = open(resultfile, "w")
        for value, term in result_list:
            if (value >= min_go_prob):
                f.write(term + " " + type[1] + " " + str(Decimal(value).quantize(Decimal("0.000"))) + "\n")
        f.flush()
        f.close()

        os.system("python2 " + ppi_py_file + " " + resultfile)



def main_process(workdir):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)
    sequence_dir = os.path.join(workdir, ppi_workspace)
    split_sequence_another(test_sequence_file, sequence_dir)

    name_list = os.listdir(sequence_dir)
    for name in name_list:

        sub_workspace = os.path.join(sequence_dir, name)

        blast_first_search(sub_workspace)
        extract_msa(sub_workspace)
        extract_ppi_template(sub_workspace)
        download_ppi(sub_workspace)
        go_homlogy_search(sub_workspace)
        go_annotation(sub_workspace)



if __name__ == "__main__":

    main_process(sys.argv[1])


