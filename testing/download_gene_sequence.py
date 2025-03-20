from Bio import SwissProt, ExPASy, Entrez, SeqIO
from config import test_sequence_filename, glm_gene_sequence_dir, glm_all_gene_sequence_database_dir
import os
from common_methods import read_name_list_from_sequence, create_dir

def fecth_entrze_id(record):

    entrez_id = ""
    for reference in record.cross_references:

        if ("GeneID" in reference):
            entrez_id = entrez_id + reference[reference.index("GeneID") + 1] + " "
    entrez_id = entrez_id.strip()

    if (len(entrez_id) == 0):
        entrez_id = None

    return entrez_id

def map_uniprot_id_to_gene_id(uniprot_id):
    try:
        handle = ExPASy.get_sprot_raw(uniprot_id)
        record = SwissProt.read(handle)

        return fecth_entrze_id(record)

    except Exception as e:
        print(f"Error map uniprot id to entrze id: {e}")
        return None

def complement_sequence(dna_sequence):

    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N':'N', 'K':'M', 'W':'W', 'M':'K', 'S':'S', 'Y':'R', 'R':'Y'}
    complement = ""
    for i in range(len(dna_sequence)-1, 0, -1):
        complement = complement + complement_map[dna_sequence[i]]

    return complement

def map_entrez_to_gi(entrez_id):
    try:
        # Set up Entrez
        Entrez.email = "your@email.com"  # Provide your email to NCBI

        # Fetch the record for the given Entrez ID
        handle = Entrez.efetch(db="gene", id=entrez_id, retmode="xml")
        record_list = Entrez.read(handle)
        handle.close()
        gi_number = ""
        strand = ""
        region_start = -1
        region_end = -1

        for record in record_list:
            if('Entrezgene_locus' in record):
                sub_record = record['Entrezgene_locus']
                for sub_sub_record in sub_record:
                    if ('Gene-commentary_accession' in sub_sub_record):
                        gi_number = sub_sub_record['Gene-commentary_accession']
                        if('Gene-commentary_seqs' in sub_sub_record):
                            for locate in sub_sub_record['Gene-commentary_seqs']:
                                if('Seq-loc_int' in locate and 'Seq-interval' in locate['Seq-loc_int'] and 'Seq-interval_from' in locate['Seq-loc_int']['Seq-interval'] and 'Seq-interval_to' in locate['Seq-loc_int']['Seq-interval']):
                                    region_start = locate['Seq-loc_int']['Seq-interval']['Seq-interval_from']
                                    region_end = locate['Seq-loc_int']['Seq-interval']['Seq-interval_to']
                                    try:
                                        strand = locate['Seq-loc_int']['Seq-interval']['Seq-interval_strand'][
                                            'Na-strand'].attributes['value']
                                    except Exception as e:
                                        strand = "plus"

                                    if(strand=="minus"):
                                        return gi_number, int(region_start), int(region_end) + 1, strand
                                    else:
                                        return gi_number, int(region_start) + 1, int(region_end) + 1, strand
                                else:
                                    try:
                                        start_list = []
                                        end_list = []
                                        strand_list = []

                                        all_str_line = str(locate)

                                        while("Seq-interval_from" in all_str_line):

                                            start = all_str_line.find("Seq-interval_from")
                                            end = all_str_line.find("Seq-interval_id")
                                            str_line = all_str_line[start:end]
                                            values = str_line.split(",")


                                            start_list.append(int(values[0].strip("'").split(":")[1].strip().strip("'")))
                                            end_list.append(int(values[1].strip("'").split(":")[1].strip().strip("'")))
                                            pos = str_line.find("attributes={'value': ")
                                            pos_line = str_line[pos:]
                                            strand = pos_line.split(":")[1].strip().split("}")[0].strip().strip("'")
                                            all_str_line = all_str_line[end+len("Seq-interval_id"):]
                                            strand_list.append(strand)

                                        all_pos = start_list + end_list
                                        max_value = max(all_pos)
                                        min_value = min(all_pos)

                                        if (strand_list[0] == "minus"):
                                            return gi_number, int(min_value) , int(max_value) + 1, strand_list[0]
                                        else:
                                            return gi_number, int(min_value) + 1, int(max_value) + 1, strand_list[0]
                                    except Exception as e:
                                        print(e)

        return gi_number, region_start, region_end, strand

    except Exception as e:
        print(f"Error mapping Entrez ID {entrez_id} to GI number with regions: {e}")
        return None, None, None, None

def extract_gene_sequence(entrez_id, gi_number, start, end, strand):
    try:
        # Set up Entrez
        Entrez.email = "yihzhu@njau.edu.cn"  # Provide your email to NCBI

        # Fetch the sequence for the given Entrez ID
        handle = Entrez.efetch(db="Nucleotide", id=gi_number, rettype="fasta", retmode="text", seq_start = start, seq_stop = end)

        # Read the sequence
        record = SeqIO.read(handle, "fasta")

        # Close the handle
        handle.close()

        if(strand=="minus"):
            return complement_sequence(record.seq)
        else:
            return str(record.seq)

    except Exception as e:
        print(f"Error extracting sequence for Entrez ID {entrez_id}: {e}")
        return None

def download_single_sequence(entrez_id, sequence_file):

    gi_number, region_start, region_end, strand = map_entrez_to_gi(entrez_id)
    sequence = ""

    if(region_start!=-1):
        sequence = extract_gene_sequence(entrez_id, gi_number, region_start, region_end, strand)

    if(sequence):
        f = open(sequence_file, "w")
        f.write(">" + entrez_id + "\n" + sequence + "\n")
        f.close()

def donwload_sequence(uniprot_id, gene_sequence_file):

    entrez_id = map_uniprot_id_to_gene_id(uniprot_id)
    if(entrez_id != None):
        try:
            download_single_sequence(entrez_id, gene_sequence_file)
        except Exception as e:
            print(f"Error extracting gene sequence {entrez_id}: {e}")
            return None

def main_process(workdir):

    test_sequence_file = os.path.join(workdir, test_sequence_filename)
    test_name_list = read_name_list_from_sequence(test_sequence_file)
    gene_sequence_dir = os.path.join(workdir, glm_gene_sequence_dir)
    create_dir(gene_sequence_dir)

    for name in test_name_list:

        origin_gene_sequence_file = os.path.join(glm_all_gene_sequence_database_dir, name + ".fasta")
        gene_sequence_file = os.path.join(gene_sequence_dir, name + ".fasta")


        if(os.path.exists(origin_gene_sequence_file)):
            os.system("cp " + origin_gene_sequence_file + " " + gene_sequence_file)
        else:
            donwload_sequence(name, gene_sequence_file)









