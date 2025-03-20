#!/usr/bin/env python
docstring='''
blast2msa.py query.fasta blastp.xml blastp.msa
    convert XML blastp outfile file "blastp.xml" to FASTA file "blastp.msa"
    aligned to query sequnce supplied by FASTA format query sequence 
    "query.fasta"
'''
import re

# hit sequence name
Hit_id_pattern=re.compile("<Hit_accession>([\w\W]+?)</Hit_accession>")
# e-value
Hsp_evalue_pattern=re.compile("<Hsp_bit\-score>([-.e\d]+?)</Hsp_bit\-score>")
# first aligned residue in query
Hsp_query_from_pattern=re.compile("<Hsp_query\-from>(\d+)</Hsp_query\-from>")
# last aligned residue in query
Hsp_query_to_pattern=re.compile("<Hsp_query\-to>(\d+)</Hsp_query\-to>")
# number of identical residues
Hsp_identity_pattern=re.compile("<Hsp_identity>(\d+)</Hsp_identity>")
# aligned query sequence
Hsp_qseq_pattern=re.compile("<Hsp_qseq>([-\w]+?)</Hsp_qseq>")
# aligned hit sequence
Hsp_hseq_pattern=re.compile("<Hsp_hseq>([-\w]+?)</Hsp_hseq>")

def blast2msa(sequence,blastp_xml=""):
    '''convert ncbi blast+ XML format output text into FASTA format text
    where all blastp hits are aligned to query sequence "sequence"
    '''
    blastp_msa=''
    qlen=len(sequence)
    for block in blastp_xml.split("<Hit>")[1:]:

        Hit_id=Hit_id_pattern.findall(block)[0]
        Hsp_evalue=Hsp_evalue_pattern.findall(block)[0]
        Hsp_identity=Hsp_identity_pattern.findall(block)[0]
        Hsp_query_from=int(Hsp_query_from_pattern.findall(block)[0])
        Hsp_query_to=int(Hsp_query_to_pattern.findall(block)[0])
        Hsp_qseq=Hsp_qseq_pattern.findall(block)[0]
        Hsp_hseq=Hsp_hseq_pattern.findall(block)[0]

        aln_len=Hsp_query_to-Hsp_query_from+1
        Hsp_qseq,Hsp_hseq=zip(*[(q,h) for (q,h) in \
            zip(Hsp_qseq,Hsp_hseq) if q!='-'])
        Hsp_hseq='-'*(Hsp_query_from-1)+''.join(Hsp_hseq)+ \
                 '-'*(qlen-Hsp_query_to)
        
        header=Hit_id+'\t'+Hsp_evalue+'\t'+Hsp_identity+'/'+str(aln_len)
        blastp_msa+='>'+header+'\n'+Hsp_hseq+'\n'
    return blastp_msa

def run_extract_msa(seqfile, xmlfile, msafile):

    sequence = read_single_fasta(seqfile)

    fp = open(xmlfile, "rU")
    blastp_xml = fp.read()
    fp.close()

    blastp_msa = blast2msa(sequence, blastp_xml)

    fp = open(msafile, "w")
    fp.write(blastp_msa)
    fp.close()


def read_single_fasta(fasta_file):
    '''read single entry FASTA format sequence file "fasta_file"
    and return the sequence'''
    fp=open(fasta_file,'rU')
    txt=fp.read().strip()
    fp.close()

    if txt.startswith('>'): # fasta format:
        sequence=''.join([line.strip() for line in txt.splitlines()[1:]])
    else: # plain text
        sequence=''.join([line.strip() for line in txt.splitlines()])
    return sequence
