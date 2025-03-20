#!/usr/bin/env python
'''configuration file for webserver'''
import os

# root path for MetaGO backend server: /nfs/amino-home/zhanglabs/MetaGO
#rootdir=os.path.dirname(os.path.dirname(
    #os.path.abspath(os.path.dirname(__file__))))
rootdir=os.path.join(os.getenv("HOME"),"MetaGO")

#### job scheduling system ####
# excutable to query job statistics
qstat="qstat -f|grep Job_Name"
# excutable to query job statistics
qsub="qsub "
# maximum number of jobs to run
max_job_num=100

#### webportal frontend server ####
frontend_server="zhanglab.ccmb.med.umich.edu"
# user name at frontend server
frontend_username="zhanglabs"
# directory of data folder for individual job at frontend server
frontend_outputdir="/www/html/MetaGO/output"

#### webpage display ####
# HTTP URL to PDB entries
rcsb_link="http://www.rcsb.org/pdb/explore/explore.do?structureId="
# HTTP URL to BioLiP entries
biolip_link="http://zhanglab.ccmb.med.umich.edu/BioLiP/sym.cgi?code="
# HTTP URL to EC entries
ec_link="http://enzyme.expasy.org/EC/"
# HTTP URL to Amigo GO website
go_link="http://amigo.geneontology.org/amigo/term/"
# HTTP URL to MetaGO web portal http://zhanglab.ccmb.med.umich.edu/MetaGO
http_link="http://%s/%s"%(frontend_server,os.path.basename(rootdir))
# citation to paper
citation="<li>Chengxin Zhang, Peter L Freddolino, and Yang Zhang. <i>MetaGO: A Structure and Protein-Protein Interaction Network Based Pipeline for Automated Protein Function Annotations</i>. in submission."
# javascripts for JSmol and 3Dmol
javascript_list=["/jmol/JSmol.min.js", "/jmol/Jmol2.js",
    "/3Dmol/3Dmol-min.js"]
# molecular graphic system: 3Dmol, JSmol
protein_viewer="3Dmol"

#### directory of executables ####
bindir=os.path.join(rootdir,"bin")
# main script to run the full MetaGO pipeline
runCOFACTOR=os.path.join(bindir,"runCOFACTOR.pl")

#### user submission log ####
logdir=os.path.join(rootdir,"log")
# list of unfinished jobs
pending_txt=os.path.join(logdir,"list_pending.txt")
# log file only present when there are too many pending jobs
log_txt=os.path.join(logdir,"log.txt")

#### data output ####
outputdir=os.path.join(rootdir,"output") 
# prefix of job ID
jobID_prefix="MTG"
# list of output files
output_list=[
    "index.html",     # result webpage
    "result.tar.bz2", # result tarball
    "seq.fasta",      # input sequence
    "input.pdb",      # input structure
    "GOsearchresult_final_MF.svg",  # DAG for MF
    "GOsearchresult_final_BP.svg",  # DAG for BP
    "GOsearchresult_final_CC.svg",  # DAG for CC
    "GOsearchresult_final_MF.csv",  # prediction for MF
    "GOsearchresult_final_BP.csv",  # prediction for BP
    "GOsearchresult_final_CC.csv",  # prediction for CC
]
optional_output_pattern_list=["model1_*.pdb","BSITE"]

#### flock ####
# lock target file whom crontab scripts try to put flock to 
# avoid multiple instance of crontab scripts
lock_target=os.path.join(logdir,"lock.zcx")
