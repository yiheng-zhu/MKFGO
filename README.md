# MFKGO

MFKGO is a composite protein function prediction model in the context of Gene Ontology (GO) through the integration of five complementary pipelines (i.e., HFRGO, PLMGO, PPIGO, NAIGP, and DLMGO) built on multi-source biological data. 

## System Requirements
### 1. Conda Environment: 
Python==3.8.5  
Tensorflow-gpu==2.6.0  
CUDA>=11.3  
cudnn>=8.2.1 
### 2. Software  
(1) <a href="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/">BLAST</a>, used to generate PSSM feature.  
(2) <a href="https://github.com/jas-preet/SPOT-1D-LM">SPOT-1D-LM</a> , used to generate SSCM feature.  
(3) <a href="https://www.ebi.ac.uk/interpro/download/">InterProScan</a> , used to generate FDBV feature.  
(4) <a href="https://github.com/agemagician/ProtTrans">ProtTrans</a>, used to generate protein language model-based feature.  
(5) <a href="https://github.com/instadeepai/nucleotide-transformer">Nucleotide Transformers</a>, used to generate DNA language model-based feature. 




