# MFKGO

MFKGO is a composite protein function prediction model in the context of Gene Ontology (GO) through the integration of five complementary pipelines (i.e., HFRGO, PLMGO, PPIGO, NAIGP, and DLMGO) built on multi-source biological data. 

## System Requirements
### 1. Conda Environment: 
(1) Python==3.8.5  
(2) Tensorflow-gpu==2.6.0  
(3) CUDA>=11.3  
(4) cudnn>=8.2.1 
### 2. Software  
(1) <a href="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/">BLAST</a>, used to generate PSSM feature.  
(2) <a href="https://github.com/jas-preet/SPOT-1D-LM">SPOT-1D-LM</a> , used to generate SSCM feature.  
(3) <a href="https://www.ebi.ac.uk/interpro/download/">InterProScan</a> , used to generate FDBV feature.  
(4) <a href="https://github.com/agemagician/ProtTrans">ProtTrans Transformers</a>, used to generate protein language model-based feature.  
(5) <a href="https://github.com/instadeepai/nucleotide-transformer">Nucleotide Transformers</a>, used to generate DNA language model-based feature.
### 3. Database  
(1) <a href="https://www.uniprot.org/help/downloads">Swiss-Prot database</a>  
(2) <a href="https://string-db.org/cgi/download">PPI sequence database with the version of 12.0 (protein.sequences.v12.0.fa.gz)</a>  
    Note: These two databases should be formatted by the "makeblastdb" command in BLAST software.  
(3) Library of MFKGO

## Prediction
1. rename your protein sequence file as "test.fasta" with fasta format.
2. create a new directory (e.g. test_example), which contains the "test.fasta"  
3. modify the file paths of the above-mentioned sofwares and databases in config.py
4. Running prediction  
   <b>python ./testing/main_process.py ./test_example/ 1 (or 0)</b>  
    "1" means that we run all of five pipelines for GO predictions.  
    "0" means that we run four pipelines excluding the DLMGO for GO predictions.  
   (a) hand_craft_method.py (HFRGO)  
   (b) plm_method.py (PLMGO)  
   (c) ppi_method.py (PPIGO)
   (d) naive_method.py  (NAIGO)  
   (e) glm_method.py  (DLMGO)  
   (f) ensemble_method.py (ensemble procedure for five GO prediction pipelines)
5. Outputs  
   (a) ./test_example/hand_craft/ (The prediction results for HFRGO, see details in final_cross_entropy_MF/BP/CC_new)  
   (b) ./test_example/plm/ (The prediction results for PLMGO, see details in final_cross_entropy_MF/BP/CC_new)  
   (c) ./test_example/ppi/ (The prediction results for PPIGO, see details in ppi_result_MF/BP/CC_new)  
   (d) ./test_example/naive/ (The prediction results for NAIGO, see details in naive_result_MF/BP/CC_new)  
   (e) ./test_example/glm/ (The prediction results for DLMGO, see details in final_cross_entropy_MF/BP/CC_new)  
   (f) ./test_example/ensemble/ (The final ensemble prediction results of MKFGO integrating all of five pipelines, see details in final_cross_entropy_MF/BP/CC_new)  
   (g) ./test_example/ensemble_withoutdlmgo/ (The final ensemble prediction results of MKFGO integrating four pipelines without DLMGO, see details in final_cross_entropy_MF/BP/CC_new)             

## Training (Optional)
### 1. HFRGO
   (a) Extract PSSM features, see details in ./testing/generate_pssm_feature.py  
   (b) Extract SSCM features, see details in ./testing/generate_ss_feature.py  
   (c) Extract generate_interpro_feature.py, see details in ./testing/generate_interpro_feature.py  
   (d) Training HFRGO, see details in ./training/LSTM_Combine_PSSM_SS_InterPro_Attention_Triplet.py  
### 2. PLMGO
   (a) Extract Prottrans features, see details in ./testing/prottrans_extract.py  
   (b) Training PLMGO: see details in ./training/Triplet_Network_With_Global_Loss.py
### 3. PPIGO
   <code>see details in ./testing/ppi_method.py</code>
### 4. NAIGO
   <code>see details in ./testing/naive_method.py </code>
### 5. DLMGO
   (a) download gene sequence using UniProt ID  
       <code>see details in ./testing/download_gene_sequence.py</code>  
         
   (b) extract feature embeddings using the Nucleotide Transformers  
       <code>see details in ./testing/nctrans_extract.py</code>  
         
   (c) Training DLMGO  
       <code>see details in ./training/Triplet_Network_With_Global_Loss.py</code>  
### 6. Ensembles
   (a) Create Data File, see details in ./training/Create_Single_Sample_File.py  
   (b) Training MKFGO using fully connected neural networks, see details in ./training/MLP_Ensemble_SKL.py  
   




