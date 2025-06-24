# MKFGO

MKFGO is a composite protein function prediction model in the context of Gene Ontology (GO) through the integration of five complementary pipelines (i.e., HFRGO, PLMGO, PPIGO, NAIGP, and DLMGO) built on multi-source biological data. 

## System Requirements
### 1. Conda Environment: 
<li> python==3.8.5  </li>
  
<li> tensorflow-gpu==2.6.0 </li>  
  
<li> pytorch==2.0.0  </li>
   
<li> CUDA>=11.7  </li>
   
<li> cudnn>=8.2.1 </li>  


### 2. Software  
<li> <a href="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/">BLAST</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate PSSM feature &nbsp ###### </li>
  
<li> <a href="https://github.com/jas-preet/SPOT-1D-LM">SPOT-1D-LM</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp######   &nbsp generate SSCM feature &nbsp ######  </li>
  
<li> <a href="https://www.ebi.ac.uk/interpro/download/">InterProScan</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate FDBV feature &nbsp ######  </li>
  
<li> <a href="https://github.com/agemagician/ProtTrans">ProtTrans Transformers</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp###### &nbsp generate protein language model-based feature &nbsp ######  </li>

<li> <a href="https://github.com/facebookresearch/esm">ESM-1b Transformers</a>&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp######  &nbsp generate protein language model-based feature used in SPOT-1D-LM &nbsp ###### </li>
  
<li> <a href="https://github.com/instadeepai/nucleotide-transformer">Nucleotide Transformers</a> &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp######  &nbsp generate DNA language model-based feature &nbsp ######  </li>  

### 3. Data
<li> <a href="https://www.uniprot.org/help/downloads">Swiss-Prot database</a>  </li>
  
<li> <a href="https://string-db.org/cgi/download">PPI sequence database with the version of 12.0 (protein.sequences.v12.0.fa.gz)</a>  </li>
  
<li> <a href="http://www.jcu-qiulab.com/static/servers/GOA/library.zip">Library of MKFGO </a>  </li>

<li> <a href="http://www.jcu-qiulab.com/static/servers/GOA/benchmark_dataset.zip">Benchmark datasets</a>  </li>
  

## Prediction
1. Rename your protein sequence file as "test.fasta" with fasta format.
2. Create a new directory (e.g. test_example), which contains the "test.fasta"  
3. Modify the file paths of the above-mentioned sofwares and databases in the ./testing/config.py
  
4. Running prediction
     
   <code> python ./testing/main_process.py ./test_example/ 1 (or 0)</code>
   <ul>
     
    <li> "1" means that we run all of five pipelines for GO predictions. </li>  
    <li>"0" means that we run four pipelines excluding the DLMGO for GO predictions. </li>
   </ul>

   In main_process.py, five python scripts are orderly performed, including:
   <ul>  
   <li> hand_craft_method.py (HFRGO) </li>
   <li> plm_method.py (PLMGO) </li>
   <li> ppi_method.py (PPIGO) </li>
   <li> naive_method.py  (NAIGO) </li>   
   <li> glm_method.py  (DLMGO) </li>
   <li> ensemble_method.py (ensemble procedure for five GO prediction pipelines) </li>
   </ul>
     
5. Outputs  
   <ul>
   <li> ./test_example/pssm/   &nbsp&nbsp&nbsp ### &nbsp PSSM features &nbsp ### </li>
   <li> ./test_example/ss/  &nbsp&nbsp&nbsp ### &nbsp SSCM features &nbsp ### </li>
   <li> ./test_example/interpro/   &nbsp&nbsp&nbsp ### &nbsp FDBV features &nbsp ### </li>
   <li> ./test_example/hand_craft/   &nbsp&nbsp&nbsp ### &nbsp The prediction results for HFRGO &nbsp ### </li>
   <li> ./test_example/plm/    &nbsp&nbsp&nbsp ### &nbsp The prediction results for PLMGO &nbsp ### </li>
   <li> ./test_example/ppi/    &nbsp&nbsp&nbsp ### &nbsp The prediction results for PPIGO &nbsp ### </li>
   <li> ./test_example/naive/    &nbsp&nbsp&nbsp ### &nbsp The prediction results for NAIGO &nbsp ### </li>
   <li> ./test_example/glm/    &nbsp&nbsp&nbsp ### &nbsp The prediction results for DLMGO &nbsp ### </li>
   <li> ./test_example/ensemble/    &nbsp&nbsp&nbsp ### &nbsp The ensemble prediction results of all five pipelines &nbsp ###</li>
   <li> ./test_example/ensemble_withoutdlmgo/    &nbsp&nbsp&nbsp ### &nbsp The ensemble prediction results of four pipelines without DLMGO &nbsp ### </li>
   </ul>

6. Non-coding gene function prediction (Optional)  
   <ul>
   <li>DLMGO supports function prediction for <b> non-coding genes </b> using DNA sequences as input, complementing existing gene function prediction models such as TripletGO. </li>
   <li>You can directly run the <b>glm_method_non_coding.py </b> to perform function prediction for non-coding genes.
   </li>
   <li> e.g., python glm_method_non_conding.py ./test_example/
   <li>The input DNA sequences should be placed in ./test_example/gene_sequence/
   </li>
   <li>
       The prediction results could be found in ./test_example/glm/ 
   </li>
   </li>
   
   </ul>
   

## Training (Optional)
### 1. HFRGO
   <li> Extract PSSM features
     
   <code> ./testing/generate_pssm_feature.py</code>  
   </li>
     
   <li> Extract SSCM features  

   <code> ./testing/generate_ss_feature.py </code> 
   </li>
     
   <li> Extract generate_interpro_feature.py  
     
   <code> ./testing/generate_interpro_feature.py</code>  
   </li>
     
   <li> Training HFRGO  
     
   <code> ./training/LSTM_Combine_PSSM_SS_InterPro_Attention_Triplet.py</code>  
   </li>
   
### 2. PLMGO
   <li> Extract Prottrans features  
     
   <code> ./testing/prottrans_extract.py</code>  
   </li>
     
   <li> Training PLMGO  
     
   <code> ./training/Triplet_Network_With_Global_Loss.py </code>
   </li>
   
### 3. PPIGO
   <code> ./testing/ppi_method.py</code>
   
### 4. NAIGO
   <code> ./testing/naive_method.py </code>
   
### 5. DLMGO

   <li> Download gene sequence using UniProt ID  
     
   <code>./testing/download_gene_sequence.py</code>  
   </li>
         
   <li> Extract feature embeddings using the Nucleotide Transformers  
     
   <code>./testing/nctrans_extract.py</code>  
   </li>
         
   <li> Training DLMGO  
     
   <code>./training/Triplet_Network_With_Global_Loss.py</code>  
  </li>
  
### 6. Ensembles
   <li> Create data file  
     
   <code> ./training/Create_Single_Sample_File.py </code>  
   </li>
      
   <li> Training MKFGO using fully connected neural networks  
     
   <code> ./training/MLP_Ensemble_SKL.py  </code>
   </li>
   
### 7. Evaluation
<code> ./training/Evaluation.py </code>  




