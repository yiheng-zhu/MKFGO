import os

test_sequence_filename = "test_seq.fasta"
sequence_cut_off = 1024
all_data_dir = "/data/yihengzhu/GOA/resource/source_code/data/"
all_model_number = 5


##########################################################################################################################

# gpu
device_type = "cpu"
device_type_dict = dict()
device_type_dict["gpu"] = dict()
device_type_dict["gpu"]["device"] = "cuda:1"
device_type_dict["gpu"]["device_id"] = "1"
device_type_dict["gpu"]["gpu_ratio"] = 1.0
device_type_dict["cpu"] = dict()
device_type_dict["cpu"]["device"] = "cpu"
device_type_dict["cpu"]["device_id"] = ""

##########################################################################################################################







##########################################################################################################################

# GO Terms
go_term_dir = os.path.join(all_data_dir, "go_terms")
go_obo_file = os.path.join(go_term_dir, "go-basic.obo")
excludeGO = "GO:0003674,GO:0008150,GO:0005575"
go_type_list = ["MF", "BP", "CC"]
go_label_size = {"MF": 6858, "BP": 19687, "CC": 2830}
go_label_size_small = {"MF": 6353, "BP": 19131, "CC": 2714}
post_deal_script_file = "/data/yihengzhu/GOA/resource/source_code/code/Find_Parents.py"
min_go_prob = 0.01

##########################################################################################################################






##########################################################################################################################

#pssm
blast_workdir = "/data/yihengzhu/toolbars/sequence_homology_tools/PSSM/"
blast_script_file = os.path.join(blast_workdir, "ncbi-blast-2.15.0+/bin/psiblast")
blast_database = os.path.join(blast_workdir, "swiss_prot/uniprot_sprot")
blast_workspace = "pssm"
blast_sequence_dir = "sequence"
blast_original_pssm_dir = "original_pssm"
blast_log_pssm_dir = "log_pssm"
blast_pssm_array_dir = "pssm_array"
blast_e_value = 0.001
blast_iterations = 3
pssm_col_num = 20

##########################################################################################################################





##########################################################################################################################
#ss
ss_workspace = "ss"
ss_name_file = "name"
ss_seq_dir = "seq"
ss_result_dir = "pred_results"

ss_esm1b_feature_dir = "esm1b_features"
ss_esm_env_python_dir = "/home/yihengzhu/anaconda3/envs/ESM/bin/"
ss_esm_fe_script_file = "/data/yihengzhu/GOA/resource/source_code/code/esm_extract.py"
ss_esm1b_model_name = "esm1b_t33_650M_UR50S"
ss_esm1b_model_layer = 33
ss_esm1b_token_type = "per_tok"

ss_prottrans_feature_dir = "prottrans_features"
ss_prottrans_env_python_dir = "/home/yihengzhu/anaconda3/envs/ProtTrans/bin/"
ss_prottrans_fe_script_file = "/data/yihengzhu/GOA/resource/source_code/code/prottrans_extract.py"

ss_pred_script_file = "/data/yihengzhu/toolbars/SSPred/SPOT-1D-LM-main/run_inference.py"
ss_pred_model_dir = "/data/yihengzhu/toolbars/SSPred/SPOT-1D-LM-main/checkpoints/"

ss_type = ["H" , "G" , "I" , "E" , "B" , "T", "S", "C"]
ss_type_number = 8
ss_array_dir = "ss_array"

##########################################################################################################################





##########################################################################################################################

#interproscan
interproscan_workdir = "/data/yihengzhu/toolbars/sequence_homology_tools/InterPro/interproscan-5.69-101.0/"
interproscan_script_file = os.path.join(interproscan_workdir, "interproscan.sh")
interproscan_entry_list_file = os.path.join(interproscan_workdir, "entry.list")
interproscan_workspace = "interpro"
interproscan_result_filename = "inital_interpro_result"
interproscan_temp_filedir = "interpro_temp"
interproscan_resultdir = "interpro_results"
interproscan_featuredir = "interpro_array"

##########################################################################################################################







##########################################################################################################################

#hand craft
hc_workspace = "hand_craft"
hc_pssm_feature_size = 20
hc_ss_feature_size = 10
hc_interpro_feature_size = 45899
hc_embedding_size = 128
hc_batch_size = 128
hc_mask_dim = 256
hc_rnn_unit = 128
hc_attention_head_number = 8
hc_interpro_layer_fc_number = {"MF":1024, "BP":1024, "CC":64}
hc_full_connect_number = 1024
hc_drop_prob = 0.8
hc_learning_rate = 0.0001


hc_t_cut_off = 0.8
hc_t_margin = 0.1
hc_alpha = 0.01

hc_data_dir = os.path.join(all_data_dir, hc_workspace)
hc_train_feature_dir = os.path.join(hc_data_dir, "features")
hc_train_pssm_feature_dir = os.path.join(hc_train_feature_dir, "pssm_array")
hc_train_ss_feature_dir = os.path.join(hc_train_feature_dir, "ss_array")
hc_train_interpro_feature_dir = os.path.join(hc_train_feature_dir, "interpro_array")


hc_train_name = "train_name_list"
hc_train_label = "train_label"
hc_train_label_onehot = "train_label_one_hot"
hc_train_label = "train_label"
hc_term_list = "term_list"
hc_train_seq_length = "train_seq_length"
hc_train_embeddings_name = "embeddings"
hc_train_embedding_dir = "train_embeddings"
hc_train_name_random_dir = "train_name_random"
hc_train_name_random = "name_random"


hc_test_name = "test_name_list"
hc_test_label_onehot = "test_label_one_hot"
hc_test_seq_length = "test_seq_length"

hc_model_name = "model"

hc_cross_entropy_name = "cross_entropy"
hc_cross_entropy_dir = hc_cross_entropy_name + "_result"
hc_final_cross_entropy_name = "final_" + hc_cross_entropy_name
hc_final_cross_entropy_dir = "final_" + hc_cross_entropy_name + "_result"

hc_distance_dir = "distance"
hc_average_distance_dir = "average_" + hc_distance_dir

hc_final_triplet_name = "final_triplet"
hc_final_triplet_dir = "final_triplet_result"

hc_final_combine_name = "final_combine"
hc_final_combine_dir = "final_combine_result"

hc_round_name = "round"

k_number_dict = {"MF": 30, "BP": 80, "CC": 100}
hc_combine_weight_dict = {"MF": 0.7, "BP": 0.4, "CC": 0.7}

##########################################################################################################################



##########################################################################################################################
# plm
plm_workspace = "plm"
plm_data_dir = os.path.join(all_data_dir, plm_workspace)


plm_term_list = "term_list"
plm_test_name = "test_name_list"
plm_test_label_onehot = "test_label_one_hot"
plm_test_feature = "test_feature"

plm_model_name = "model"
plm_result_dir = "cross_entropy_result"
plm_result_name= "cross_entropy"
plm_round_name = "round"

plm_final_result_dir = "final_cross_entropy_result"
plm_final_result_name= "final_cross_entropy"


plm_feature_size = 1024
plm_full_connect_number = 1024
plm_learning_rate = 0.0001
plm_batch_size = 128

##########################################################################################################################


##########################################################################################################################
# ppi
ppi_workspace = "ppi"
ppi_data_dir = os.path.join(all_data_dir, ppi_workspace)
ppi_sequence_database_dir = os.path.join(ppi_data_dir, "sequence_database")
ppi_sequence_database_file = os.path.join(ppi_sequence_database_dir, "protein.sequences.v12.0.fa")
ppi_sequence_file = os.path.join(ppi_sequence_database_dir, "all_sequences_string_database_v12.fasta")
go_database_dir = os.path.join(ppi_data_dir, "GO_database")

ppi_blast_file = os.path.join(blast_workdir, "ncbi-blast-2.15.0+/bin/blastp")

ppi_seq_name = "seq.fatsa"
ppi_blast_xml_file = "blast.xml"
ppi_blast_msa_file = "blast.msa"
ppi_template_name = "ppi_template"
ppi_list_file_name = "ppi_list"
string_api_url = "https://version-12-0.string-db.org/api"
ppi_output_format = "tsv-no-header"
ppi_method_type = "interaction_partners"
go_database_name = "train_sequence.fasta"
go_template_name = "go_template"
ppi_homology_cut_off = 0.95
ppi_go_cut_off = 0.95
go_term_name = "Terms"
max_ppi_number = 100
ppi_result_name = "ppi_result"

ppi_py_file = "/data/yihengzhu/GOA/resource/source_code/code/Find_Parents_New.py"



##########################################################################################################################






##########################################################################################################################

# glm
glm_workspace = "glm"
glm_data_dir = os.path.join(all_data_dir, glm_workspace)
glm_gene_sequence_dir = "gene_sequence"
glm_single_feature_dim = 2560
glm_combine_feature_dim = glm_single_feature_dim*2
glm_human_model_name = "InstaDeepAI/nucleotide-transformer-2.5b-1000g"
glm_multi_species_model_name ="InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
glm_max_subsplits_number = 200
glm_all_gene_sequence_database_dir = os.path.join(all_data_dir, glm_workspace, "all_gene_sequence")
glm_human_feature_dir = "human_feature"
glm_multi_species_feature_dir = "multi_species_feature"

glm_env_python_dir = "/home/yihengzhu/anaconda3/envs/NCTrans_Hub/bin/"
glm_nctrans_script_file = "/data/yihengzhu/GOA/resource/source_code/code/nctrans_extract.py"


glm_term_list = "term_list"
glm_test_name = "test_name_list"
glm_test_label_onehot = "test_label_one_hot"
glm_test_feature = "test_feature"

glm_model_name = "model"
glm_result_dir = "cross_entropy_result"
glm_result_name= "cross_entropy"
glm_round_name = "round"
glm_final_result_dir = "final_cross_entropy_result"
glm_final_result_name= "final_cross_entropy"

glm_feature_size = 5120
glm_full_connect_number = 1024
glm_learning_rate = 0.0001
glm_batch_size = 128

##########################################################################################################################






##########################################################################################################################

# naive
naive_method_name = "naive"
naive_workspace = "naive"
naive_probability_dir = os.path.join(all_data_dir, naive_workspace)

##########################################################################################################################




##########################################################################################################################

# ensemble
ensemble_weight_dict = dict()
ensemble_weight_dict[hc_workspace]  = {"MF": 1.0, "BP": 1.0, "CC": 0.01}
ensemble_weight_dict[plm_workspace] = {"MF": 1.0, "BP": 1.0, "CC": 1.0}
ensemble_weight_dict[ppi_workspace] = {"MF": 0.01, "BP": 1.0, "CC": 1.0}
ensemble_weight_dict[glm_workspace] = {"MF": 1.0, "BP": 1.0, "CC": 1.0}
ensemble_weight_dict[naive_workspace] = {"MF": 1.0, "BP": 1.0, "CC": 0.01}
ensemble_test_feature = "test_feature"

ensemble_modeldir = "model"
ensemble_modelname = "model"
ensemble_test_name = "test_name_list"

ensemble_result_dir = "cross_entropy_result"
ensemble_result_name= "cross_entropy"
ensemble_round_name = "round"
ensemble_final_result_dir = "final_cross_entropy_result"
ensemble_final_result_name= "final_cross_entropy"

##########################################################################################################################


