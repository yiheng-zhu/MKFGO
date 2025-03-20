import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
from config import device_type_dict, glm_single_feature_dim, glm_max_subsplits_number, device_type


def extract_feature(sequence_dir, feature_dir, model_name):

    device = torch.device(device_type_dict[device_type]["device"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    max_length = tokenizer.model_max_length//2

    name_list = os.listdir(sequence_dir)

    for name in name_list:

        sequence_file = os.path.join(sequence_dir, name)
        f = open(sequence_file, "r")
        f.readline()
        sequence = f.readline().strip()
        f.close()

        feature_file = os.path.join(feature_dir, name.split(".")[0].strip() + ".npy")

        final_mean_sequence_embeddings = np.zeros(glm_single_feature_dim)
        count = 0

        while (count == 0 or len(sequence) >= max_length):

            tokens_ids = tokenizer.batch_encode_plus([sequence[0:max_length]], return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"].to(device)

            # Compute the embeddings
            attention_mask = tokens_ids != tokenizer.pad_token_id
            torch_outs = model(
                tokens_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True
            )

            embeddings = torch_outs['hidden_states'][-1]
            attention_mask = torch.unsqueeze(attention_mask, dim=-1)

            mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
            final_mean_sequence_embeddings = final_mean_sequence_embeddings + mean_sequence_embeddings[0].detach().cpu().numpy()
            count = count + 1
            sequence = sequence[max_length:]

            if (count > glm_max_subsplits_number):
                break

        np.save(feature_file, final_mean_sequence_embeddings / count)
        torch.cuda.empty_cache()

if __name__ == "__main__":

    extract_feature(sys.argv[1], sys.argv[2], sys.argv[3])



