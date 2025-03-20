def caculate(result_dict, label_dict, protein_name, t):

    tp = 0.0
    fp = 0.0
    number_real_label = 0.0

    if (protein_name in label_dict):
        number_real_label = len(label_dict[protein_name])

    if not result_dict:
        return tp, fp, number_real_label

    for term in result_dict.keys():
        if(float(result_dict[term])>=t):
            if(protein_name in label_dict and term in label_dict[protein_name]):
                tp = tp + 1
            else:
                fp = fp + 1

    return tp, fp, number_real_label