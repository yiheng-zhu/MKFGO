
def calculate(result_dict, label_list, ic_dict, t):

    mi = 0.0
    for term in result_dict:
        if(float(result_dict[term])>=t and term not in label_list):
            mi = mi + ic_dict[term]

    ru = 0.0
    for term in label_list:
        if(term not in result_dict or float(result_dict[term])<t):
            ru = ru + ic_dict[term]

    return mi, ru





