import sys

def read(labelfile):  # read label

    label_dict = dict()
    f=open(labelfile,"r")
    line_txt = f.read()
    f.close()

    all_term_list = []
    name_list = []

    for line in line_txt.splitlines():
        values = line.strip().split()
        term_list = values[1].split(",")
        label_dict[values[0]] = term_list
        all_term_list.extend(term_list)
        name_list.append(values[0])

    all_term_list = list(set(all_term_list))

    return label_dict, name_list, all_term_list

def print_dict(label_dict):

    for term in label_dict.keys():
        print(term)
        print(label_dict[term])

if __name__ == '__main__':

    print_dict(read(sys.argv[1]))
