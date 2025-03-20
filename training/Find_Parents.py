from module import obo2csv
obo_url="/data/yihengzhu/GOA/go-basic.obo"
excludeGO = "GO:0003674,GO:0008150,GO:0005575"
#excludeGO = ""
import sys
import os

def get_obo_dict():  # read obo_dict

    fp = open(obo_url, 'rU')
    obo_txt = fp.read()
    fp.close()
    obo_dict = obo2csv.parse_obo_txt(obo_txt)
    return obo_dict

def get_obsolete(obo_dict): # get obsolete terms

    return obo_dict.obsolete()

def find_parents(obo_dict, GOterm): # find parents

    obsolete_list=get_obsolete(obo_dict)

    if(GOterm in obsolete_list):
        return []
    else:
        if(obo_dict.is_a(GOterm, direct=False, name=False, number=False)==""):
            return []
        parent_term_list = obo_dict.is_a(GOterm, direct=False, name=False, number=False).split()
        return list(set(parent_term_list)-set(excludeGO.split(",")))

def find_parents_from_file(originfile, obo_dict):  # find parents from file

    f=open(originfile, "rU")
    text=f.read()
    f.close()

    type=""
    temp_dict=dict()

    for line in text.splitlines():
        values=line.strip().split()
        term=values[0]
        type=values[1]
        pro=float(values[2])
        if(pro>=0.01):
            temp_dict[term] = pro

    new_dict = dict()
    for term in temp_dict:
        parent_list = find_parents(obo_dict, term)
        for parent in parent_list:
            if (parent not in new_dict or temp_dict[term] > new_dict[parent]):
                new_dict[parent] = temp_dict[term]

    f=open(originfile, "w")
    for term in new_dict:
        f.write(term+" "+type+" "+str(new_dict[term])+"\n")
    f.flush()
    f.close()

def find_parents_from_file(originfile, new_file, obo_dict):  # find parents from file

    f=open(originfile, "rU")
    text=f.read()
    f.close()

    type=""
    temp_dict=dict()

    for line in text.splitlines():
        values=line.strip().split()
        term=values[0]
        type=values[1]
        pro=float(values[2])
        if (pro >= 0.01):
            temp_dict[term] = pro

    new_dict = dict()
    for term in temp_dict:
        parent_list = find_parents(obo_dict, term)
        for parent in parent_list:
            if (parent not in new_dict or temp_dict[term] > new_dict[parent]):
                new_dict[parent] = temp_dict[term]

    f=open(new_file, "w")
    for term in new_dict:
        f.write(term+" "+type+" "+str(new_dict[term])+"\n")
    f.flush()
    f.close()

def sort_result(originfile):  # sort result

    f = open(originfile, "rU")
    text = f.read()
    f.close()

    temp_dict = dict()

    for line in text.splitlines():
        values = line.strip().split()
        term = values[0]
        type = values[1]
        pro = float(values[2])
        if(pro>=0.01):
            temp_dict[term] = pro

    result_list=[(temp_dict[term], term) for term in temp_dict]
    result_list = sorted(result_list, reverse=True)

    f=open(originfile, "w")
    for value, term in result_list:
        f.write(term+" "+type+" "+str(value)+"\n")
    f.flush()
    f.close()



if __name__=="__main__":

    workdir = sys.argv[1]
    type = sys.argv[2]
    name_list = os.listdir(workdir)
    obo_dict = get_obo_dict()

    for name in name_list:

        result_file = workdir+"/"+name+"/"+sys.argv[3]+"_"+type
        find_parents_from_file(result_file, result_file + "_new", obo_dict)
        sort_result(result_file + "_new")

