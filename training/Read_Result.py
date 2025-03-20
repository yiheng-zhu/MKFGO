import os
import sys

def read(resultfile): #read results

    result_dict = dict()
    if (os.path.exists(resultfile) == False):
        return result_dict

    f=open(resultfile,"rU")
    line_txt = f.read()
    f.close()

    for line in line_txt.splitlines():
        result_dict[line.split()[0]] = line.split()[2]


    return result_dict

def print_dict(result_dict):
    for term in result_dict.keys():
        print(term+" "+result_dict[term])

if __name__ == '__main__':

    print_dict(read(sys.argv[1]))