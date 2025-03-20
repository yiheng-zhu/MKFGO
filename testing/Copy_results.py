import os
import sys

def read_name_list(name_list_file):  # read name list

    f = open(name_list_file, "r")
    text = f.read()
    f.close()

    return text.splitlines()

def copy_results(origindir, copydir, name_list_file):

    name_list = read_name_list(name_list_file)
    for name in name_list:
        if(os.path.exists(origindir + "/" + name + "/")):
            os.system("cp -r " + origindir + "/" + name + "/ " + copydir + "/" + name + "/")

copy_results(sys.argv[1], sys.argv[2], sys.argv[3])

