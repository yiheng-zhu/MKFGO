import math
import sys
def find_opt_fmax(file):  # find by fmax

    f=open(file, "rU")
    line_txt=f.read()
    f.close()

    opt_fmax=0
    opt_t=0

    for line in line_txt.splitlines():

        values=line.strip().split()
        t=float(values[0][2:])
        fmax=float(values[3][5:])

        if(fmax>opt_fmax):
            opt_fmax = fmax
            opt_t = t

    return opt_t

def find_opt_Smin(file):  # find by fmax

    f=open(file, "rU")
    line_txt=f.read()
    f.close()

    opt_smin=9999999999
    opt_t=0

    for line in line_txt.splitlines():

        values=line.strip().split()
        t=float(values[0][2:])
        smin = float(values[4][5:])

        if(smin<opt_smin):
            opt_smin = smin
            opt_t = t

    return opt_t

def get_fmax_by_T(file, t):  # get measures by T

    f=open(file, "rU")
    line_txt = f.read()
    f.close()

    for line in line_txt.splitlines():

        values = line.strip().split()
        temp_t = float(values[0][2:])

        if(math.fabs(t-temp_t)<0.0000000000001):
            return [values[0], values[1], values[2], values[3], values[4], values[5], values[6]]
    return []

def get_measures_files(file1, file2):

    values_list1 = get_fmax_by_T(file2, find_opt_fmax(file1))
    values_list2 = get_fmax_by_T(file2, find_opt_Smin(file2))
    values_list1.insert(5, values_list2[4])

    return values_list1

