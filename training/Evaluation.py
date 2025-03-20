import Read_Label as rl
import Read_Result as rr
import sys
import os
import math
import Thread_Evaluation as te
from decimal import Decimal
import Create_AUROC as ca

ic_file = "/data/yihengzhu/GOA/ic_list"

class evaluation(object):  #initiation

    def __init__(self, labelfile, resultdir, resultfile_name, rocfile):

        self.labelfile = labelfile
        self.resultdir = resultdir
        self.resultfile_name = resultfile_name
        self.rocfile = rocfile

        self.label_dict, self.name_list, self.all_term_list = rl.read(labelfile)
        self.result_list_dict=self.get_result()
        self.ic_dict = self.read_ic_list()

        self.opt_t = 0
        self.opt_pr = 0
        self.opt_rc = 0
        self.opt_fmax = 0
        self.opt_smin = 0
        self.opt_pr_count = 0
        self.opt_rc_count = 0
        self.aupr = 0

    def create_dir(self, workdir):  #create dir

        if (not os.path.exists(workdir)):
            os.makedirs(workdir)

    def read_ic_list(self):   #read information content

        f = open(ic_file, "r")
        text = f.read()
        f.close()

        ic_dict = dict()

        for line in text.splitlines():
            line = line.strip()
            value = line.split()
            ic_dict[value[0]] = float(value[1])

        return ic_dict

    def get_result(self):  #read result

        result_list_dict = dict()
        list_dir = os.listdir(self.resultdir)

        for name in list_dir:
            result_list_dict[name] = rr.read(self.resultdir + "/" + name + "/" + self.resultfile_name)

        return result_list_dict

    def process(self):  #main process

        line=""
        pr_list = []
        rc_list = []
        for t in range(0, 1001, 5):

            thread_e=te.thread_evaluation(self.label_dict, self.result_list_dict, self.ic_dict, t/1000.0)
            thread_e.run()
            pr, rc, fmax, smin, pr_count, rc_count = thread_e.get_evaluation_indexs()

            pr_list.append(pr)
            rc_list.append(rc)

            line=line + "t=" + str(Decimal(t/1000.0).quantize(Decimal("0.00000"))) +\
                 " pr=" + str(Decimal(pr).quantize(Decimal("0.00000"))) +\
                 " rc=" + str(Decimal(rc).quantize(Decimal("0.00000"))) +\
                 " fmax=" + str(Decimal(fmax).quantize(Decimal("0.00000"))) + \
                 " smin=" + str(Decimal(smin).quantize(Decimal("0.00000"))) + \
                 " pr_count=" + str(int(pr_count)) +\
                 " rc_count=" + str(int(rc_count)) +"\n"

            if(fmax > self.opt_fmax):

                self.opt_t = t
                self.opt_pr = pr
                self.opt_rc = rc
                self.opt_fmax = fmax
                self.opt_pr_count = pr_count
                self.opt_rc_count = rc_count

            if (smin < self.opt_smin):
                self.opt_smin = smin

        f=open(self.rocfile, "w")
        f.write(line)
        f.flush()
        f.close()

        for i in range(len(rc_list)-1):
            self.aupr = self.aupr + 0.5*(pr_list[i]+pr_list[i+1])*math.fabs(rc_list[i+1]-rc_list[i])

        self.auc = ca.create_auc(self.label_dict, self.result_list_dict, self.name_list, self.all_term_list)

    def get_opt_result(self):

        return self.opt_t, self.opt_pr, self.opt_rc, self.opt_fmax, self.opt_smin, self.opt_pr_count, self.opt_rc_count

    def get_aupr(self):

        return Decimal(self.aupr).quantize(Decimal("0.000"))

    def get_auc(self):

        return Decimal(self.auc).quantize(Decimal("0.000"))


if __name__ == '__main__':

    ev = evaluation(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    ev.process()
    opt_t, opt_pr, opt_rc, opt_fmax, opt_smin, opt_pr_count, opt_rc_count = ev.get_opt_result()
    print(opt_t)
    print(opt_pr)
    print(opt_rc)
    print(opt_fmax)
    print(opt_pr_count)
    print(opt_rc_count)



