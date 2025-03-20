import Caculate_pr_tc_each_protein as cp
import threading
import Calculate_ru_mi as cr
import math

class thread_evaluation(threading.Thread):

    def __init__(self, label_dict, result_list_dict, ic_dict, t):
        super(thread_evaluation, self).__init__()

        self.label_dict = label_dict
        self.result_list_dict = result_list_dict
        self.ic_dict = ic_dict
        self.t = t
        self.pr = 0
        self.rc = 0
        self.fmax = 0
        self.pr_count = 0
        self.rc_count = 0


    def caculate(self):

        pr_count = 0.0
        rc_count = 0.0
        sum_pr = 0.0
        sum_rc = 0.0
        pr = 0.0
        rc = 0.0
        fmax = 0.0

        mi = 0.0
        ru = 0.0
        smin = 0.0


        for name in self.result_list_dict.keys():

            tp, fp, number_real_label=cp.caculate(self.result_list_dict[name], self.label_dict, name, self.t)
            single_mi, single_ru = cr.calculate(self.result_list_dict[name], self.label_dict[name], self.ic_dict, self.t)

            if(tp + fp > 0):
                pr_count = pr_count+1
                sum_pr = sum_pr + tp/(tp+fp)

            if(number_real_label > 0):
                rc_count = rc_count+1
                sum_rc = sum_rc + tp/number_real_label

                mi = mi + single_mi
                ru = ru + single_ru

        if(pr_count>0):
            pr = sum_pr/pr_count

        if(rc_count>0):
            rc = sum_rc/rc_count
            mi = mi/rc_count
            ru = ru/rc_count
            smin = math.sqrt(mi*mi + ru*ru)



        if(pr+rc>0):
            fmax = 2*pr*rc/(pr+rc)

        return pr, rc, fmax, smin, pr_count, rc_count

    def run(self):
         self.pr, self.rc, self.fmax, self.smin, self.pr_count, self.rc_count = self.caculate()

    def get_evaluation_indexs(self):
        return self.pr, self.rc, self.fmax, self.smin, self.pr_count, self.rc_count


