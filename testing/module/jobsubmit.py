#!/usr/bin/env python
docstring='''module to submit and query jobs using PBS job scheduling system
'''
import sys,os
from configure import qsub,qstat
import subprocess
from string import Template
import time

PBS_template=Template("""#!/bin/bash
#PBS -q urgent
#PBS -l $WALLTIME
#PBS -N $TAG
#PBS -o $JOBNAME.out
#PBS -e $JOBNAME.err
$CMD
""") # $WALLTIME - PBS computational resources (walltime,mem)
     # $TAG - PBS job name
     # $JOBNAME - PBS script full path; 
     # $CMD - full command to execute

def submit_job(jobname,cmd,walltime="walltime=200:00:00,mem=4000mb"):
    '''write command "cmd" to PBS script "jobname", submit the script'''
    ## parse Job_Name ##
    tag=os.path.basename(jobname).split('.')[0]

    ## check if job is already running ##
    if tag in showq():
        return "%s running, skip job submission\n"%tag

    ## write PBS script ##
    fp=open(jobname,'w')
    fp.write(PBS_template.substitute(dict(
        TAG=tag,
        JOBNAME=os.path.abspath(jobname),
        CMD=cmd,
        WALLTIME=walltime,
    )))
    fp.close()
    os.chmod(jobname, os.stat(jobname).st_mode|0111)

    ## submit PBS script ##
    while True:
        p=subprocess.Popen(qsub+' '+jobname,shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        if stdout.strip():
            sys.stdout.write(jobname+" submitted\n")
            break
        else:
            time.sleep(5) # something wrong with qsub
    return stdout.strip() # return PBS job ID

def showq():
    '''return job queue status'''
    cmd=qstat
    p=subprocess.Popen(cmd,shell=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=p.communicate()
    return stdout

if __name__=="__main__":
    sys.stderr.write(docstring)
