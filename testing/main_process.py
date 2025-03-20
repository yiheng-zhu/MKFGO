import hand_craft_method as hc
import plm_method as pm
import glm_method as gm
import naive_method as nm
import ppi_method as ppm
import sys
import ensemble_method as em


def main_process(workdir, is_use_dlmgo):

    is_use_dlmgo = bool(is_use_dlmgo)

    hc.main_process(workdir)
    pm.main_process(workdir)
    nm.main_process(workdir)

    try:
        ppm.main_process(workdir)
    except Exception as e:
        print(e)

    if(is_use_dlmgo):
        
        try:
            gm.main_process(workdir)
        except Exception as e:
            print(e)


    em.main_process(workdir, is_use_dlmgo)


if __name__ == "__main__":
    main_process(sys.argv[1], int(sys.argv[2]))


