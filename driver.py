"""
# running with different parameters
# python HNSW_syn_e_M_efc_efs.py --e 3 --M 32 --efc 300 --efs 400 >> log/HNSW_syn_e3_M32_efc300_efs400.txt 2>&1
# python HNSW_syn_e_M_efc_efs.py --e 5 --M 32 --efc 300 --efs 400 >> log/HNSW_syn_e5_M32_efc300_efs400.txt 2>&1
# python HNSW_syn_e_M_efc_efs.py --e 7 --M 32 --efc 300 --efs 400 >> log/HNSW_syn_e7_M32_efc300_efs400.txt 2>&1


# create a variable in bash for efc and have vaules from 100 to 500 with step of 100 via a loop
for ((efc=100; efc<=1000; efc+=100)); do
  efc_values+=($efc)
done

# e will be from 1 to 7 with step of 1
for ((e=1; e<=7; e+=2)); do
  e_values+=($e)
done

# M will be 16 to 64 with step of 16
for ((M=16; M<=64; M+=16)); do
  M_values+=($M)
done

# efs will be from 100 to 500 with step of 100
for ((efs=100; efs<=1000; efs+=100)); do
  efs_values+=($efs)
done

# loop through all combinations of parameters and run the script
for e in "${e_values[@]}"; do
  for M in "${M_values[@]}"; do
    for efc in "${efc_values[@]}"; do
      for efs in "${efs_values[@]}"; do
        echo "Running with e=$e, M=$M, efc=$efc, efs=$efs"
        python HNSW_syn_e_M_efc_efs.py --e $e --M $M --efc $efc --efs $efs >> log/HNSW_syn_e${e}_M${M}_efc${efc}_efs${efs}.txt 2>&1
      done
    done
  done
done
"""

import os

from np_lib import Parallel as pp

if __name__ == "__main__":
    pp_obj = pp() # usage is pp_obj(function = os.system, list_of_args = "python ...")

    e_values = [i for i in range(1, 8, 2)]  # e will be from 1 to 7 with step of 2
    M_values = [i for i in range(16, 65, 16)]  # M will be 16 to 64 with step of 16
    efc_values = [i for i in range(100, 1100, 100)]  # efc will be from 100 to 1000 with step of 100
    efs_values = [i for i in range(100, 1100, 100)]  # efs will be from 100 to 1000 with step of 100    

    for e in e_values:
        for M in M_values:
            for efc in efc_values:
                for efs in efs_values:
                    cmd = f"python HNSW_syn_e_M_efc_efs.py --e {e} --M {M} --efc {efc} --efs {efs} >> log/HNSW_syn_e{e}_M{M}_efc{efc}_efs{efs}.txt 2>&1"
                    print(f"Running with e={e}, M={M}, efc={efc}, efs={efs}")
                    for i in pp_obj(function=os.system, list_of_args=cmd):
                        print(f"Finished with e={e}, M={M}, efc={efc}, efs={efs}")