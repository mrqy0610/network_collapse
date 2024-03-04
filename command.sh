srun -c 32 -p inspur -x inspur-gpu-11,inspur-gpu-12,inspur-gpu-13,inspur-gpu-14 --gres=gpu:1 --pty bash
srun -c 32 --gres=gpu:1 -x dell-gpu-10,,dell-gpu-19,dell-gpu-23,dell-gpu-27 -p dell --pty bash
srun -c 32 --gres=gpu:1 -p sugon --pty bash
srun -c 32 -p dell --pty bash

source /home/LAB/anaconda3/bin/activate language_condition_MAT
source /home/LAB/anaconda3/bin/activate language_condition_MAT-A100

source /home/LAB/zhutc/anaconda3/bin/activate mat


 .....


conda create -y -n preference_transformer python=3.8
conda activate preference_transformer



