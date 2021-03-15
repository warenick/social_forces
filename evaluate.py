import sys
import os
sys.path.append(os.path.abspath('../social_forces'))
from Param import Param
import torch
import time
from SFM_AI import SFM_AI

if __name__ == '__main__':
    future_horizon = 12
    bs = 1000
    neighb_num = 10
    if torch.cuda.is_available():
        dev = torch.device(0)
    dev = 'cpu'
    sfm_ai = SFM_AI(device=dev)
    print("w threading ok")
    exit()

    # "biwi_eth/biwi_eth.txt"
# {'mean_ade': 1.6717669055575417, 'mean_fde': 2.8545318898700534, 'mean_dist': 2.2019563913345337}

# "crowds/crowds_zara02.txt"
# {'mean_ade': 2.058896233295572, 'mean_fde': 2.310119941316802, 'mean_dist': 1.9428932748991867}

# "crowds/students001.txt"
# {'mean_ade': 3.285138066135236, 'mean_fde': 4.164177453339989, 'mean_dist': 1.634656242470243}

# "crowds/students003.txt"
# {'mean_ade': 2.8972231082196505, 'mean_fde': 3.465533395983138, 'mean_dist': 1.6947245597839355}