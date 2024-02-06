import fire
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_dir)

from mymi.training import train_reg_seg_model

if __name__ == '__main__':
    fire.Fire(train_reg_seg_model)
