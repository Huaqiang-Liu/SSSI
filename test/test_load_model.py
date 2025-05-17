import argparse
import time
import torch
import numpy as np
from pathlib import Path
import mmap
import sys

# from ..core import ivshmem_comm as ic
sys.path.append(str(Path(__file__).resolve().parent.parent))
import core.inference_engine as ie

def test(role="host"):
    for end_layer_idx in range(25):
        start_time = time.time()
        ie.load_partitioned_model(
            ie.PAR_MODEL_DIR,
            role == "host",
            0,
            end_layer_idx
        )
        end_time = time.time()
        # 往host.txt中写入执行时间（保留4位小数，单位为秒），然后加一个空格。
        with open("host.txt", "a") as f:
            f.write(f"{end_time - start_time:.4f} ")
            f.flush()
        time.sleep(5) # 避免温度影响结果

if __name__ == "__main__":
    test("guest")