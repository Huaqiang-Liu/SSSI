# 测试共享内存传输性能，相比直接在本地将tensor加载到GPU又卸载到CPU
import argparse
import time
import torch
import numpy as np
from pathlib import Path
import mmap
import sys

# from ..core import ivshmem_comm as ic
sys.path.append(str(Path(__file__).resolve().parent.parent))
import core.ivshmem_comm as ic
import inference_engine as ie

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["host", "guest"], required=True, help="执行者角色")
    parser.add_argument("--mode", choices=["baseline", "ivshmem"], required=True, help="baseline或共享内存测试")
    return parser.parse_args()

def test_baseline():
    tensor = torch.randn(1024, 1024).half()  # 2MB，在CPU上
    start = time.time()
    # 转移到GPU，再转移回来
    tensor = tensor.cuda()
    tensor = tensor.cpu()
    end = time.time()
    print(f"[Baseline] CPU -> CUDA -> CPU time: {(end - start)*1000:.4f} ms")

def test_guest_ivshmem(shm_path):
    # 生成tensor -> serialize -> 分块 -> 写共享内存 -> 等待host返回 -> 读取共享内存 -> 组装 -> deserialize
    tensor = torch.randn(1024, 1024).half()
    print("[Guest] Tensor generated.")
    start = time.time()
    serialized = ic.serialize_tensor(tensor)
    blocks = ic.split_tensor_bytes(serialized, msg_id=1)

    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

        print("[Guest] Sending tensor to host...")
        ic.write_blocks(shm, blocks)

        print("[Guest] Waiting for host to return tensor...")
        returned_blocks = []
        while True:
            returned_blocks = ic.read_blocks(shm)
            if len(returned_blocks) > 0 and ic.get_msg_id(returned_blocks[0]) > 1:
                break
            else:
                print("[Guest] No data received yet. Waiting...")
                time.sleep(0.01)
        returned_bytes = ic.assemble_blocks(returned_blocks)
        returned_tensor = ic.deserialize_tensor(returned_bytes)
        print("[Guest] Tensor received and reconstructed.")
        end = time.time()
        print(f"[Guest] Round-trip time: {(end - start)*1000:.4f} ms")

def test_host_ivshmem(shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        ic.clear_shm(shm)

        print("[Host] Waiting for tensor from guest...")
        blocks = []
        while True:
            blocks = ic.read_blocks(shm)
            if len(blocks) > 0:
                break
            else:
                print("[Host] No data received yet. Waiting...")
                time.sleep(0.01)
        
        print("[Host] Tensor received. Deserialize and load to GPU...")
        serialized = ic.assemble_blocks(blocks)
        tensor = ic.deserialize_tensor(serialized)
        tensor = tensor.cuda()
        print("[Host] Tensor loaded to GPU. Unload and send back to guest...")
        tensor = tensor.cpu()
        serialized = ic.serialize_tensor(tensor)
        blocks = ic.split_tensor_bytes(serialized, msg_id=ic.get_msg_id(blocks[0])+1)
        
        ic.write_blocks(shm, blocks)
        print("[Host] Tensor sent back.")

def main():
    args = parse_args()

    if args.mode == "baseline":
        if args.role != "host":
            print("Baseline模式只在host上运行")
            return
        test_baseline()
    else:
        if args.role == "guest":
            shm_path = "/sys/bus/pci/devices/0000:00:02.0/resource2"
            test_guest_ivshmem(shm_path)
        elif args.role == "host":
            shm_path = "/dev/shm/shm1"
            test_host_ivshmem(shm_path)

if __name__ == "__main__":
    main()
