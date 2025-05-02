# 测试共享内存传输性能，相比直接在本地将tensor加载到GPU又卸载到CPU
import argparse
import time
import torch
import numpy as np
from pathlib import Path
import mmap

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["host", "guest"], required=True, help="执行者角色")
    parser.add_argument("--mode", choices=["baseline", "ivshmem"], required=True, help="baseline或共享内存测试")
    return parser.parse_args()

def test_baseline():
    tensor = torch.randn(1024, 1024).half().cuda()  # 2MB
    start = time.time()
    cpu_tensor = tensor.cpu()
    end = time.time()
    print(f"[Baseline] CUDA -> CPU time: {(end - start)*1000:.2f} ms")

def test_guest_ivshmem(shm_path):
    # 生成tensor -> serialize -> 分块 -> 写共享内存 -> 等待host返回 -> 读取共享内存 -> 组装 -> deserialize
    tensor = torch.randn(1024, 1024).half()
    print("[Guest] Tensor generated.")
    serialized = ic.serialize_tensor(tensor)
    blocks = ic.split_tensor_bytes(serialized, msg_id=1)

    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

        print("[Guest] Sending tensor to host...")
        start = time.time()
        ic.write_blocks(shm, blocks)

        print("[Guest] Waiting for host to return tensor...")
        returned_blocks = ic.read_blocks(shm)
        end = time.time()

        print(f"[Guest] Round-trip time: {(end - start)*1000:.2f} ms")
        returned_bytes = ic.assemble_blocks(returned_blocks)
        returned_tensor = ic.deserialize_tensor(returned_bytes)
        print("[Guest] Tensor received and reconstructed.")

def test_host_ivshmem(shm_path):
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)

        print("[Host] Waiting for tensor from guest...")
        blocks = ic.read_blocks(shm)
        print("[Host] Tensor received. Sending it back...")
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
