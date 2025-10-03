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
        ic.write_blocks(shm, blocks, "host")

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
        
        ic.write_blocks(shm, blocks, "host")
        print("[Host] Tensor sent back.")

def test_lora(shm_path):
    '''
    此时推理一定在host上开始，所有普通层都在host，lora层在guest
    每一层推理的流程如下：
    host得到输入后
	如果本层有LoRA，将输入发给guest，确保guest接收完毕后host再开始推理
		如果host先于guest推理完，那自然不必多说，阻塞等待数据即可
		如果host后于guest推理完，需要等待guest把数据发来
	如果没有，正常推理，得到输出直接给下一层作为输入（进入下一个小循环）
    本函数测试一个小循环，最复杂的情况：本层和下一层都有lora层，首先host将tensor1发给guest，
    guest收到后将tensor2发回host，host收到后将tensor1和tensor2相加作为本层输出。
    '''
    with open(shm_path, "r+b") as f:
        shm = mmap.mmap(f.fileno(), 16 * 1024 * 1024)
        ic.clear_shm(shm)

        if shm_path == "/sys/bus/pci/devices/0000:00:02.0/resource2": # guest
            # guest先等host发来tensor1
            print("[Guest] Waiting for tensor1 from host...")
            blocks = []
            while True:
                blocks = ic.read_blocks(shm, "guest")
                if len(blocks) > 0 and ic.get_msg_id(blocks[0]) == 1:
                    break
                else:
                    print("[Guest] No data received yet. Waiting...")
                    time.sleep(0.01)
            print("[Guest] Tensor1 received. Deserialize and load to GPU...")
            serialized = ic.assemble_blocks(blocks)
            tensor1 = ic.deserialize_tensor(serialized)
            print("[Guest] Tensor1 loaded to GPU.")

            # guest进行lora推理，得到tensor2
            tensor2 = tensor1 * 2  # 模拟lora层操作
            tensor2 = tensor2.cpu()
            serialized = ic.serialize_tensor(tensor2)
            blocks = ic.split_tensor_bytes(serialized, msg_id=2)

            print("[Guest] Sending tensor2 back to host...")
            ic.write_blocks(shm, blocks, "guest")
            print("[Guest] Tensor2 sent back to host.")

        elif shm_path == "/dev/shm/shm1": # host
            # host生成tensor1并发给guest
            tensor1 = torch.randn(1024, 1024).half()
            print("[Host] Tensor1 generated.")
            serialized = ic.serialize_tensor(tensor1)
            blocks = ic.split_tensor_bytes(serialized, msg_id=1)

            print("[Host] Sending tensor1 to guest...")
            ic.write_blocks(shm, blocks, "host")

            # host等guest发回tensor2
            print("[Host] Waiting for tensor2 from guest...")
            blocks = []
            while True:
                blocks = ic.read_blocks(shm, "host")
                if len(blocks) > 0 and ic.get_msg_id(blocks[0]) == 2:
                    break
                else:
                    print("[Host] No data received yet. Waiting...")
                    time.sleep(0.01)
            print("[Host] Tensor2 received. Deserialize and load to GPU...")
            serialized = ic.assemble_blocks
        else:
            print("未知shm路径")
            return


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
            # test_guest_ivshmem(shm_path)
            test_lora(shm_path)
        elif args.role == "host":
            shm_path = "/dev/shm/shm1"
            # test_host_ivshmem(shm_path)
            test_lora(shm_path)


if __name__ == "__main__":
    main()
