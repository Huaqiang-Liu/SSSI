import torch
import numpy as np
import json
import struct
import time


# header，block（header+payload）的定义，序列化和反序列化的函数
LOCK_OFFSET = 0  # 读写锁占用1字节
BLOCK_SIZE = 4096 + 9
HEADER_SIZE = 9
PAYLOAD_SIZE = 4096
MAX_BLOCK_NUM = 4087 # (16*1024*1024 - 1)//BLOCK_SIZE，1是读写锁使用的1个字节

# <即小端法，I = uint32, H = uint16, B = uint8
# msg_id uint32 一层输出的所有tensor共用一个msg_id
# seq_id uint16 该tensor的第几个block
# is_last uint8 该block是否是最后一个
# payload_len uint16 该block的payload长度，上限是4096，所以uint16就够了
BLOCK_HEADER_FORMAT = '<IHBH'

# 辅助函数
def get_msg_id(block: bytes) -> int:
    # 解析block的header，获取msg_id
    header = block[:HEADER_SIZE]
    msg_id, _, _, _ = struct.unpack(BLOCK_HEADER_FORMAT, header)
    return msg_id

def clear_shm(shm): # 清空共享内存
    shm[LOCK_OFFSET + 1:] = bytearray(len(shm) - 1)


# 一层的输出->字节
def serialize_tensor(tensor: torch.Tensor) -> bytes:
    np_array = tensor.detach().cpu().numpy()
    meta = {
        'shape': list(np_array.shape),
        'dtype': str(np_array.dtype),
    }
    meta_bytes = json.dumps(meta).encode('utf-8')
    meta_len = len(meta_bytes).to_bytes(4, 'little')  # prepend 4-byte length
    return meta_len + meta_bytes + np_array.tobytes()

# 产生header+字节->blocks
def split_tensor_bytes(serialized: bytes, msg_id: int):
    blocks = []
    total_len = len(serialized)
    num_blocks = (total_len + PAYLOAD_SIZE - 1) // PAYLOAD_SIZE

    for seq_id in range(num_blocks):
        start = seq_id * PAYLOAD_SIZE
        end = min(start + PAYLOAD_SIZE, total_len)
        payload = serialized[start:end]
        payload_len = len(payload)
        is_last = 1 if seq_id == num_blocks - 1 else 0

        header = struct.pack(BLOCK_HEADER_FORMAT, msg_id, seq_id, is_last, payload_len)
        assert len(header) == HEADER_SIZE, 'Header must be exactly 11 bytes'
        blocks.append(header + payload)

    return blocks

# 中间的传输过程见ivshmem_comm.py

# blocks->tensor的字节序列
def assemble_blocks(blocks: list) -> bytes:
    blocks.sort(key=lambda b: struct.unpack('<H', b[4:6])[0]) # 按seq_id排序
    payloads = [b[HEADER_SIZE:HEADER_SIZE + struct.unpack('<H', b[7:9])[0]] for b in blocks]
    return b''.join(payloads)

# tensor的字节序列->给下一层的tensor
def deserialize_tensor(data: bytes, use_gpu: bool = False) -> torch.Tensor:
    meta_len = int.from_bytes(data[:4], 'little')
    meta = json.loads(data[4:4+meta_len].decode('utf-8'))
    tensor_data = data[4+meta_len:]
    np_array = np.frombuffer(tensor_data, dtype=meta['dtype']).reshape(meta['shape'])
    if use_gpu:
        return torch.from_numpy(np_array.copy()).cuda()
    return torch.from_numpy(np_array.copy())





# 共享内存通信逻辑：不冲突地从共享内存读写，利用共享内存的第一个字节管理
# 虽然host和guest访问shared memory的方式不同，但已经在run_host/guest中处理，将其作为字节数组shm传入
# 一次锁住整块共享内存，因为实际场景下推完一层交给下一层是单向的，也只有组装起完整的tensor才能开始推下一层。

# 虽然不是原子的，但是同时只有一方写，另一方只读，所以只要写了一点，就不可能在写完之前读到，再不就是读空。反之亦然
def acquire_lock(shm):
    while True:
        if shm[LOCK_OFFSET] == 0:
            shm[LOCK_OFFSET] = 1
            break
        time.sleep(0.001)  # Sleep 1ms to reduce busy-wait

def release_lock(shm):
    shm[LOCK_OFFSET] = 0


def write_blocks(shm, blocks):
    acquire_lock(shm)
    try:
        offset = LOCK_OFFSET + 1
        block_count = 0
        for block in blocks:
            if block_count > 0 and block_count % MAX_BLOCK_NUM == 0:
                # 先清空共享内存
                clear_shm(shm)
                release_lock(shm)
                time.sleep(1)
                acquire_lock(shm)
                offset = LOCK_OFFSET + 1
            shm[offset:offset+len(block)] = block
            offset += BLOCK_SIZE
            block_count += 1
    finally:
        release_lock(shm)

def read_blocks(shm):
    acquire_lock(shm)
    try:
        blocks = []
        offset = LOCK_OFFSET + 1
        while offset + HEADER_SIZE <= len(shm): # 实际上就是offset <= len(shm)，这么写保险些而已（下一行）
            header = shm[offset:offset+HEADER_SIZE]
            if all(b == 0 for b in header):
                break
            msg_id, seq_id, is_last, payload_len = struct.unpack(BLOCK_HEADER_FORMAT, header)
            payload_start = offset + HEADER_SIZE
            payload_end = payload_start + payload_len
            payload = shm[payload_start:payload_end]
            full_block = header + payload
            blocks.append(full_block)
            offset += BLOCK_SIZE
            if is_last:
                break
            else:
                # 可能读完了整个共享内存，仍没有读完整个tensor，这时要清空共享内存并从头开始，等1秒让写方接着写
                if len(blocks) > 0 and len(blocks) % MAX_BLOCK_NUM == 0:
                    clear_shm(shm)
                    release_lock(shm)
                    time.sleep(1)
                    acquire_lock(shm)
                    offset = LOCK_OFFSET + 1
        return blocks
    finally:
        release_lock(shm)


if __name__ == '__main__':
    # print("===== CPU Test =====")
    # cpu_tensor = torch.randn(100, 100, 100)
    # cpu_bytes = serialize_tensor(cpu_tensor)
    # cpu_blocks = split_tensor_bytes(cpu_bytes, 2)
    # cpu_assembled = assemble_blocks(cpu_blocks)
    # cpu_reconstructed = deserialize_tensor(cpu_assembled, use_gpu=False)
    # print(f"CPU tensor shape after round trip: {cpu_reconstructed.shape}")

    if torch.cuda.is_available():
        print("===== GPU Test =====")
        gpu_tensor = torch.randn(100, 100, 100).cuda()
        gpu_bytes = serialize_tensor(gpu_tensor)
        gpu_blocks = split_tensor_bytes(gpu_bytes, 3)
        gpu_assembled = assemble_blocks(gpu_blocks)
        gpu_reconstructed = deserialize_tensor(gpu_assembled, use_gpu=True)
        print(f"GPU tensor shape after round trip: {gpu_reconstructed.shape}")