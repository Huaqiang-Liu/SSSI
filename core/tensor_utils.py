# header，block（header+payload）的定义，序列化和反序列化的函数
import torch
import numpy as np
import json
import struct

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

