# 共享内存通信逻辑：不冲突地从共享内存读写，利用共享内存的第一个字节管理
# 虽然host和guest访问shared memory的方式不同，但已经在run_host/guest中处理，将其作为字节数组shm传入
# 一次锁住整块共享内存，因为实际场景下推完一层交给下一层是单向的，也只有组装起完整的tensor才能开始推下一层。
import struct
import tensor_utils as tu
import time

# 虽然不是原子的，但是同时只有一方写，另一方只读，所以只要写了一点，就不可能在写完之前读到，再不就是读空。反之亦然
def acquire_lock(shm):
    while True:
        if shm[tu.LOCK_OFFSET] == 0:
            shm[tu.LOCK_OFFSET] = 1
            break
        time.sleep(0.001)  # Sleep 1ms to reduce busy-wait

def release_lock(shm):
    shm[tu.LOCK_OFFSET] = 0


def write_blocks(shm, blocks):
    acquire_lock(shm)
    try:
        offset = tu.LOCK_OFFSET + 1
        block_count = 0
        for block in blocks:
            if block_count > 0 and block_count % tu.MAX_BLOCK_NUM == 0:
                # 先清空共享内存
                shm[tu.LOCK_OFFSET + 1:] = bytearray(len(shm) - 1)
                release_lock(shm)
                time.sleep(1)
                acquire_lock(shm)
                offset = tu.LOCK_OFFSET + 1
            shm[offset:offset+len(block)] = block
            offset += tu.BLOCK_SIZE
            block_count += 1
    finally:
        release_lock(shm)

def read_blocks(shm):
    acquire_lock(shm)
    try:
        blocks = []
        offset = tu.LOCK_OFFSET + 1
        while offset + tu.HEADER_SIZE <= len(shm): # 实际上就是offset <= len(shm)，这么写保险些而已（下一行）
            header = shm[offset:offset+tu.HEADER_SIZE]
            if all(b == 0 for b in header):
                break
            msg_id, seq_id, is_last, payload_len = struct.unpack(tu.BLOCK_HEADER_FORMAT, header)
            payload_start = offset + tu.HEADER_SIZE
            payload_end = payload_start + payload_len
            payload = shm[payload_start:payload_end]
            full_block = header + payload
            blocks.append(full_block)
            offset += tu.BLOCK_SIZE
            if is_last:
                break
            else:
                # 可能读完了整个共享内存，仍没有读完整个tensor，这时要清空共享内存并从头开始，等1秒让写方接着写
                if len(blocks) > 0 and len(blocks) % tu.MAX_BLOCK_NUM == 0:
                    shm[tu.LOCK_OFFSET + 1:] = bytearray(len(shm) - 1)
                    release_lock(shm)
                    time.sleep(1)
                    acquire_lock(shm)
                    offset = tu.LOCK_OFFSET + 1
        return blocks
    finally:
        release_lock(shm)
