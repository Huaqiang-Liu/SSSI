import sys, os, json, time, mmap, torch, argparse

HOST_SHM_PATH = "/dev/shm/shm1"
GUEST_SHM_PATH = "/sys/bus/pci/devices/0000:00:02.0/resource2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, required=True, help='Role: host or guest')
    args = parser.parse_args()

    role = args.role
    if role == 'host':
        shm_path = HOST_SHM_PATH
    else:
        shm_path = GUEST_SHM_PATH

    sizes = [1024 * (2 ** i) for i in range(15)]  # 1KB to 16MB
    results = {}

    for size in sizes:
        with open(shm_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), size)
            data = bytearray(size)  # zero-initialized array
            start_time = time.time()
            mm[0:len(data)] = data
            end_time = time.time()
            mm.close()
            duration = end_time - start_time
            results[size] = duration
            print(f"Role: {role}, Size: {size} bytes, Time: {duration:.6f} seconds")




