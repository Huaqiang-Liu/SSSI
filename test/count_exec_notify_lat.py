# log_guest_worker_0.txt的每一行是一个时间戳，表示开始forward的时间
# log_guest_master.txt的每一行是一个时间戳，表示主线程通知所有worker的时间
# 计算通知到开始执行的时间间隔之和
import os
if __name__ == "__main__":
    with open("log_guest_worker_0.txt", "r") as f_worker, open("log_guest_master.txt", "r") as f_master:
        worker_times = [float(line.strip()) for line in f_worker.readlines()]
        master_times = [float(line.strip()) for line in f_master.readlines()]

    if len(worker_times) != len(master_times):
        print("日志行数不匹配！")
    else:
        total_latency = 0.0
        for w_time, m_time in zip(worker_times, master_times):
            latency = w_time - m_time
            total_latency += latency
            print(f"通知到开始执行的延迟: {latency:.6f} 秒")

        print(f"总的通知到开始执行的延迟: {total_latency:.6f} 秒")