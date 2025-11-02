def sum_column(file_path, column_index=0):
    total = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            numbers = line.split()
            if len(numbers) >= 1:
                try:
                    total += float(numbers[column_index])
                except ValueError:
                    print(f"Warning: Could not convert {numbers[0]} to float.")
    return f"{total:6f}"

# 一列上的对应数相减，求和(file2的数减去file1的数)
def sum_column_diff(file1_path, file2_path, column_index=0):
    total = 0.0
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            numbers1 = line1.split()
            numbers2 = line2.split()
            if len(numbers1) >= 1 and len(numbers2) >= 1:
                try:
                    total += float(numbers2[column_index]) - float(numbers1[column_index])
                except ValueError:
                    print(f"Warning: Could not convert to float.")
    return f"{total:6f}"

if __name__ == "__main__":
    print(f"总传输时间host{sum_column('log_host.txt', 0)}秒，guest{sum_column('log_guest.txt', 0)}秒")
    print(f"guest上forward函数的执行时间为{sum_column('log_guest.txt', 1)}秒")
    print(f"guest上传输以外的时间为{sum_column('log_guest.txt', 2)}秒")
    print(f"host上forward函数中，除了传输和等待guest的时间为{sum_column('log_host.txt', 1)}秒")
    print(f"host等待guest的总时间为{sum_column('log_host.txt', 2)}秒")

    tensor2bytes_time = sum_column('log_tensor2bytes.txt', 0)
    tensor_bytes_and_module_name2blocks_time = sum_column('log_tensor_bytes_and_module_name2blocks.txt', 0)
    blocks2tensor_bytes_and_module_name_time = sum_column('log_blocks2tensor_bytes_and_module_name.txt', 0)
    bytes2tensor_time = sum_column('log_bytes2tensor.txt', 0)
    print(f"tensor2bytes的总时间为{tensor2bytes_time}秒")
    print(f"tensor_bytes_and_module_name2blocks的总时间为{tensor_bytes_and_module_name2blocks_time}秒")
    print(f"blocks2tensor_bytes_and_module_name的总时间为{blocks2tensor_bytes_and_module_name_time}秒")
    print(f"bytes2tensor的总时间为{bytes2tensor_time}秒")
    print(f"四个函数的总时间为{tensor2bytes_time + tensor_bytes_and_module_name2blocks_time + blocks2tensor_bytes_and_module_name_time + bytes2tensor_time}秒")
    print(f"host写完之后过了{sum_column_diff('log_host.txt', 'log_guest.txt', 3)}，guest开始读")
    print(f"guest写完之后过了{sum_column_diff('log_guest.txt', 'log_host.txt', 4)}，host开始读")


