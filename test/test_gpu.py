import torch
import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 启用同步错误报告
os.environ['CUDA_VERBOSE_LAUNCH'] = "1"   # 显示内核启动细节
def test_gpu(device_id=0):
    try:
        # 初始化设备
        device = torch.device(f'cuda:{device_id}')
        print(f"\n===== 测试 GPU {device_id} =====")

        # 1. 基础 CUDA 可用性检查
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        print(f"[√] CUDA 可用性检查通过")

        # 2. 显存分配测试（阶梯式压力测试）
        print("\n--- 显存分配测试 ---")
        total_mem = torch.cuda.get_device_properties(device).total_memory
        print(f"总显存: {total_mem / 1024**3:.2f} GB")

        chunk_size = 256 * 1024**2  # 从 256MB 开始测试
        allocated = 0
        while True:
            try:
                # 尝试分配显存块
                block = torch.empty(chunk_size, dtype=torch.uint8, device=device)
                allocated += chunk_size
                print(f"已分配: {allocated / 1024**3:.2f} GB", end='\r')
                chunk_size += 256 * 1024**2  # 每次增加 256MB
            except RuntimeError as e:
                print(f"\n[!] 显存分配失败（预期行为）: {str(e)}")
                break

        # 3. 计算正确性验证（矩阵乘法）
        print("\n--- 计算正确性测试 ---")
        size = 5  # 大矩阵维度
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        cpu_result = torch.matmul(a.cpu(), b.cpu())
        if torch.allclose(c.cpu(), cpu_result, atol=1e-4):
            print("[√] 矩阵乘法验证通过")
        else:
            raise RuntimeError("GPU 计算结果与 CPU 不一致")

        # 4. 多卡通信测试（如果可用）
        if torch.cuda.device_count() > 1:
            print("\n--- 多卡通信测试 ---")
            other_device = torch.device("cuda:1" if device_id == 0 else "cuda:0")
            data = torch.randn(10000, 10000, device=device)
            data_copy = data.to(other_device)
            if torch.allclose(data, data_copy.to(device)):
                print("[√] 跨卡数据传输验证通过")
            else:
                raise RuntimeError("跨卡数据不一致")

        # 5. 性能基准测试
        print("\n--- 性能基准测试 ---")
        from time import time
        iterations = 100
        start_time = time()
        for _ in range(iterations):
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            torch.matmul(a, b)
        elapsed = time() - start_time
        print(f"完成 {iterations} 次矩阵乘法耗时: {elapsed:.2f} 秒")
        print(f"平均每次计算耗时: {elapsed/iterations*1000:.2f} ms")

        print("\n===== 所有测试通过 =====")

    except Exception as e:
        print(f"\n[!] 测试失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 测试所有可用 GPU
    for device_id in range(torch.cuda.device_count()):
        test_gpu(device_id)