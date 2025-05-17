# 文件组织

```txt
.
├── client
│   ├── client.py                  推理流程入口，读取config，非推理开始端先执行，
│   │                              运行时命令行参数指定host/guest
│   ├── config_guest.json
│   ├── config_host.json
│   └── __init__.py
├── core
│   ├── inference_engine.py        推理相关工具函数
│   ├── __init__.py
│   ├── ivshmem_comm.py            通信协议和控制流程相关代码
│   ├── llama_model.py             修改的模型定义文件，支持子模型推理
│   └── model_partition.py         模型权重文件切分代码，读取model/llama2下的文件，
│                                  切分后写入model/llama2-partitioned
├── model
│   ├── llama2                     模型原始权重文件，模型config，tokenizer所在
│   └── llama2-partitioned         切分的模型权重文件所在
├── README.md
├── scripts
│   ├── clear_ivshmem.sh           每次推理完成后清理ivshmem的内容
│   ├── create_vm.sh               创建支持SEV的虚拟机
│   ├── start_vm.sh                启动带有ivshmem的虚拟机
│   └── update_code.sh             更新模型文件到虚拟机
└── test                           无关紧要的小测试代码
    ├── __init__.py
    ├── test_gpu.py
    ├── test_load_model.py
    └── test_transfer.py
```

# 搭建环境时的注意事项

### 虚拟机ssh配置

物理机往虚拟机传文件，在虚拟机上开ssh之后，给qemu配置端口转发，具体地，在启动虚拟机配置文件中`netdev user`这一行，即网络配置，加上`,hostfwd=tcp::2222-:22`，把guest的22端口转发到host的2222端口，然后即可使用scripts/update_code.sh脚本。

### 虚拟机连不上网

虚拟机初始状态缺乏DNS解析，解决方法是:
1. 给/etc/netplan/00-config.yaml添加

```yaml
network:
  version: 2
  renderer: networkd       # 使用 systemd-networkd
  ethernets:
    enp0s3:
      dhcp4: true         # 启用 DHCPv4
      dhcp6: false        # 禁用 IPv6（可选）
```

然后运行

```bash
netplan apply
systemctl restart systemd-networkd
```

2. 给/etc/systemd/resolved.conf添加

```ini
DNS=114.114.114.114 8.8.8.8
FallbackDNS=1.1.1.1
Domains=~.
```

### 杂项

1. 使用的Ubuntu 24.04，python和pip环境需要用venv，在root账户下具体操作：

```bash
apt install python3.12-venv
python3 -m venv /root/pytorch-env
source /root/pytorch-env/bin/activate # 激活虚拟环境
deactivate # 退出虚拟环境
```