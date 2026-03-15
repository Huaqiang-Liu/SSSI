<!-- # 🔧 Prerequisites
- **Python**: 3.12 or higher
- **Git**: For repository cloning and submodule management
- **Cuda** compilation tools: 12.0 or higher
- **Pytorch**: 2.7.0+cu128 or higher
- **QEMU** emulator: 8.2.2
- **Libvirt**: 10.0.0
- **Conda**: For environment management (recommended)

# 🚀 Set up
### 1. Clone the Repository
```shell
git clone https://github.com/SEC-bench/SEC-bench.git
cd pkus
```

### 2. Configure a VM with AMD-SEV support
- Create a qcow2-format virtual disk and modify the `disk` option in `scripts/create_vm.sh`
- Prepare the VM image and modify the `location` option in `scripts/create_vm.sh`

```shell
bash scripts/create_vm.sh
```

### 3. Start VM
Modify `hda` option and `drive` option in `scripts/start_vm.sh` and then run
```shell
bash scripts/start_vm.sh
```

### 4. Run the main program
```shell
python client.py --role=host
python client.py --role=guest
```


# 📣 Precautions
1. You can define the trace by changing the options at the beginning of client.py
2. Remember modifing the model path in client.py before running
3. When unexpected memory leak happens if you hack the code, use `scripts/clear_ivshmem.sh` to clear the IVSHMEM
4. You can use `scripts/update_code.sh` to easily synchronous your code in TEE. -->


# 搭建环境流程：
## 创建虚拟机
1. 安装和配置qemu
2. 创建`/var/lib/libvirt/images/ubuntu-sev.qcow2`虚拟磁盘镜像
3. 准备系统镜像`/var/lib/libvirt/images/ubuntu-24.04.2-live-server-amd64.iso`，使用`scripts/create_vm.sh`创建

## 运行虚拟机
运行`scripts/start_vm.sh`，如果其它终端里在使用虚拟机，忘记`shutdown -h now`了就关闭了该终端，就会显示磁盘镜像文件被占用。这时在root下（`sudo su`）运行`lsof /var/lib/libvirt/images/ubuntu-sev.qcow2`，查找到虚拟机管理进程，`kill -9 <pid>`即可

## 项目结构
有两种分层推理的方式。以下面的llama-3-1b模型为例：
```txt
PeftModel(
  (base_model): LoraModel(
    (model): LlamaForCausalLM(
      (model): LlamaModel(
        (embed_tokens): Embedding(128256, 2048)
        (layers): ModuleList(
          (0-15): 16 x LlamaDecoderLayer(
            (self_attn): LlamaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=512, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=512, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
            )
            (mlp): LlamaMLP(
              (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
              (up_proj): lora.Linear(
                (base_layer): Linear(in_features=2048, out_features=8192, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=8192, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=8192, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=8192, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (act_fn): SiLU()
            )
            (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
            (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
          )
        )
        (norm): LlamaRMSNorm((2048,), eps=1e-05)
        (rotary_emb): LlamaRotaryEmbedding()
      )
      (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
    )
  )
)
```
1. `client.py`是把模型按lora相关和非lora相关区分的。比如与`base_layer`处于同一层级的其它“细粒度”的层，就是lora相关的。运行的时候，非lora的在host上用GPU推理，lora的将输入传输到guest（虚拟机）上，用CPU推理之后将输出数据返回到host。
2. `client_coarse.py`顾名思义，粒度更粗，将模型分为Embedding，16 x LlamaDecoderLayer，norm，lm_head这几个部分，1+16+1+1中的每一层都可能在host/guest上推理
3. `scripts/update_code.sh`用来向虚拟机同步代码文件
4. 测试失败时，用于host-guest通信的虚拟内存中可能有残留数据，影响下一次测试，所以可以用`scripts/clear_ivshmem.sh`清空这片区域。注意仅在host上使用。
5. `ivshmem_comm.py`的内容是通信协议的代码。




## 测试运行
```sh
# 在root下，虚拟环境为/root/pytorch-env。在host和guest上同时运行
python client(-coarse).py --role=host或guest
```


