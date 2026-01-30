# 基于大语言模型与证据理论融合的智能制造轴承故障诊断系统

1. 对本系统的介绍
2. 数据集构建
3. 训练参数
4. 文件介绍
5. Autodl下载大模型教程配置
6. Colab下的CUDA配置和Pytorch
7. ollama本地部署.gguf格式Model

> Colab下的CUDA配置和Pytorch是完成作业过程中遇到的问题，因为Colab免费的GPU显存为16GB，训练过程中遇到CUDA out of momery,训练小模型可以尝试。

## 一、系统简要介绍

### 1.主要任务

依据电机/风机等传感器振动信号数据，结合**Dempster–Shafer 证据理论（DST）**和**大语言模型**完成：

​	**健康阶段判别（Stage）**：**Normal / Degrading / Severe**

​	**故障类型识别（Fault Type）**：**正常（Normal）**/**外圈故障（Outer race）**/**内圈故障（Inner race）**/**滚动体故障（Ball/Roller）**

并且给出**关键证据 / 建议动作 / 不确定性与下一步**

------

### 2.整体方法简略介绍

1) 用 **IMS run-to-failure、CWRU、XJTU-SY** 三套公开数据构建统一数据流水线（统一采样率/窗口切片/特征与z-score）。
2) 将样本整理成 **Stanford Alpaca** 指令格式，用**LORA微调**更新模型1%-10%参数，使用 **SFTTrainer** 对大模型进行监督微调（SFT），使其能输出结构化诊断报告。
3) 推理阶段引入 **Dempster–Shafer 证据理论（DST）** 融合多视角证据（时域/频域），输出 `BetP / 冲突度K / 无知度m(Θ)` 等可信度指标。
4) 传感器数据输入，将“特征 + DST融合结果”一并输入 LLM，生成四段式报告：**结论 / 关键证据 / 建议动作 / 不确定性与下一步**。

​	**系统整体流程图**如下：
<img width="1447" height="842" alt="系统总体流程图" src="https://github.com/user-attachments/assets/9d3efe8c-3749-4750-9cf4-d7880f0683c6" />

### 3.训练前后大模型对同一输入的回答对比
**训练前**
<img width="1155" height="555" alt="{B272CE0E-D62D-4F64-871F-C207F407AF06}" src="https://github.com/user-attachments/assets/63ca1e78-4d4d-4f40-a00a-555cd4a172e0" />
**本文的系统回答**
<img width="1702" height="953" alt="{3876501C-CEAD-4049-BCD6-C491E425D31A}" src="https://github.com/user-attachments/assets/3955ef59-70e2-4733-b134-9da39414f1c4" />

## 二、数据集介绍及数据集构造处理

### A：四分类（Normal/Inner/Outer/Ball）的振动主数据集

#### cwru_alpaca_format

- CWRU Bearing Data Center（振动）**：明确包含内圈/外圈/滚动体（ball）等单点故障，适合做四分类 baseline 与特征验证。[Case School of Engineering+1](https://engineering.case.edu/bearingdatacenter)
  [Normal Baseline Data](https://engineering.case.edu/bearingdatacenter/normal-baseline-data)
  [48k Drive End Bearing Fault Data](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)

### B：退化/“在变坏”（run-to-failure）数据集

> 运行处理程序后，请将生成的baseline_by_bearing.json移除，否则读取数据时候会报错

要让 B 输出“变坏趋势/严重度/提前预警”，核心就是 **run-to-failure**

#### XJTU-SY_Bearing_Datasets

15 套轴承完整跑到失效，3 种工况）GitHub](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets)

#### IMS Bearings

（NASA/UC Cincinnati，跑到失效）经典退化与早预警基准（适合做趋势异常、提前量评估）。[NASA开放数据门户](https://data.nasa.gov/dataset/ims-bearings)

## 三、训练参数

> ./ModelLoad.ipynb中为训练模型过程，这里只展示部分。主要使用Unsloth库

**LoRA微调**

```python
# LoRA进行微调更新模型1%-10%参数
"""
LoRA:LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
矩阵A使用高斯初始化,先降维,矩阵B使用全0初始化再升维,维度控制参数是矩阵的秩r,一般为1,6,8,16
"""
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout= 0,
    bias  = "none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
```

SFTTrainer参数

```python
#设置训练参数
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    eval_dataset= eval_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,#指定处理数据时使用的并行进程数，以加快数据预处理速度。
    packing = False, # 可以让短序列的训练速度提高5倍。
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 16,
        warmup_steps = 5,
        max_steps = 60,  # 微调步数
        learning_rate = 2e-4, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        per_device_eval_batch_size=2,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)
```

## 四、项目文件介绍

### 1.数据集处理(/Pretrain/Data_Hading)

#### A

对CWRU数据集进行处理

1) train_alpaca.jsonl

- 训练集（用来做 LoRA/SFT 微调）
- 每一行是一条 JSON（Alpaca 样本）：
  - instruction：系统约束 + 任务要求（四段式输出、禁止编造等）
  - input：该切片的上下文（采样率、窗口、工况、特征、相对正常基线的 z-score）
  - output：标准答案（结论/关键证据/建议动作/不确定性与下一步）
  - id：样本编号

2) val_alpaca.jsonl

- 验证集（训练中用来做 eval / 早停 / 调参）

3) test_alpaca.jsonl

- **测试集**（训练完最终评估用，不建议参与训练/调参）
- 格式同上

4) alpaca_all.json

- **同一批数据的“单文件合集版”**
- 不是 jsonl（不是一行一个样本），而是一个 **JSON 数组**：`[ {...}, {...}, ... ]`

#### B

##### IMS.py

将ims数据集重采样48k，转换为alpaca格式

IMS 数据文件，目录结构类似：

```
IMS_DATA/
  1st_test/
    2003.10.22.12.06.24
    2003.10.22.12.06.25
    ...
  2nd_test/
    2004.02.12.10.32.39
    ...
  3rd_test/
    2004.03.04.09.27.46
    ...
```

运行**Pretrain/Data_Handing/B/IMS.py**

会生成：

```
ims_alpaca_48k_out/
  1st_test_train_alpaca.jsonl
  1st_test_val_alpaca.jsonl
  1st_test_test_alpaca.jsonl
  1st_test_baseline.json
  1st_test_segments_metadata.csv
  ...
```

<img width="1292" height="140" alt="image-20260119210825634" src="https://github.com/user-attachments/assets/92af5176-f7bb-4e2e-b2e8-1d79bb608292" />


##### XJTU.py

运行XJTU.py

输出目录 `xjtu_c3_alpaca_48k_out/` 里会有：

- `train_alpaca.jsonl` / `val_alpaca.jsonl` / `test_alpaca.jsonl`
  - 标准 Stanford Alpaca：每行一个 `{instruction,input,output,id}`
- `alpaca_all.json`
  - 把 train/val/test 合并成一个 JSON 数组版本
- `segments_metadata.csv`
  - 每条样本对应哪个 bearing / 哪个原始文件 / 哪个窗口（方便回溯）
- `baseline_by_bearing.json`
  - 每个 bearing 的早期健康基线（用于算 z-score）

<img width="1383" height="207" alt="image-20260123172643657" src="https://github.com/user-attachments/assets/aed90e98-da77-46df-91c0-48c16aa20e21" />


### 2.证据理论模块(./DST_Function)

对新输入片段提取特征与 z-score，分别由时域/频域轻量分类器（由train_stage_views_from_alpaca.py通过数据集训练）给出各健康阶段（或故障类型）的概率作为证据源，映射为BPA并进行可靠性折扣后用D-S规则融合，得到融合质量m(⋅)、冲突度 K、无知度 m(Θ) 以及用于决策的BetP；

#### /evidence

evidence/constants.py

evidence/ds_fusion.py（证据理论核心：prob→BPA→融合→BetP→证据块）

evidence/text_parse.py（解析文本输入的 z-score）

evidence/alpaca_loader.py（通用递归加载 IMS(1st/2nd/3rd)/XJTU/CWRU）

evidence/prob_sources_model.py（小分类器证据源）

#### /scripts

scripts/infer_stage_funsion_only.py（输入只经过证据理论模块的输出)

scripts/infer_stage_funsion_unsloth.py（输入经过证据理论模块以及大语言模型，系统的最终功能)

scripts/train_stage_views_from_alpaca.py（训练两个视角小分类器：IMS+XJTU）

> 会输出：
> `models/stage_time.joblib`
> `models/stage_freq.joblib`

### 3.Pretrain下其他文件

#### bisect_test.py 

检验unsloth是否有warning

#### check_torch.py

查看Pytorch是否是GPU版

#### Data_Load.py

加载训练数据集的方法，在Test_ModelLoad.py中调用

#### Test_ModelLoad.py

测试模型训练

### 4.ModelLoad.ipynb

模型训练的脚本

## 五.模型选择

### 1.基础通识大模型

选择Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning

下载地址为：
[DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning · HF Mirror（国内镜像）](https://hf-mirror.com/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning)

[DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning · Hugging Face（国外原址）](https://huggingface.co/DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning)

### 2.ollama部署.gguf格式模型方法 

> ollama部署需要.gguf格式的model文件

1. 选择**ollama**进行部署，**Model Location**中选择模型所在文件夹。

 <img width="1669" height="1016" alt="image-20260108183636505" src="https://github.com/user-attachments/assets/de9fbe8e-39c3-42a3-9998-1d9c50c41c0a" />

2. 创建**ModelFile**文件夹，在ModelFile文件夹中创建文件，例如**modelload.txt**，里面填写 **FROM 下载的模型文件名称**

   ```
   FROM MODEL.BF16.gguf
   ```

   > 该文件中还可以填写**系统提示词**，将**系统提示词固化**

   **ModelFile**文件夹中还保存有下载的模型文件，整体如下图

  <img width="760" height="314" alt="image-20260108184630852" src="https://github.com/user-attachments/assets/09dc35e4-f640-48be-a358-b068a046d4b4" />

   <img width="874" height="330" alt="image-20260129203147360" src="https://github.com/user-attachments/assets/31ac47a9-64f0-44ae-acb6-d7faaa8dbd7d" />

3. **ollama加载模型**

  ModelFile文件夹中右键空白位置，选择在终端打开

   ```python
   ollama create IBDF -f ./ModelFile
   ```

   > create后是模型的别名，自己命名

  <img width="1134" height="244" alt="image-20260129203332731" src="https://github.com/user-attachments/assets/2d8b9dfe-7afa-459a-968d-834371ad7424" />
  
    
    ollama list
    

  即可查看是否成功

  <img width="665" height="106" alt="image-20260129203353334" src="https://github.com/user-attachments/assets/238e5bc9-5272-4565-8646-b748fde0bfb3" />

4. **网页运行(可选)**

   因为ollama软件即可有对话框，这步可以选择进行。

   安装插件后点击，会自动加载ollama。

   [Page Assist - A Web UI for Local AI Models - Microsoft Edge Addons](https://microsoftedge.microsoft.com/addons/detail/page-assist-a-web-ui-fo/ogkogooadflifpmmidmhjedogicnhooa?hl=zh-CN)

## 六、unsloth安装

```python
pip install "unsloth[cu128-ampere-torch280] @ git+https://github.com/unslothai/unsloth.git"
```

注意

```
pip install unsloth
```

会自动安装cpu版本的PyTorch(且是最新版),后续更换为GPU版本的PyTorch较复杂

如想使用，参考以下步骤

使用清华源，就不用科学上网

```
pip install unsloth -i https://pypi.tuna.tsinghua.edu.cn/simple
```

删除CPU版pytorch

```
pip uninstall torch torchvision torchaudio -y
```

**安装 GPU 版本的 PyTorch (使用阿里云镜像或官方源)**

> 🚸 **版本选择至关重要：**
>
> - **Unsloth 兼容性：** 检查你安装的 Unsloth 版本对 PyTorch 的最低版本要求 (例如，`unsloth-2025.5.9` 可能需要 `torch>=2.4.0`)。
> - **xformers 兼容性：** 步骤2中与 Unsloth 一同安装的 `xformers` 版本 (例如 `xformers-0.0.30`) 通常与当时一同安装的 PyTorch CPU 版本 (例如 `torch-2.7.0`) 兼容。
> - **CUDA 版本：** 确保选择与你的 NVIDIA 驱动和本地 CUDA Toolkit 版本匹配的 PyTorch (例如 `cu118` 对应 CUDA 11.8)。
>
> **建议：** 尝试安装与步骤2中 Unsloth 初始依赖的 PyTorch 版本号相同，但带有正确 CUDA 后缀的 PyTorch。例如，如果初始安装了 `torch-2.7.0` (CPU)，则目标是安装 `torch==2.7.0` 的 `cu118` 版本。

## 七、Autodl下载Huggingface镜像网站的模型方法

huggingface-cli命令已经被废弃，提供的方法是参考这篇文章[http://www.mynw.cn/news/820073.html](http://www.mynw.cn/news/820073.html#:~:text=bash%3A huggingface-cli%3A command not found 错误的根本原因是 huggingface-hub 1.0.0,及以上版本废弃了原有的 huggingface-cli 命令。 解决方案是通过升级或安装最新的 huggingface-hub，并使用新的 hf download 命令来下载模型，同时可设置镜像源（HF_ENDPOINT）加速下载。)文章中方法也有问题，使用我的方法可以运行。

**终端运行以下命令**

```
#安装依赖：
pip install -U huggingface_hub
#设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
#下载( huggingface-cli命令已经被废弃)
hf download DavidAU/Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning --local-dir /root/autodl-tmp/LLM/MODEL
```

> download后是**模型名称**，--local--dir后是**想要下载的路径**
>
> 下载模型过程中可能会多次Error，重复执行命令即可！！！可能会缺失文件！！！模型中较小文件可以在命令执行完之后，看缺少什么，自行下载后上传，否则会报错

<img width="679" height="144" alt="image-20260127130854418" src="https://github.com/user-attachments/assets/740dba9f-fece-4e5c-a49e-9bf57f1a191d" />

## 八、Colab中CUDA配置和Pytorch

> Colab左侧上传的文件会在关闭后消失！！！
可以参考这篇文章[在google colab上搭建pytorch深度学习环境_colab pytorch-CSDN博客](https://blog.csdn.net/qq_35644010/article/details/136110052)，但是安装CUDA和cudnn可以按我的来(从官网复制命令即可)
```
#查看Colab的Cuda配置
!nvcc --version
```

```
!apt-get --purge remove cuda nvidia* libnvidia-*
!dpkg -l | grep cuda- | awk '{print $2}' | xargs -n1 dpkg --purge
!apt-get remove cuda-*
!apt autoremove!apt-get update
```

```
#查看Ubuntu版本
!lsb_release -a
!apt autoremove
```

```
#安装CUDA
!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
!sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
!wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
!sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
!sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update
!sudo apt-get -y install cuda-toolkit-12-8
```
找到对应版本的命令复制运行即可，$要改成!哦
<img width="2052" height="1185" alt="{0C9914B3-4916-4B1F-A410-9D4B7765CA03}" src="https://github.com/user-attachments/assets/34ed4c37-b15f-4799-a5fd-ab619a367295" />
```
#安装cudnn
!wget https://developer.download.nvidia.com/compute/cudnn/9.18.1/local_installers/cudnn-local-repo-ubuntu2204-9.18.1_1.0-1_amd64.deb
!sudo dpkg -i cudnn-local-repo-ubuntu2204-9.18.1_1.0-1_amd64.deb
!sudo cp /var/cudnn-local-repo-ubuntu2204-9.18.1/cudnn-*-keyring.gpg /usr/share/keyrings/
!sudo apt-get update
!sudo apt-get -y install cudnn
```

```
#查看Colab的Cuda配置，是否安装成功
!nvcc --version
```

```
#安装Pytorch(GPU)
!pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

