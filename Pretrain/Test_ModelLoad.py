from unsloth import FastLanguageModel
import torch
max_seq_length = 2048
dtype = None    #自动发现需要的数据类型
load_in_4bit = True #4位量化，缩小模型容量。但会牺牲精度(资源有限)
# 使用llama 3.1 8B 
model,tokenizer = FastLanguageModel.from_pretrained(
    model_name="./MODEL/",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print(model)
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

alpaca_prompt = """你是工业轴承故障诊断助手。
你只能基于输入的特征与z-score推理，禁止编造未提供的信息。
输出必须严格包含四段：结论、关键证据、建议动作、不确定性与下一步。

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

# FastLanguageModel.for_inference(model)
# inputs = tokenizer(
#     [
#         alpaca_prompt.format(
#             """你是工业轴承故障诊断助手。
# 你只能基于输入的特征与z-score推理，禁止编造未提供的信息。
# 输出必须严格包含四段：结论、关键证据、建议动作、不确定性与下一步。""",#instruction
# """特征（z-score）：
# - rms_mean=0.182 (z=+2.60)
# - crest_mean=4.85 (z=+2.20)
# - kurt_mean=5.40 (z=+3.10)
# - hf_ratio_gt5k=0.058 (z=+2.70)
# - band_8_20k=0.10 (z=+3.00)

# 任务：判断健康阶段（Normal/Degrading/Severe）并输出四段式报告。""",#input
# "",#output -leave this blank for generation!
#         )
#     ],return_tensors="pt").to("cuda")
# output = model.generate(**inputs,max_new_tokens=256,use_cache=True)
# tokenizer.batch_decode(output)
# print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])

from Data_Load import load_alpaca_any
from datasets import concatenate_datasets

ims  = load_alpaca_any(r"./Data/IMS/ims_alpaca_48k_out/")
cwru = load_alpaca_any(r"./Data/CWRU Bearing Data Center/")
xjtu = load_alpaca_any(r"./Data/XJTU-SY_Bearing_Datasets/xjtu_c3_alpaca_48k_out/")

train_ds = concatenate_datasets([ims["train"], cwru["train"], xjtu["train"]]).shuffle(seed=42)
# validation 可能某个数据集没有，就只拼存在的
eval_parts = []
for ds in [ims, cwru, xjtu]:
    if "validation" in ds:
        eval_parts.append(ds["validation"])
eval_ds = concatenate_datasets(eval_parts).shuffle(seed=42) if eval_parts else None
train_ds = train_ds.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_ds.column_names)
eval_ds = eval_ds.map(
    formatting_prompts_func,
    batched=True,
     remove_columns=eval_ds.column_names)
print("train =", len(train_ds))
print("eval  =", len(eval_ds) if eval_ds else None)


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported




trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,#指定处理数据时使用的并行进程数，以加快数据预处理速度。
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        max_steps = 60,  # 微调步数
        learning_rate = 2e-4, # 学习率
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

FastLanguageModel.for_inference(model)
inputs = tokenizer(
    [
        alpaca_prompt.format(
            """你是工业轴承故障诊断助手。
你只能基于输入的特征与z-score推理，禁止编造未提供的信息。
输出必须严格包含四段：结论、关键证据、建议动作、不确定性与下一步。""",#instruction
"""特征（z-score）：
- rms_mean=0.182 (z=+2.60)
- crest_mean=4.85 (z=+2.20)
- kurt_mean=5.40 (z=+3.10)
- hf_ratio_gt5k=0.058 (z=+2.70)
- band_8_20k=0.10 (z=+3.00)

任务：判断健康阶段（Normal/Degrading/Severe）并输出四段式报告。""",#input
"",#output -leave this blank for generation!
        )
    ],return_tensors="pt").to("cuda")
output = model.generate(**inputs,max_new_tokens=256,use_cache=True)
tokenizer.batch_decode(output)
print(tokenizer.batch_decode(output, skip_special_tokens=True)[0])
