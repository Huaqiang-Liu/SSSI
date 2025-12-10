from datasets import load_dataset

# 1. 获取 SST-2 的一个句子（输入 Prompt）
print("--- 1. SST-2 (Sentiment Classification) 示例 ---")
# 加载 GLUE 的 SST2 配置
sst2_dataset = load_dataset("glue", "sst2", split="train")
# 打印第 0 条数据（电影评论）
sst2_example = sst2_dataset[0]
print(f"输入句子 (Prompt): '{sst2_example['sentence']}'")
print(f"原始标签 (Label): {sst2_example['label']} (0=negative, 1=positive)")

print("\n--- 2. MNLI (Natural Language Inference) 示例 ---")
# 加载 GLUE 的 MNLI 配置
mnli_dataset = load_dataset("glue", "mnli", split="train")
# 打印第 0 条数据（句子对）
mnli_example = mnli_dataset[0]
print(f"前提句 (Premise): '{mnli_example['premise']}'")
print(f"假设句 (Hypothesis/Prompt): '{mnli_example['hypothesis']}'")
# NLI 的输入是 (Premise, Hypothesis) 句子对

print("\n--- 3. SQuAD 1.0 (Extractive QA) 示例 ---")
# 加载 SQuAD 1.0
squad_dataset = load_dataset("squad", split="train")
# 打印第 0 条数据（Context 和 Question）
squad_example = squad_dataset[0]
print(f"段落 (Context): '{squad_example['context'][:300]}...'")
print(f"问题 (Question/Prompt): '{squad_example['question']}'")
print(f"预期答案 (Answer): '{squad_example['answers']['text'][92]}'")
# QA 的输入是 (Context, Question) 句子对，通常 Question 是 Prompt