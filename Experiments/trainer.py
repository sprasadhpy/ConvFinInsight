import os
import json
import random
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B",
    max_seq_length=2048,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
data_path = "data/train_turn.json"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    try:
        full_data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON format in dataset: {e}")
valid_data = [
    turn for turn in full_data
    if isinstance(turn, dict)
    and "qa" in turn
    and isinstance(turn["qa"], dict)
    and "question" in turn["qa"]
    and "answer" in turn["qa"]
    and isinstance(turn["qa"]["question"], str)
    and isinstance(turn["qa"]["answer"], str)
]
if not valid_data:
    raise ValueError("No valid QA entries found in the dataset.")
def flatten_table(table):
    if not table or not isinstance(table, list) or not all(isinstance(row, list) for row in table):
        return ""
    headers, *rows = table
    return "\n".join([" | ".join(headers)] + [" | ".join(row) for row in rows])
def make_prompt(turn, use_math=True):
    if not isinstance(turn, dict):
        raise TypeError("Each turn must be a dictionary.")
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table = flatten_table(turn.get("table", []))
    question = turn.get("qa", {}).get("question", "").strip()
    if not question:
        raise ValueError("Missing question in turn.")
    instruction = "You are a financial analyst assistant. Use the context to answer the question clearly and enclose the answer in <ANSWER>...</ANSWER>."
    if use_math:
        instruction += " If you do calculations, also enclose expressions in <MATH>...</MATH>."
    return f"""{instruction}
Context:
{pre}
Table:
{table}
{post}
Question: {question}
Answer:"""
def prepare_sft_dataset(data, use_math=True):
    if not isinstance(data, list):
        raise TypeError("Data must be a list of turn dictionaries.")
    examples = []
    for turn in data:
        try:
            prompt = make_prompt(turn, use_math=use_math)
            answer = turn["qa"]["answer"].strip()
            full_text = f"{prompt} <ANSWER>{answer}</ANSWER>"
            examples.append({"text": full_text})
        except Exception:
            continue
    return examples
sft_examples = prepare_sft_dataset(valid_data, use_math=True)
if not sft_examples:
    raise ValueError("No SFT examples generated.")
dataset = Dataset.from_list(sft_examples)
dataset = dataset.train_test_split(test_size=0.1)
if "train" not in dataset or "test" not in dataset:
    raise KeyError("Failed to split dataset into train and test.")
train_size = min(len(dataset["train"]), 1000)
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(train_size))
training_args = TrainingArguments(
    output_dir="./qwen3-sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=True,
    report_to="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_seq_length=2048,
    args=training_args,
)
trainer.train()
trainer.save_model("./qwen3-sft-final")
tokenizer.save_pretrained("./qwen3-sft-final")
