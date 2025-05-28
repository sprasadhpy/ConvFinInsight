import os
import json
import re
import time
import random
import pandas as pd
import torch
from pathlib import Path
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import pipeline
from data.loader import load_data
load_dotenv()
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
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
def ask_qwen(prompt):
    out = pipe(prompt, return_full_text=False)[0]["generated_text"]
    return out.strip()
try:
    full_data = load_data("train_turn")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")
valid_data = [turn for turn in full_data if "qa" in turn and "question" in turn["qa"] and "answer" in turn["qa"]]
N = 50
sampled_data = random.sample(valid_data, N)
def flatten_table(table):
    headers = table[0]
    rows = table[1:]
    lines = [" | ".join(headers)] + [" | ".join(row) for row in rows]
    return "\n".join(lines)
def make_prompt(turn, use_math=False):
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table = flatten_table(turn["table"])
    q = turn["qa"]["question"]
    instruction = "You are a financial analyst assistant. Use the context to answer the question clearly and enclose the answer in <ANSWER>...</ANSWER>."
    if use_math:
        instruction += " If you do calculations, also enclose expressions in <MATH>...</MATH>."
    return f"""{instruction}
Context:
{pre}
Table:
{table}
{post}
Question: {q}
Answer:"""
def extract_math_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def extract_math_expr(text):
    match = re.search(r"<MATH>(.*?)</MATH>", text, re.DOTALL)
    return match.group(1).strip() if match else None
def evaluate_expression(expr):
    try:
        return eval(expr, {"__builtins__": {}})
    except:
        return None
def extract_number(text):
    try:
        nums = re.findall(r"-?\d+\.?\d*", text)
        return float(nums[-1]) if nums else None
    except:
        return None
def score(pred, gold, rel_tol=0.03, abs_tol=0.05):
    if pred is None or gold is None:
        return 0, "Missing number"
    try:
        err = abs(pred - gold)
        if err <= max(rel_tol * abs(gold), abs_tol):
            return 1, "Match"
        return 0, f"Diff {err:.4f}"
    except:
        return 0, "Invalid comparison"
results = []
for i, turn in enumerate(sampled_data):
    gold = extract_number(turn["qa"]["answer"])
    prompt1 = make_prompt(turn, use_math=False)
    t0 = time.time()
    gen1 = ask_qwen(prompt1)
    t1 = time.time()
    pred1 = extract_number(extract_math_answer(gen1))
    s1, m1 = score(pred1, gold)
    prompt2 = make_prompt(turn, use_math=True)
    t2 = time.time()
    gen2 = ask_qwen(prompt2)
    expr = extract_math_expr(gen2)
    calc = evaluate_expression(expr) if expr else None
    pred2 = calc if calc is not None else extract_number(extract_math_answer(gen2))
    s2, m2 = score(pred2, gold)
    t3 = time.time()
    results.append({
        "question": turn["qa"]["question"],
        "gold": gold,
        "pred_qwen": pred1,
        "score_qwen": s1,
        "match_qwen": m1,
        "latency_qwen": t1 - t0,
        "pred_qwen_calc": pred2,
        "score_qwen_calc": s2,
        "match_qwen_calc": m2,
        "latency_qwen_calc": t3 - t2,
        "expr": expr,
        "gen_qwen": gen1,
        "gen_qwen_calc": gen2,
    })
    print(f"\nQ{i+1}: {turn['qa']['question']}")
    print("\nQwen (LLM Only)")
    print(f"Predicted:\n{gen1}\nExpected: {turn['qa']['answer']}")
    print(m1)
    print("\nQwen + Calculator")
    print(f"Predicted:\n{gen2}\nExpected: {turn['qa']['answer']}")
    print(m2)
Path("outputs").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv("outputs/qwen_method_comparison_results.csv", index=False)
acc_qwen = sum(r["score_qwen"] for r in results) / N
acc_qwen_calc = sum(r["score_qwen_calc"] for r in results) / N
print(f"\nAccuracy Qwen LLM-only: {acc_qwen*100:.1f}%")
print(f"Accuracy Qwen + Calculator: {acc_qwen_calc*100:.1f}%")
