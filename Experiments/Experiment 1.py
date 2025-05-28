
import os
import json
import re
import time
import random
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from data.loader import load_data

# I load API keys and other secrets from a .env file

load_dotenv()
#  track all LLM calls for debugging and review
tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, callbacks=[tracer])
@traceable(name="Ask ConvFinQA")
# this is my main helper to call the model and get a clean string response
def ask_gpt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()
try:
    full_data = load_data("train_turn")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")
# I only want samples that actually have a question and an answer
valid_data = [
    turn for turn in full_data
    if isinstance(turn, dict) and "qa" in turn and "question" in turn["qa"] and "answer" in turn["qa"]
]
print(f"Total entries: {len(full_data)} | Valid QA entries: {len(valid_data)}")

N = 200
# I’ll randomly sample N examples to run this eval 200 
if len(valid_data) < N:
    raise ValueError(f"Insufficient valid QA data: only {len(valid_data)} available.")
sampled_data = random.sample(valid_data, N)

def flatten_table(table: list) -> str:
    if not table or not isinstance(table, list) or not all(isinstance(row, list) for row in table):
        return ""
    headers = table[0]
    rows = table[1:]
    lines = [" | ".join(map(str, headers))]
    lines += [" | ".join(map(str, row)) for row in rows]
    return "\n".join(lines)
# This builds my prompt depending on whether I want the LLM to show math
def make_prompt(turn: dict, use_math: bool = False) -> str:
    if not isinstance(turn, dict):
        raise TypeError("Turn must be a dictionary.")
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table_text = flatten_table(turn.get("table", []))
    question = turn.get("qa", {}).get("question", "")
    instructions = "You are a financial analyst assistant. Use the context below to answer the question."
    if use_math:
        instructions += " If calculations are needed, enclose the math expression in <MATH>...</MATH> and the result in <ANSWER>...</ANSWER>."
    else:
        instructions += " Directly provide the final answer in <ANSWER>...</ANSWER>."
    return f"{instructions}\n\nContext:\n{pre}\n\nTable:\n{table_text}\n\n{post}\n\nQuestion: {question}\nAnswer:"
# I pull out the <ANSWER> from the LLM's response or just return the raw text
def extract_math_answer(text: str) -> str:
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def extract_math_expression(text: str) -> str:
    match = re.search(r"<MATH>(.*?)</MATH>", text, re.DOTALL)
    return match.group(1).strip() if match else None
def evaluate_expression(expr: str) -> float:
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None
# This lets me pull a number from a text blob I'll just take the last number
def extract_number(text: str) -> float:
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None
def score_with_message(pred: float, gold: float, rel_tol: float = 0.03, abs_tol: float = 0.05) -> tuple[int, str]:
    if pred is None or gold is None:
        return 0, "Unable to compare (missing number)"
    err = abs(pred - gold)
    if err <= max(rel_tol * abs(gold), abs_tol):
        return 1, "Match (numeric)"
    return 0, f"No match (diff {err:.4f})"
results = []
for i, turn in enumerate(sampled_data):
    gold = extract_number(turn["qa"]["answer"])
    conv_type = turn.get("conv_type", "unknown")
    prompt1 = make_prompt(turn, use_math=False)
    t0 = time.time()
    gen1 = ask_gpt(prompt1)
    t1 = time.time()
    pred1 = extract_number(extract_math_answer(gen1))
    score1, msg1 = score_with_message(pred1, gold)
    # Second try Let LLM use <MATH> if it wants
    prompt2 = make_prompt(turn, use_math=True)
    t2 = time.time()
    gen2 = ask_gpt(prompt2)
    expr = extract_math_expression(gen2)
    calc_result = evaluate_expression(expr) if expr else None
    pred2 = calc_result if calc_result is not None else extract_number(extract_math_answer(gen2))
    score2, msg2 = score_with_message(pred2, gold)
    t3 = time.time()
    results.append({
        "question": turn["qa"]["question"],
        "conv_type": conv_type,
        "gold": gold,
        "pred_llm": pred1,
        "score_llm": score1,
        "match_llm": msg1,
        "latency_llm": t1 - t0,
        "pred_calc": pred2,
        "score_calc": score2,
        "match_calc": msg2,
        "latency_calc": t3 - t2,
        "expr": expr,
        "gen_llm": gen1,
        "gen_calc": gen2,
    }) #print out intermediate results
    print(f"\nQ{i+1} [{conv_type}] {turn['qa']['question']}")
    print("LLM Only")
    print(f"Predicted:\n{gen1}\nExpected: {turn['qa']['answer']}")
    print(msg1)
    print("LLM + Calculator")
    print(f"Predicted:\n{gen2}\nExpected: {turn['qa']['answer']}")
    print(msg2)
df = pd.DataFrame(results)
for ctype in df["conv_type"].unique():
    subset = df[df["conv_type"] == ctype]
    acc1 = subset["score_llm"].mean() * 100
    acc2 = subset["score_calc"].mean() * 100
    print(f"\nAccuracy for '{ctype}': LLM only = {acc1:.1f}% | LLM+Calculator = {acc2:.1f}%")
df.to_csv("method_comparison_classified_200.csv", index=False)
