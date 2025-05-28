import os
import json
import re
import time
import random
import pandas as pd
from IPython.display import display
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from dotenv import load_dotenv
load_dotenv()
required_vars = [
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT"
]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, callbacks=[tracer])

@traceable(name="Ask ConvFinQA")
def ask_gpt(prompt):
    return llm([HumanMessage(content=prompt)]).content.strip()
data_path = "data/train_turn.json"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Could not find data file at: {data_path}")

with open(data_path, "r") as f:
    full_data = json.load(f)

valid_data = [t for t in full_data if "qa" in t and "question" in t["qa"] and "answer" in t["qa"]]
print(f"Total entries: {len(full_data)} | Valid QA entries: {len(valid_data)}")

sampled_data = random.sample(valid_data, min(200, len(valid_data)))
#### Guardrails fn defined from Agentic standards 
def apply_guardrails(turn):
    issues = []
    question = str(turn.get("qa", {}).get("question", "")).strip()
    answer = str(turn.get("qa", {}).get("answer", "")).strip()
    steps = turn.get("qa", {}).get("steps", [])
    raw_reasoning = str(turn.get("reasoning", "") or turn.get("steps", "")).strip()
    if not question:
        issues.append("Missing question")
    if not answer:
        issues.append("Missing answer")
    if not steps and not raw_reasoning:
        issues.append("Missing reasoning steps")
    if "%" in answer and not any("divide" in (step.get("op", "") or "").lower() for step in steps):
        issues.append("Percentage answer without division step")
    if raw_reasoning and ("→" not in raw_reasoning or "(" not in raw_reasoning):
        issues.append("Ill-formatted reasoning steps")
    return issues
def flatten_table(table):
    headers = table[0]
    rows = table[1:]
    return "\n".join([" | ".join(headers)] + [" | ".join(row) for row in rows])
def make_prompt(turn, use_math=False):
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table_text = flatten_table(turn["table"])
    question = turn["qa"]["question"]
    instructions = "You are a financial analyst assistant. Use the context below to answer the question."
    instructions += " If calculations are needed, enclose math in <MATH>...</MATH> and result in <ANSWER>...</ANSWER>." if use_math else " Provide final answer in <ANSWER>...</ANSWER>."
    return f"""{instructions}
Context:
{pre}
Table:
{table_text}
{post}
Question: {question}
Answer:"""
def extract_math_answer(text):
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()
def extract_math_expression(text):
    m = re.search(r"<MATH>(.*?)</MATH>", text, re.DOTALL)
    return m.group(1).strip() if m else None
def evaluate_expression(expr):
    try:
        return eval(expr, {"__builtins__": {}})
    except:
        return None
def extract_number(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None
def score_with_message(pred, gold, rel_tol=0.03, abs_tol=0.05):
    if pred is None or gold is None:
        return 0, "Missing number"
    err = abs(pred - gold)
    return (1, "Match") if err <= max(rel_tol * abs(gold), abs_tol) else (0, f"No match (diff {err:.4f})")
results = []
for i, turn in enumerate(sampled_data):
    guardrail_issues = apply_guardrails(turn)
    if guardrail_issues:
        print(f"Skipped Q{i+1} due to guardrail violations: {guardrail_issues}")
        continue
    gold = extract_number(turn["qa"]["answer"])
    conv_type = turn.get("conv_type", "unknown")
    prompt1 = make_prompt(turn, use_math=False)
    t0 = time.time()
    gen1 = ask_gpt(prompt1)
    t1 = time.time()
    time.sleep(1.5)
    pred1 = extract_number(extract_math_answer(gen1))
    score1, msg1 = score_with_message(pred1, gold)
    prompt2 = make_prompt(turn, use_math=True)
    t2 = time.time()
    gen2 = ask_gpt(prompt2)
    time.sleep(1.5)
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
    })
results_df = pd.DataFrame(results)
results_df.to_csv("method_comparison_guarded.csv", index=False)
print("saved evaluation to method_comparison_guarded.csv")
