import os
import re
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from data.loader import load_data

load_dotenv()
tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, callbacks=[tracer])
@traceable(name="AskAgent")
def ask_gpt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    return llm([HumanMessage(content=prompt)]).content.strip()
def extract_math_answer(text: str) -> str:
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def extract_number(text: str):
    if not isinstance(text, str):
        return None
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None
def score(pred, gold, rel_tol=0.03, abs_tol=0.05):
    if pred is None or gold is None:
        return 0, "Missing number"
    err = abs(pred - gold)
    if err <= max(rel_tol * abs(gold), abs_tol):
        return 1, "Correct"
    return 0, f"Incorrect (diff {err:.4f})"
Path("outputs").mkdir(exist_ok=True)
results = []

def run_self_eval_agent(sample: dict, index: int) -> None:
    if not isinstance(sample, dict) or "qa" not in sample or "question" not in sample["qa"] or "answer" not in sample["qa"]:
        return
    context = "\n".join(sample.get("pre_text", [])) + "\n" + "\n".join(sample.get("post_text", []))
    table = sample.get("table", [])
    table_text = "\n".join([" | ".join(row) for row in table]) if table else ""
    question = sample["qa"]["question"]
    gold = extract_number(sample["qa"]["answer"])
    base_prompt = (
        "You are a financial analyst assistant. Think step by step to solve the question.\n"
        "Always begin your reasoning with 'Wait...' to indicate you are thinking.\n\n"
        f"Context:\n{context}\n\nTable:\n{table_text}\n\nQuestion: {question}\n\nWait..."
    )
    t0 = time.time()
    first_answer = ask_gpt(base_prompt)
    t1 = time.time()
    pred1 = extract_number(extract_math_answer(first_answer))
    score1, status1 = score(pred1, gold)
    if score1 == 0:
        critique_prompt = (
            "You answered incorrectly.\n\n"
            f"Context:\n{context}\n\nTable:\n{table_text}\n\n"
            f"Question: {question}\nYour previous answer: {first_answer}\n"
            f"Expected: {sample['qa']['answer']}\n\n"
            "Explain the mistake briefly before retrying."
        )
        critique = ask_gpt(critique_prompt)
        retry_prompt = (
            f"{critique}\n\nNow think again, carefully. Begin with 'Wait...' "
            "and show all intermediate steps using <MATH>...</MATH> and the final answer in <ANSWER>...</ANSWER>.\n\nWait..."
        )
        retry_answer = ask_gpt(retry_prompt)
        pred2 = extract_number(extract_math_answer(retry_answer))
        score2, status2 = score(pred2, gold)
    else:
        critique = ""
        retry_answer = first_answer
        pred2 = pred1
        score2 = score1
        status2 = status1
    results.append({
        "index": index,
        "question": question,
        "gold_answer": gold,
        "first_answer": first_answer,
        "first_score": score1,
        "reason": critique,
        "retry_answer": retry_answer,
        "retry_score": score2
    })
try:
    data = load_data("train_turn")
except Exception as e:
    raise RuntimeError(f"Failed to load data: {e}")
valid_data = [
    d for d in data if isinstance(d, dict) and "qa" in d and "question" in d["qa"] and "answer" in d["qa"]
]
for i, example in enumerate(valid_data[:200]):
    run_self_eval_agent(example, i)
df = pd.DataFrame(results)
df.to_csv("outputs/langchain_self_eval_waittoken.csv", index=False)
initial_accuracy = df["first_score"].mean() * 100
revised_accuracy = df["retry_score"].mean() * 100
improvement = revised_accuracy - initial_accuracy
print(f"Initial Accuracy: {initial_accuracy:.2f}%")
print(f"Revised Accuracy: {revised_accuracy:.2f}%")
print(f"Improvement: {improvement:.2f}%")

