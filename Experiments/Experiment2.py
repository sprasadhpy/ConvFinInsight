import os
import re
import time
import random
import pandas as pd
from typing import Optional, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from data.loader import load_data


load_dotenv()
tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, callbacks=[tracer])
@traceable(name="Ask ConvFinQA")
def ask_gpt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("Prompt must be a string.")
    response = llm([HumanMessage(content=prompt)])
    return response.content.strip()
try:
    full_data = load_data("train_turn")
except Exception as e:
    raise RuntimeError(f"Failed to load dataset: {e}")
valid_data = [
    turn for turn in full_data
    if isinstance(turn, dict) and "qa" in turn and "question" in turn["qa"] and "answer" in turn["qa"]
]
print(f"Total entries: {len(full_data)} | Valid QA entries: {len(valid_data)}")
N = 200
sampled_data = random.sample(valid_data, N)
strategies = ["ExplicitCoT", "ReflectionCoT", "Scratchpad", "PoT", "EEDP", "Agentic"]
def flatten_table(table: List[List[str]]) -> str:
    if not table or not isinstance(table, list):
        return ""
    try:
        headers = table[0]
        rows = table[1:]
        lines = [" | ".join(headers)]
        for row in rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)
    except Exception:
        return ""
def make_prompt(turn: dict, strategy: str) -> str:
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table_text = flatten_table(turn.get("table", []))
    question = turn["qa"]["question"]
    if strategy == "ExplicitCoT":
        instructions = "You are a financial assistant. Explain your reasoning step by step before answering."
    elif strategy == "ReflectionCoT":
        instructions = "Reflect carefully on each part of the question before answering. State any missing information before proceeding."
    elif strategy == "Scratchpad":
        instructions = "Use the scratchpad below to write down intermediate steps. After thinking, provide the final answer."
    elif strategy == "PoT":
        instructions = "Wrap all math expressions in <MATH>...</MATH> and the final result in <ANSWER>...</ANSWER>."
    elif strategy == "EEDP":
        instructions = "Break down the question into sub-questions. Solve each, and combine the results. Say 'Insufficient data' if anything is missing."
    elif strategy == "Agentic":
        instructions = (
            "You are FinBot, an agentic financial QA assistant.\n"
            "- Declare limitations first. Say 'Insufficient data' if inputs are incomplete.\n"
            "- Reason step-by-step using symbolic CoT.\n"
            "- Use only data visible in the context. Do not hallucinate or guess.\n"
            "- Wrap math in <MATH>...</MATH> and the final result in <ANSWER>...</ANSWER>.\n"
            "- Follow fallback rules instead of fabricating numbers."
        )
    else:
        instructions = "Answer the question below."

    return f"""{instructions}
Context:
{pre}
Table:
{table_text}
{post}
Question: {question}
Answer:"""

def extract_math_answer(text: str) -> str:
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def extract_math_expression(text: str) -> Optional[str]:
    match = re.search(r"<MATH>(.*?)</MATH>", text, re.DOTALL)
    return match.group(1).strip() if match else None
def evaluate_expression(expr: str) -> Optional[float]:
    try:
        return eval(expr, {"__builtins__": {}})
    except Exception:
        return None
def extract_number(text: str) -> Optional[float]:
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None
def score_with_message(pred: Optional[float], gold: Optional[float],
                       rel_tol: float = 0.03, abs_tol: float = 0.05) -> Tuple[int, str]:
    if pred is None or gold is None:
        return 0, "Unable to compare (missing number)"
    err = abs(pred - gold)
    if err <= max(rel_tol * abs(gold), abs_tol):
        return 1, "Match (numeric)"
    else:
        return 0, f"No match (diff {err:.4f})"
results = []
for i, turn in enumerate(sampled_data):
    gold = extract_number(turn["qa"]["answer"])
    for strategy in strategies:
        prompt = make_prompt(turn, strategy)
        t0 = time.time()
        try:
            gen = ask_gpt(prompt)
        except Exception as e:
            print(f"Error for Q{i+1} [{strategy}]: {e}")
            continue
        t1 = time.time()
        expr = extract_math_expression(gen)
        calc_result = evaluate_expression(expr) if expr else None
        pred = calc_result if calc_result is not None else extract_number(extract_math_answer(gen))
        score, msg = score_with_message(pred, gold)

        result = {
            "question_id": f"Q{i+1:03d}",
            "strategy": strategy,
            "question": turn["qa"]["question"],
            "gold": gold,
            "pred": pred,
            "score": score,
            "match": msg,
            "latency": t1 - t0,
            "expr": expr,
            "gen": gen,
        }
        results.append(result)

        print(f"[{result['question_id']}][{strategy}] Score: {score}, Time: {result['latency']:.2f}s, Match: {msg}")

        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv("intermediate_results.csv", index=False)
            print("Saved checkpoint to intermediate_results.csv")
print("\nAccuracy by prompting strategy")
df = pd.DataFrame(results)
for strat in strategies:
    subset = df[df["strategy"] == strat]
    acc = subset["score"].mean() * 100 if not subset.empty else 0.0
    print(f"{strat}: Accuracy = {acc:.1f}%")
df.to_csv("final_results.csv", index=False)
print("\ results saved to final_results.csv")

