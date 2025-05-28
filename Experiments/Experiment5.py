import os
import json
import re
import time
import random
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import traceable
from langchain.callbacks.tracers import LangChainTracer
from data.loader import load_data
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
FMP_API_KEY = os.getenv("FMP_API_KEY")
required_vars = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
    "LANGCHAIN_PROJECT": LANGCHAIN_PROJECT,
    "LANGCHAIN_ENDPOINT": LANGCHAIN_ENDPOINT,
    "FMP_API_KEY": FMP_API_KEY
}

missing = [k for k, v in required_vars.items() if not v]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
tracer = LangChainTracer()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2, callbacks=[tracer])
@traceable(name="Ask ConvFinQA")
def ask_gpt(prompt):
    return llm([HumanMessage(content=prompt)]).content.strip()
def clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
full_data = load_data("train_turn")
valid_data = [turn for turn in full_data if "qa" in turn and "question" in turn["qa"] and "answer" in turn["qa"]]
def is_stock_price_question(turn):
    q = turn["qa"]["question"].lower()
    return "stock price" in q or "share price" in q or "price per share" in q
price_related_data = [turn for turn in valid_data if is_stock_price_question(turn)]
sampled_data = random.sample(price_related_data, min(200, len(price_related_data)))
def flatten_table(table):
    headers = table[0]
    rows = table[1:]
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)
def make_prompt(turn, use_math=False):
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table_text = flatten_table(turn["table"])
    question = turn["qa"]["question"]
    instructions = "You are a financial analyst assistant. Use the context below to answer the question."
    if use_math:
        instructions += " If calculations are needed, enclose the math expression in <MATH>...</MATH> and the result in <ANSWER>...</ANSWER>."
    else:
        instructions += " Directly provide the final answer in <ANSWER>...</ANSWER>."
    return f"""{instructions}
Context:
{pre}

Table:
{table_text}

{post}

Question: {question}
Answer:"""

def extract_math_answer(text):
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
def extract_math_expression(text):
    match = re.search(r"<MATH>(.*?)</MATH>", text, re.DOTALL)
    return match.group(1).strip() if match else None
def evaluate_expression(expr):
    try:
        return eval(expr, {"__builtins__": {}})
    except:
        return None
def extract_number(text):
    if not isinstance(text, str):
        return None
    nums = re.findall(r"-?\d+\.?\d*", text)
    return float(nums[-1]) if nums else None
def score_with_message(pred, gold, rel_tol=0.03, abs_tol=0.05):
    if pred is None or gold is None:
        return 0, "Unable to compare (missing number)"
    try:
        err = abs(pred - gold)
    except Exception as e:
        return 0, f"Error comparing values: {e}"
    if err <= max(rel_tol * abs(gold), abs_tol):
        return 1, "Match (numeric)"
    else:
        return 0, f"No match (diff {err:.4f})"

def extract_ticker_symbol(turn):
    id_field = turn.get("id", "")
    match = re.search(r"Single_([A-Z]{1,5})/\d{4}", id_field)
    return match.group(1) if match else None
def fetch_fmp_price(symbol):
    url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data and "price" in data[0]:
                return data[0]["price"]
    except:
        return None
    return None
results = []
for i, turn in enumerate(sampled_data):
    gold = extract_number(turn["qa"]["answer"])
    conv_id = turn.get("id", "")
    turn_ind = turn.get("turn_ind", None)
    prompt1 = make_prompt(turn, use_math=False)
    t0 = time.time()
    gen1 = ask_gpt(prompt1)
    t1 = time.time()
    pred1 = extract_number(extract_math_answer(gen1))
    score1, msg1 = score_with_message(pred1, gold)
    prompt2 = make_prompt(turn, use_math=True)
    t2 = time.time()
    gen2 = ask_gpt(prompt2)
    expr = extract_math_expression(gen2)
    calc_result = evaluate_expression(expr) if expr else None
    fallback_number = extract_number(extract_math_answer(gen2))
    pred2 = calc_result if isinstance(calc_result, (int, float)) else fallback_number
    score2, msg2 = score_with_message(pred2, gold)
    t3 = time.time()
    ticker = extract_ticker_symbol(turn)
    fmp_price, gen3, pred3, score3, msg3 = None, None, None, None, None
    if ticker:
        fmp_price = fetch_fmp_price(ticker)
        fmp_context = f"The current market price of {ticker} from FMP API is ${fmp_price}." if fmp_price else ""
        prompt3 = f"{make_prompt(turn, use_math=True)}\n\nAdditional Context:\n{fmp_context}"
        gen3 = ask_gpt(prompt3)
        expr3 = extract_math_expression(gen3)
        calc_result3 = evaluate_expression(expr3) if expr3 else None
        fallback_number3 = extract_number(extract_math_answer(gen3))
        pred3 = calc_result3 if isinstance(calc_result3, (int, float)) else fallback_number3
        score3, msg3 = score_with_message(pred3, gold)

    results.append({
        "id": conv_id,
        "turn_ind": turn_ind,
        "question": turn["qa"]["question"],
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
        "gen_llm": clean_text(gen1),
        "gen_calc": clean_text(gen2),
        "ticker": ticker,
        "fmp_price": fmp_price,
        "gen_calc_fmp": clean_text(gen3),
        "pred_calc_fmp": pred3,
        "score_calc_fmp": score3,
        "match_calc_fmp": msg3
    })

Path("outputs").mkdir(exist_ok=True)
df = pd.DataFrame(results)
df.to_csv("outputs/stock_price_accuracy_analysis.csv", index=False)
