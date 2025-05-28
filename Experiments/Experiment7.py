import os
import json
import re
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from retrieve import RelevantDocumentRetriever
load_dotenv()
REQUIRED_ENV_VARS = ["OPENAI_API_KEY"]
missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing environment variables: {', '.join(missing_vars)}")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {filepath}: {e}")
def extract_last_number(text):
    if not isinstance(text, str):
        return None
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else None
def fuzzy_numeric_match(predicted, gold, tolerance=0.02):
    try:
        p = float(predicted.replace("%", "").strip())
        g = float(gold.replace("%", "").strip())
        return abs(p - g) / max(abs(g), 1e-6) <= tolerance
    except Exception:
        return False
eval_data = load_json("convfinqa_eval.json")
retriever = RelevantDocumentRetriever(data_path="ConvFinQA/data/train.json")
results = []
for example in tqdm(eval_data):
    question = example.get("question", "")
    gold = example.get("gold_answer", "").strip()
    if not question or not gold:
        continue
    try:
        docs = retriever.query(question)
        if not docs:
            raise ValueError("No documents retrieved.")
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "You are a financial analyst. Answer the question using step-by-step reasoning based on the context. "
            "Include any formulas or numerical calculations used.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = llm([HumanMessage(content=prompt)]).content.strip()
        predicted = extract_last_number(response)
        exact_match = predicted.lower() == gold.lower() if predicted else False
        results.append({
            "question": question,
            "gold_answer": gold,
            "reasoning": response,
            "predicted_final": predicted,
            "exact_match": exact_match })
    except Exception as e:
        results.append({
            "question": question,
            "gold_answer": gold,
            "reasoning": "ERROR",
            "predicted_final": "ERROR",
            "exact_match": False,
            "error": str(e)})
df = pd.DataFrame(results)##logging results for the same 
df["fuzzy_match"] = df.apply(
    lambda row: fuzzy_numeric_match(str(row["predicted_final"]), str(row["gold_answer"])), axis=1
)
df.to_csv("eval_results_oracle_gpt35.csv", index=False)
exact = df["exact_match"].sum()
fuzzy = df["fuzzy_match"].sum()
total = len(df)
print(f"Exact Match Accuracy: {exact}/{total} = {round(100 * exact / total, 2)}%")
print(f"Fuzzy Match Accuracy (±2%): {fuzzy}/{total} = {round(100 * fuzzy / total, 2)}%")
