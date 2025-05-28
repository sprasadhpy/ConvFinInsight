import torch
import json
import random
import pandas as pd
from transformers import pipeline
from unsloth import FastLanguageModel
def load_model(model_path, max_length=2048):
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("Invalid model path provided.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    return model, tokenizer
def setup_pipeline(model, tokenizer, max_tokens=512):
    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must not be None.")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=max_tokens)
def load_data(filepath, sample_size=50):
    if not filepath.endswith(".json"):
        raise ValueError("Expected a JSON file.")
    with open(filepath, "r") as f:
        data = json.load(f)
    valid_data = [turn for turn in data if "qa" in turn and "question" in turn["qa"] and "answer" in turn["qa"]]
    if sample_size > len(valid_data):
        sample_size = len(valid_data)
    return random.sample(valid_data, sample_size)
def flatten_table(table):
    if not table or not isinstance(table, list) or not table[0]:
        return ""
    headers = table[0]
    rows = table[1:]
    lines = [" | ".join(headers)]
    for row in rows:
        if len(row) == len(headers):
            lines.append(" | ".join(row))
    return "\n".join(lines)
def make_prompt(turn, use_math=True):
    if "qa" not in turn or "question" not in turn["qa"] or "table" not in turn:
        return ""
    pre = "\n".join(turn.get("pre_text", []))
    post = "\n".join(turn.get("post_text", []))
    table = flatten_table(turn["table"])
    question = turn["qa"]["question"]
    instruction = "You are a financial analyst assistant. Use the context to answer the question clearly and enclose the answer in <ANSWER>...</ANSWER>."
    if use_math:
        instruction += " If you do calculations, also enclose expressions in <MATH>...</MATH>."
    return f"{instruction}\n\nContext:\n{pre}\n\nTable:\n{table}\n\n{post}\n\nQuestion: {question}\nAnswer:"
def run_inference(pipe, data):
    results = []
    for i, turn in enumerate(data):
        prompt = make_prompt(turn, use_math=True)
        if not prompt:
            continue
        try:
            output = pipe(prompt, return_full_text=False)[0].get("generated_text", "").strip()
        except Exception as e:
            output = f"Error: {str(e)}"
        results.append({
            "question": turn["qa"]["question"],
            "expected": turn["qa"]["answer"],
            "generated": output,
        })
    return results
def save_results(results, path):
    if not results or not isinstance(results, list):
        raise ValueError("Results should be a non-empty list.")
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
if __name__ == "__main__":
    model_path = "./qwen3-sft-final"
    data_path = "data/train_turn.json"
    output_path = "qwen3_sft_inference_50.csv"
    model, tokenizer = load_model(model_path)
    pipe = setup_pipeline(model, tokenizer)
    sampled_data = load_data(data_path, sample_size=50)
    inference_results = run_inference(pipe, sampled_data)
    save_results(inference_results, output_path)
