import os
import json
import subprocess
import pandas as pd

REPO_URL = "https://github.com/czyssrs/ConvFinQA.git"
REPO_DIR = "ConvFinQA"
DATA_ZIP = "data.zip"
DATA_DIR = "data"

FILES = {
    "Train Dialogues": os.path.join(DATA_DIR, "train.json"),
    "Train Turns": os.path.join(DATA_DIR, "train_turn.json"),
    "Dev Dialogues": os.path.join(DATA_DIR, "dev.json"),
    "Dev Turns": os.path.join(DATA_DIR, "dev_turn.json"),
    "Test Dialogues (Private)": os.path.join(DATA_DIR, "test_private.json"),
    "Test Turns (Private)": os.path.join(DATA_DIR, "test_turn_private.json"),
}

def clone_repo():
    if not os.path.exists(REPO_DIR):
        print("Cloning ConvFinQA repository...")
        subprocess.run(["git", "clone", REPO_URL], check=True)
    else:
        print("Repository already exists.")
def unzip_data():
    os.chdir(REPO_DIR)
    if os.path.exists(DATA_ZIP):
        print("Unzipping data.zip...")
        subprocess.run(["unzip", "-o", DATA_ZIP], check=True)
    else:
        print("data.zip not found.")
    os.chdir("..")
def count_samples():
    print("\nDataset sample Counts:")
    for name, path in FILES.items():
        full_path = os.path.join(REPO_DIR, path)
        with open(full_path, "r") as f:
            data = json.load(f)
        print(f"{name}: {len(data)} samples")
def inspect_sample(sample_index=0):
    path = os.path.join(REPO_DIR, FILES["Train Dialogues"])
    with open(path, "r") as f:
        train_data = json.load(f)

    sample = train_data[sample_index]

    print(" Document:", sample.get("filename", "N/A"))
    print(" QA Question:", sample["qa"]["question"])
    print(" Answer:", sample["qa"]["answer"])
    print("\n" + "=" * 80 + "\n")
    print(" Pre-Text (Excerpt):")
    for line in sample.get("pre_text", [])[:5]:
        print("-", line)
    print("\n Post-Text (Excerpt):")
    for line in sample.get("post_text", [])[:5]:
        print("-", line)
    print("\n Financial Table (from PDF page):")
    table_data = sample["table"]
    table_df = pd.DataFrame(table_data[1:], columns=table_data[0])
    print(table_df)
    print("\n Reasoning Steps:")
    for step in sample["qa"].get("steps", []):
        print(f"• {step['op']}({step['arg1']}, {step['arg2']}) → {step['res']}")
    annotation = sample.get("annotation", {})
    if "dialogue_break" in annotation:
        print("\n Dialogue Breakdown:")
        for i, question in enumerate(annotation["dialogue_break"]):
            print(f"Q{i}: {question}")
