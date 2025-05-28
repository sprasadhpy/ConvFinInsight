import os
import json
import pandas as pd
import re
from statistics import mean
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS

# -Ensure u have the loader.py ---
from data.loader import FILES, REPO_DIR
sns.set(style="whitegrid")
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def tokenize_filtered(text):
    stopwords = set(STOPWORDS)
    return [w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()) if w not in stopwords]

def classify_question(q):
    q = q.lower()
    if any(kw in q for kw in ["percent", "percentage", "rate", "ratio"]):
        return "Percentage/Ratio"
    elif any(kw in q for kw in ["increase", "decrease", "change", "difference", "subtract", "minus"]):
        return "Change/Difference"
    elif any(kw in q for kw in ["sum", "total", "add", "combined"]):
        return "Addition/Sum"
    elif any(kw in q for kw in ["average", "mean"]):
        return "Average"
    elif any(kw in q for kw in ["value", "report", "amount", "figure"]):
        return "Direct Lookup"
    else:
        return "Other"

def compute_stats(data):
    num_conversations = len(data)
    total_questions, total_question_length, total_sentences = 0, 0, 0
    total_table_rows, tokens_all_inputs = 0, []
    vocab, report_pages = set(), set()

    for conv in data:
        annotation = conv.get("annotation", {})
        dialogue = annotation.get("dialogue_break", [])
        total_questions += len(dialogue)
        total_question_length += sum(len(tokenize(q)) for q in dialogue)

        text = conv.get("pre_text", []) + conv.get("post_text", [])
        total_sentences += sum(t.count('.') + t.count('!') + t.count('?') for t in text)

        tokens = tokenize(" ".join(text))
        vocab.update(tokens)
        tokens_all_inputs.append(len(tokens))

        table = conv.get("table", [])
        total_table_rows += max(len(table) - 1, 0)

        if 'filename' in conv and '/' in conv['filename']:
            report_pages.add(conv['filename'])

    return {
        "Number of Conversations": num_conversations,
        "Number of Questions": total_questions,
        "Report Pages": len(set(r.split('/')[1] for r in report_pages)),
        "Vocabulary Size": len(vocab),
        "Avg. # questions per conversation": round(total_questions / num_conversations, 2),
        "Avg. question length": round(total_question_length / total_questions, 2) if total_questions else 0,
        "Avg. # sentences in input text": round(total_sentences / num_conversations, 2),
        "Avg. # rows in input table": round(total_table_rows / num_conversations, 2),
        "Avg. # tokens in all inputs (text & table)": round(mean(tokens_all_inputs), 2),
        "Max. # tokens in all inputs (text & table)": max(tokens_all_inputs) if tokens_all_inputs else 0
    }

all_stats = {}
for name in ["Train Dialogues", "Dev Dialogues", "Test Dialogues (Private)"]:
    path = os.path.join(REPO_DIR, FILES[name])
    with open(path, "r") as f:
        data = json.load(f)
    all_stats[name] = compute_stats(data)

df_all = pd.DataFrame(all_stats).T.reset_index().rename(columns={'index': 'Dataset'})

with open(os.path.join(REPO_DIR, FILES["Train Dialogues"]), "r") as f:
    train_data = json.load(f)

dependency_distances = []
for convo in train_data:
    turn_programs = convo.get("annotation", {}).get("turn_program", [])
    for i, prog in enumerate(turn_programs):
        refs = re.findall(r"#(\d+)", prog)
        distances = [i - int(ref) for ref in refs if int(ref) < i]
        dependency_distances.append(max(distances) if distances else 0)


rows = []
for sample in train_data:
    steps = sample.get("qa", {}).get("steps", [])
    reasoning_steps = "\n".join([f"• {step['op']}({step['arg1']}, {step['arg2']}) → {step['res']}" for step in steps])
    dialogue = sample.get("annotation", {}).get("dialogue_break", [])
    dialogue_text = "\n".join([f"Q{i}: {line}" for i, line in enumerate(dialogue)])
    rows.append({
        "Document": sample.get("filename", "N/A"),
        "QA Question": sample.get("qa", {}).get("question", ""),
        "Answer": sample.get("qa", {}).get("answer", ""),
        "Reasoning Steps": reasoning_steps,
        "Dialogue Breakdown": dialogue_text
    })
pd.DataFrame(rows).to_excel("convfinqa_summary_export.xlsx", index=False)

lines = [sample.get("qa", {}).get("question", "") for sample in train_data]
for sample in train_data:
    lines.extend(sample.get("annotation", {}).get("dialogue_break", []))
all_tokens = [token for line in lines for token in tokenize_filtered(line)]
stock_keywords = {"stock", "market", "equity", "share", "price", "volume", "ticker", "dividend", "index", "nasdaq", "nyse", "capital", "return"}
finance_keywords = {"revenue", "expense", "profit", "debt", "asset", "liability", "ebitda", "cash", "flow", "balance", "income", "net", "loss", "operation", "growth"}
stock_count = sum(1 for t in all_tokens if t in stock_keywords)
finance_count = sum(1 for t in all_tokens if t in finance_keywords)


qtype_counts = pd.Series([
    classify_question(sample.get("qa", {}).get("question", ""))
    for sample in train_data
]).value_counts()


operator_counter = Counter()
for sample in train_data:
    for step in sample.get("qa", {}).get("steps", []):
        operator_counter[step["op"]] += 1
operator_freq = pd.Series(operator_counter).sort_values(ascending=False)

company_year = []
for sample in train_data:
    parts = sample.get("filename", "").split('/')
    if len(parts) >= 2:
        company = parts[0].upper()
        match = re.search(r'\b(20\d{2})\b', parts[1])
        if match:
            year = match.group(1)
            company_year.append((company, year))
cy_df = pd.DataFrame(company_year, columns=["Company", "Year"])
heatmap_data = cy_df.value_counts().unstack(fill_value=0).sort_index()

unit_counter = Counter()
for sample in train_data:
    ans = sample.get("qa", {}).get("answer", "").lower()
    for unit in ["%", "$", "million", "billion", "thousand"]:
        if unit in ans:
            unit_counter[unit] += 1
unit_freq = pd.Series(unit_counter).sort_values(ascending=False)


top10_companies = cy_df["Company"].value_counts().nlargest(10)
fig, axs = plt.subplots(4, 2, figsize=(20, 20))


melted = df_all.melt(id_vars="Dataset", var_name="Metric", value_name="Value")
sns.barplot(data=melted, x="Value", y="Metric", hue="Dataset", ax=axs[0, 0])
axs[0, 0].set_title("ConvFinQA Dataset Statistics")
axs[0, 0].legend()


axs[0, 1].hist(dependency_distances, bins=range(0, max(dependency_distances)+2), align='left', rwidth=0.8, color='skyblue')
axs[0, 1].set_title("Longest Dependency Distance")


qtype_counts.plot(kind='bar', ax=axs[1, 0], color='purple')
axs[1, 0].set_title("Question Type Distribution")

operator_freq.plot(kind='bar', ax=axs[1, 1], color='teal')
axs[1, 1].set_title("Program Operator Frequency")


sns.heatmap(heatmap_data, cmap="YlGnBu", ax=axs[2, 0])
axs[2, 0].set_title("Report Coverage: Company × Year")

unit_freq.plot(kind='bar', ax=axs[2, 1], color='darkorange')
axs[2, 1].set_title("Most Common Answer Units")


axs[3, 0].bar(["Stock Terms", "Finance Terms"], [stock_count, finance_count], color=["steelblue", "darkgreen"])
axs[3, 0].set_title("Keyword Usage in QA + Dialogue")

sns.barplot(x=top10_companies.index, y=top10_companies.values, ax=axs[3, 1], palette="crest")
axs[3, 1].set_title("Top 10 Companies by Mentions")

plt.tight_layout()
plt.show()
