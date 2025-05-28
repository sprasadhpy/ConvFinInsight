# ConvFinInsight

**ConvFinInsight** is a Large Language Model (LLM)-powered financial reasoning framework built on top of the ConvFinQA dataset.  
It combines:

- Modular document parsing  
-  Context-aware retrieval  
-  Chain-of-thought generation  

to answer complex financial questions grounded in semi-structured documents like earnings reports and financial statements.

---

##  Environment Setup

First, set up the environment:

```bash
conda env create -f environment.yml
conda activate convfinqa
pip install python-dotenv

```
API Configuration

Create a .env file in the root directory with the following:

```bash
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
FMP_API_KEY=your_fmp_api_key
```


 These keys enable:
 Access to OpenAI for LLM reasoning
 LangChain orchestration
 Financial data retrieval via FMP
 Important: Never commit your .env file. Add it to your .gitignore.


Project Structure


ConvFinInsight/
├── Experiment1/           # Prompting Evaluation on ConvFinQA
├── Experiment2/           # Prompting Strategies Explored
├── Experiment3/           # Self-Reflective Evaluation
├── Experiment4/           # Extended Reasoning with the “Wait...” Scaffolding Technique
├── Experiment5/           # FMP API Integration
├── Experiment6/           # Qwen3-14B Evaluation 
├── Experiment7/           # RAG Benchmarking
├── Experiment8/           # Guardrail-Augmented Evaluation of Financial QA
└── Experiment9            # Qwen3-14B With SFT Finetuning



 
