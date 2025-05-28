# ConvFinInsight

**ConvFinInsight** is an LLM-powered financial reasoning framework built on top of the ConvFinQA dataset. It is designed to tackle complex financial questions grounded in semi-structured documents such as earnings reports and financial statements.  
Key components include:

- **Modular document parsing**  
- **Context-aware retrieval**  
- **Chain-of-thought generation**

---

##  Environment Setup

To get started, set up the environment:

```bash
conda env create -f environment.yml
conda activate convfinqa
pip install python-dotenv
```

### API Configuration

Create a `.env` file in the root directory with the following content:

```bash
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
FMP_API_KEY=your_fmp_api_key
```

These keys enable:

- Access to OpenAI for LLM-based reasoning  
- LangChain orchestration and memory  
- Financial data retrieval from FMP

>  **Important**: Never commit your `.env` file to version control. Ensure it is listed in your `.gitignore`.

---

##  Project Structure

```
ConvFinInsight/
├── Experiment1/   # Prompting evaluation on ConvFinQA
├── Experiment2/   # Exploration of prompting strategies
├── Experiment3/   # Self-reflective evaluation
├── Experiment4/   # Extended reasoning with “Wait...” scaffolding
├── Experiment5/   # FMP API integration
├── Experiment6/   # Evaluation using Qwen3-14B
├── Experiment7/   # Retrieval-augmented generation (RAG) benchmarking
├── Experiment8/   # Guardrail-augmented evaluation for financial QA
└── Experiment9/   # Fine-tuning Qwen3-14B with SFT
 
