cat <<EOF > README.md
# ConvFinInsight 

## ConvFinInsight is an LLM-powered financial reasoning framework built on top of the ConvFinQA dataset.  
It combines:

-  Modular document parsing  
-  Context-aware retrieval  
-  Chain-of-thought generation  

to answer complex financial questions grounded in semi-structured documents.

---

## 🔧 Environment Setup

Set up the environment with the following steps:

\`\`\`bash
conda env create -f environment.yml
conda activate convfinqa
pip install python-dotenv
\`\`\`

---

##  API Configuration

Create a \`.env\` file in the root directory with the following contents:

\`\`\`env
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
FMP_API_KEY=your_fmp_api_key
\`\`\`

>  These keys enable access to OpenAI, LangChain orchestration, and financial data via FMP.  
> ⚠ **Note:** Never commit your \`.env\` file. Ensure it's listed in your \`.gitignore\`.

---

