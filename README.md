# First RAG Pipeline (Open-Source LLM)

This project provides a **first implementation of a Retrieval-Augmented Generation (RAG) pipeline** using a practical use case:

- **Use case:** Internal HR Policy Q&A assistant
- **Embedding model (open source):** `sentence-transformers/all-MiniLM-L6-v2`
- **Generator model (open source):** `google/flan-t5-base`

## What it does

1. Loads `.txt` policy documents from `data/`
2. Splits them into overlapping chunks
3. Builds an embedding index for semantic retrieval
4. Retrieves top-k relevant chunks for a question
5. Generates an answer grounded in retrieved context

## Project structure

```
first-RAG/
├─ data/
│  ├─ leave_policy.txt
│  ├─ remote_work_policy.txt
│  └─ reimbursement_policy.txt
├─ rag_pipeline/
│  ├─ __init__.py
│  ├─ chunking.py
│  ├─ retriever.py
│  ├─ generator.py
│  └─ pipeline.py
├─ main.py
└─ requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Example query:

```bash
python main.py --question "How many PTO days can I carry over to next year?"
```

Another query:

```bash
python main.py --question "What is the limit for international remote work per year?"
```

## Notes

- First run downloads models from Hugging Face.
- If answer quality is low, try increasing retrieval breadth:

```bash
python main.py --question "..." --top-k 5
```

- You can replace the LLM model with another open-source model:

```bash
python main.py --question "..." --llm-model "google/flan-t5-large"
```
