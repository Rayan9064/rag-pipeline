# First RAG Pipeline (Open-Source LLM)


This project provides an advanced **Retrieval-Augmented Generation (RAG) pipeline** with:

- **Multi-format ingestion:** Supports `.pdf`, `.docx`, and `.txt` files in the `data/` folder
- **Vector database:** Uses FAISS for scalable, persistent semantic retrieval
- **Feedback loop:** Logs every question, answer, user feedback, and latency to `rag_feedback_log.csv`
- **Model monitoring:** Tracks answer latency and accuracy for improvement
- **Use case:** Internal HR Policy Q&A assistant (easily adaptable)
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Generator model:** `google/flan-t5-base`

## What it does

1. Loads `.pdf`, `.docx`, and `.txt` documents from `data/`
2. Splits them into overlapping chunks
3. Builds a FAISS vector index for semantic retrieval
4. Retrieves top-k relevant chunks for a question
5. Generates an answer grounded in retrieved context
6. Logs every interaction and feedback for monitoring and future improvement

## Project structure

```
first-RAG/
├─ data/
│  ├─ leave_policy.txt
│  ├─ remote_work_policy.txt
│  ├─ reimbursement_policy.txt
│  ├─ your_docs.pdf
│  └─ your_docs.docx
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


## Run (Interactive CLI)

```bash
python main.py
```

You will be prompted to enter questions. Type `exit` to quit.


## Features

- **Multi-format ingestion:** Place `.pdf`, `.docx`, and `.txt` files in `data/`.
- **Scalable retrieval:** Uses FAISS vector DB for fast, large-scale search.
- **Feedback loop:** After each answer, mark if it was correct; all data is logged to `rag_feedback_log.csv`.
- **Monitoring:** Latency and accuracy are tracked for every query.
- **Easy extensibility:** Swap models, add new data, or adapt to new use cases.

## Notes

- First run downloads models from Hugging Face.
- If answer quality is low, try increasing retrieval breadth with `--top-k 5`.
- You can replace the LLM model with another open-source model using `--llm-model`.

