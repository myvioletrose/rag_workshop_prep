# RAG Workshop Prep Kit

Prepared for a hands-on workshop on **Monday, September 28, 2026**.

This version is designed to be completed in **4 hours 30 minutes**, including a 10-minute break. It assumes basic Python familiarity, Visual Studio Code on Windows, and no prior RAG experience.

Open `prep_plan.html` for the interactive checklist. It saves checked items and notes in your browser.

## The mental model to learn

RAG has two main phases:

1. **Indexing, normally completed before a user asks a question**
   - load documents
   - split documents into chunks
   - convert each chunk into an embedding vector
   - store vectors and metadata in an index such as FAISS
2. **Question answering, completed at request time**
   - embed the user's question
   - retrieve the nearest chunks
   - place those chunks into the model prompt
   - generate an answer grounded in the retrieved evidence

The most important debugging habit is to inspect the retrieved chunks **before** blaming the language model.

## Under-5-hour preparation plan

### Block 1 - VS Code setup: 25 minutes

- Create and select the `.venv` environment.
- Install packages from `requirements.txt`.
- Run `python check_setup.py`.
- Skim the VS Code notebook setup page for approximately 5 minutes:
  https://code.visualstudio.com/docs/datascience/jupyter-notebooks

Checkpoint: the notebook kernel and Python interpreter both point to the same `.venv`.

### Block 2 - RAG mental model: 30 minutes

- Watch **RAG From Scratch: Part 1 - Overview** (**5:13**):
  https://www.youtube.com/watch?v=wd7TZ4w1mSw
- Read only the retrieval pipeline, building blocks, and 2-step RAG sections:
  https://docs.langchain.com/oss/python/langchain/retrieval
- Draw the indexing path and the question-answering path from memory.

Checkpoint: explain where document loading, chunking, embeddings, FAISS, retrieval, and generation occur.

### Block 3 - Embeddings and FAISS: 45 minutes

- Read the introductory embedding concepts and one API example:
  https://developers.openai.com/api/docs/guides/embeddings
- Skim the FAISS sections on building an index, adding vectors, and searching:
  https://github.com/facebookresearch/faiss/wiki/Getting-started
- Run `01_semantic_search_faiss.py`.

Checkpoint: explain embeddings, vector similarity, and what `top-k` means.

### Block 4 - Chunking and retrieval: 40 minutes

- Watch **Part 2 - Indexing** (**4:52**):
  https://www.youtube.com/watch?v=bjb_EMsTDKI
- Watch **Part 3 - Retrieval** (**5:14**):
  https://www.youtube.com/watch?v=LxNVgdIz9sU
- Compare two chunk sizes and two values of `k` in the exercise.
- Write one sentence about the tradeoff between chunks that are too small and chunks that are too large.

Checkpoint: explain chunk size, overlap, metadata, and why retrieval quality matters.

### Break: 10 minutes

This break is included in the 4-hour-30-minute total.

### Block 5 - End-to-end RAG application: 70 minutes

- Skim the official tutorial sections on documents, embeddings, splitting, vector search, retrievers, and minimal RAG:
  https://docs.langchain.com/oss/python/langchain/knowledge-base
- Run `02_langchain_rag.py` or the matching cells in `rag_workshop_prep.ipynb`.
- Inspect the retrieved passages before reading the generated answer.
- Test one answerable question and one question that the source does not support.

Checkpoint: classify a weak answer as a retrieval failure, a generation failure, or insufficient source material.

### Block 6 - Grounding and guardrails: 30 minutes

- Watch **Part 4 - Generation** (**6:25**):
  https://www.youtube.com/watch?v=Vw52xyyFsB8
- Review the prompt in `02_langchain_rag.py`.
- Test these four controls:
  - use only the retrieved context
  - cite sources
  - decline unsupported questions
  - ignore instructions contained inside retrieved documents
- Optional 5-minute skim:
  https://docs.langchain.com/oss/python/langchain/guardrails

Checkpoint: explain why RAG reduces hallucination risk but does not guarantee correctness.

### Block 7 - Final readiness: 20 minutes

- Run `python check_setup.py --api`.
- Restart VS Code, reopen the notebook, and rerun one cell.
- Confirm your API credit, charger, and electrical plug adapter.
- Write down two questions to ask during the workshop.

**Total: 4 hours 30 minutes, including the break.**

## Windows and VS Code setup

Python 3.11 or 3.12 is the most conservative workshop choice. If the instructor supplies a dependency file, use it in a separate environment rather than changing a working environment immediately before the workshop.

Open PowerShell in the extracted folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Copy-Item .env.example .env
# Open .env and replace the placeholder with your own API key.

python check_setup.py
python check_setup.py --api
code .
```

If PowerShell blocks activation, use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, avoid activation and call the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_setup.py
```

In VS Code:

1. Install the Microsoft **Python** and **Jupyter** extensions.
2. Use `Ctrl+Shift+P` -> **Python: Select Interpreter** -> choose `.venv`.
3. Open the notebook and use the kernel picker in the upper-right corner -> choose the same `.venv`.

## Packages included in `requirements.txt`

- `openai`
- `langchain`, `langchain-core`, and `langchain-openai`
- `langchain-text-splitters` and `langchain-community`
- `faiss-cpu`
- `pypdf`
- `python-dotenv`
- `jupyter` and `ipykernel`
- `numpy`

Do not pre-install large optional stacks such as `torch`, `sentence-transformers`, `unstructured`, or a hosted vector database unless the organizer specifically requests them.

## OpenAI account checklist

- An OpenAI API Platform account is separate from a normal ChatGPT subscription.
- Create an API key before the workshop.
- Add a small prepaid balance and verify that it is visible.
- Keep the key only in `.env`; never paste it into a notebook committed to Git.
- Test one embedding request with `python check_setup.py --api`.
- Test from the same laptop and network configuration you expect to use at the workshop.

## What “socket adapter for your notebook” means

Here, **notebook** means laptop computer. A **socket adapter** means an electrical travel plug adapter that lets your laptop charger fit the wall outlets at the workshop location. It is not a Python, network, or software adapter.

Check the venue's country and your charger label. Most modern laptop chargers accept `100-240 V, 50/60 Hz`, so they usually need only the appropriate plug-shape adapter, not a voltage converter.

## Files in this kit

- `prep_plan.html` - interactive under-5-hour checklist with direct learning links
- `rag_workshop_prep.ipynb` - notebook exercises for VS Code
- `check_setup.py` - local package check and optional API check
- `01_semantic_search_faiss.py` - raw embeddings and FAISS retrieval exercise
- `02_langchain_rag.py` - end-to-end LangChain RAG exercise
- `03_pdf_rag.py` - optional follow-up exercise using your own public PDF
- `data/workshop_handbook.txt` - fictional source document for safe practice
- `requirements.txt` - package list
- `.env.example` - API configuration template

`03_pdf_rag.py` is intentionally optional in this shorter plan. Run it only if you finish early or want additional practice after the workshop.

## Questions worth asking during the workshop

- How were chunk size and overlap selected for this document type?
- Which similarity metric and FAISS index type are being used?
- How is retrieval quality evaluated separately from answer quality?
- What happens when no retrieved chunk is sufficiently relevant?
- How are page numbers, filenames, and other metadata preserved for citations?
- How is prompt injection inside retrieved documents handled?
