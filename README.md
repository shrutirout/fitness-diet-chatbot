# Fitness and Diet Chatbot

A RAG-based chatbot that answers fitness and nutrition questions using content from 5 professional textbooks as its knowledge base.

## How it works

User question -> Pinecone retrieves the 5 most relevant text chunks from the books -> chunks are passed as context to Gemini -> answer is returned in the chat UI.

The books are indexed once via `store_index.py`. The Flask app connects to the existing Pinecone index at runtime.

## Stack

- Flask, LangChain
- Google Gemini 2.5 Flash Lite
- Pinecone (vector database)
- HuggingFace `all-MiniLM-L6-v2` (embeddings, 384-dim)
- Docker, GitHub Actions, AWS ECR + EC2

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
PINECONE_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

Index the knowledge base (run once):

```bash
python store_index.py
```

Run the app:

```bash
python app.py
```

Open `http://localhost:8080`

## Deployment

The CI/CD pipeline builds a Docker image on every push to `main`, pushes it to AWS ECR, and redeploys on the EC2 instance via a self-hosted GitHub Actions runner.

Required GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, `PINECONE_API_KEY`, `GOOGLE_API_KEY`, `ECR_REPO`
