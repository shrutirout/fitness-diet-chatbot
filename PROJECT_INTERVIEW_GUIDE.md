# Fitness & Diet Chatbot — Complete Interview Preparation Guide

> A deep-dive narrative of every technical decision, concept, function, and design choice.
> Written as if you are explaining your work in a technical interview.

---

## TABLE OF CONTENTS

1. [Project Elevator Pitch](#1-project-elevator-pitch)
2. [Problem Statement & Motivation](#2-problem-statement--motivation)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Core Concepts — From First Principles](#4-core-concepts--from-first-principles)
   - 4.1 What is RAG?
   - 4.2 Why not just fine-tune an LLM?
   - 4.3 What are Vector Embeddings?
   - 4.4 What is a Vector Database?
   - 4.5 Cosine Similarity
   - 4.6 What is LangChain?
5. [Data & Knowledge Base](#5-data--knowledge-base)
6. [Data Pipeline — The Indexing Phase](#6-data-pipeline--the-indexing-phase)
   - 6.1 load_pdf_file()
   - 6.2 filter_to_minimal_docs()
   - 6.3 text_split()
   - 6.4 download_hugging_face_embeddings()
   - 6.5 Pinecone Index Creation & Upsert
7. [Query Pipeline — The Runtime Phase](#7-query-pipeline--the-runtime-phase)
   - 7.1 Flask App Startup
   - 7.2 The Retriever
   - 7.3 Prompt Engineering
   - 7.4 The LLM — Google Gemini
   - 7.5 LangChain Chains
   - 7.6 The /get Endpoint
8. [Frontend — The Chat UI](#8-frontend--the-chat-ui)
9. [Package Structure & Setup](#9-package-structure--setup)
10. [Containerization with Docker](#10-containerization-with-docker)
11. [CI/CD Pipeline — GitHub Actions + AWS](#11-cicd-pipeline--github-actions--aws)
12. [Challenges, Trade-offs & Decisions](#12-challenges-trade-offs--decisions)
13. [Interview Questions & In-Depth Answers](#13-interview-questions--in-depth-answers)

---

## 1. Project Elevator Pitch

"I built an AI-powered Fitness and Diet Chatbot that acts as a personal health coach. What makes it different from just using ChatGPT is that it's **grounded in verified domain knowledge** — specifically, five professional textbooks on sports nutrition, strength training, and dietary science. The bot uses a technique called **Retrieval-Augmented Generation (RAG)**, where before the AI answers your question, it first searches a vector database of 171 megabytes of textbook content and pulls the most relevant passages. Those passages are then fed into Google's Gemini language model alongside your question, so the answer is backed by real referenced material, not hallucinated information. The whole thing is wrapped in a Flask web app, containerized with Docker, and deployed to AWS EC2 through a fully automated CI/CD pipeline using GitHub Actions."

---

## 2. Problem Statement & Motivation

General-purpose LLMs like ChatGPT or Gemini can answer fitness questions, but they have two fundamental problems:

1. **Hallucination**: They may confidently give incorrect caloric counts, wrong exercise form cues, or outdated nutritional guidelines. For health topics, this is dangerous.
2. **No domain authority**: Their answers are averaged across millions of internet pages — including forums, blogs, and misinformation — not from authoritative scientific sources.

The goal of this project was to build a chatbot that:
- Answers fitness and diet questions with **specific, actionable numbers** (calories, sets, reps, macros).
- Sources its answers from **professional textbooks** — the kind used in sports science, dietetics, and coaching certifications.
- Is **always available** via a web interface, without the user needing to manually read through hundreds of pages of books.

The solution is a RAG pipeline where the books are the ground truth, and the LLM is the communicator.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ONE-TIME INDEXING PHASE                      │
│   (store_index.py — run manually before the app starts)         │
│                                                                 │
│  5 PDF Books (171 MB)                                           │
│       │                                                         │
│       ▼  PyPDFLoader + DirectoryLoader                          │
│  Document objects (pages with text)                             │
│       │                                                         │
│       ▼  filter_to_minimal_docs()                               │
│  Strip all metadata except 'source' filename                    │
│       │                                                         │
│       ▼  RecursiveCharacterTextSplitter (500 chars, 20 overlap) │
│  ~N text chunks                                                 │
│       │                                                         │
│       ▼  HuggingFace all-MiniLM-L6-v2                          │
│  384-dimensional float vectors per chunk                        │
│       │                                                         │
│       ▼  PineconeVectorStore.from_documents()                   │
│  Pinecone Index: "diet-fitness-chatbot" (AWS us-east-1)         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     RUNTIME QUERY PHASE                         │
│   (app.py — runs as a Flask web server)                         │
│                                                                 │
│  User types question in chat.html                               │
│       │                                                         │
│       ▼  jQuery AJAX POST to /get                               │
│  Flask receives msg                                             │
│       │                                                         │
│       ▼  Query embedded → similarity search in Pinecone         │
│  Top-5 most relevant text chunks retrieved                      │
│       │                                                         │
│       ▼  create_stuff_documents_chain                           │
│  Chunks stuffed into {context} of system_prompt                 │
│       │                                                         │
│       ▼  ChatGoogleGenerativeAI (Gemini 2.5 Flash Lite, T=0.3)  │
│  LLM generates grounded, specific answer                        │
│       │                                                         │
│       ▼  response["answer"] returned to browser                 │
│  Chat UI renders bot bubble                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Concepts — From First Principles

### 4.1 What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It is a design pattern that combines two things:

- **Retrieval**: Searching a knowledge base for documents relevant to a query.
- **Generation**: Using an LLM to generate a natural language answer, augmented by the retrieved documents.

The key insight is that LLMs have a **fixed knowledge cutoff** and **limited context windows**. You cannot feed 171 MB of books into an LLM prompt — the context would be too large and too expensive. But you also don't want the LLM to rely only on what it "memorized" during training.

RAG solves this by acting as a smart search-then-summarize pipeline:
1. At query time, find the relevant 2-3 pages out of 171 MB.
2. Give ONLY those pages to the LLM.
3. The LLM generates a coherent answer from that focused context.

This makes the system both efficient (small prompts) and accurate (grounded in specific text).

### 4.2 Why Not Just Fine-Tune an LLM?

Fine-tuning is training the model on your domain data so it "learns" it. This sounds appealing, but has several problems for our use case:

- **Cost**: Fine-tuning a large model requires significant GPU compute and thousands of dollars.
- **Data format**: Fine-tuning works best with question-answer pairs. We have PDFs, not labeled datasets.
- **Staleness**: If the books are updated or new books added, you must re-fine-tune. With RAG, you just re-index.
- **No transparency**: A fine-tuned model cannot tell you where it learned something. RAG can return the exact source chunks used.
- **Hallucination**: Fine-tuning reduces hallucination but doesn't eliminate it, and the model can still conflate information from training data. RAG pins the answer to specific retrieved text.

For our use case — a fixed set of authoritative books — RAG is cheaper, faster to update, and more reliable.

### 4.3 What are Vector Embeddings?

A vector embedding is a **list of floating-point numbers** that represents the semantic meaning of a piece of text. The key property is: **texts with similar meaning have vectors that are geometrically close to each other**.

For example:
- "How many calories in chicken breast?" → `[0.23, -0.41, 0.87, ..., 0.12]` (384 numbers)
- "Caloric content of grilled chicken" → `[0.25, -0.39, 0.85, ..., 0.11]` (very close)
- "How to do a squat?" → `[0.61, 0.22, -0.43, ..., 0.78]` (far away)

This is done by a neural network (the embedding model) that has been trained to map semantically similar sentences to nearby points in a 384-dimensional space.

We use `sentence-transformers/all-MiniLM-L6-v2` — a lightweight but high-quality model that produces 384-dimensional vectors. It is fast, runs locally (no API call needed for embedding), and is sufficient for retrieval tasks.

### 4.4 What is a Vector Database?

A vector database is a specialized database optimized for storing and searching vector embeddings. Unlike SQL (exact match: `WHERE name = 'John'`) or Elasticsearch (keyword match), a vector DB does **approximate nearest neighbor (ANN) search** — given a query vector, find the k stored vectors that are closest to it.

We use **Pinecone**, a managed, serverless vector database hosted on AWS. We chose it because:
- Fully managed (no server setup, no index maintenance).
- Serverless tier is available (pay per query, not per hour).
- Native LangChain integration via `langchain-pinecone`.
- Handles scaling automatically.

Our Pinecone index `diet-fitness-chatbot` stores:
- Dimension: 384 (matching our embedding model)
- Metric: Cosine similarity
- Cloud: AWS, Region: us-east-1

### 4.5 Cosine Similarity

Cosine similarity measures the **angle** between two vectors, not their magnitude. The formula is:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Result of 1.0 = identical direction (semantically identical).
- Result of 0.0 = perpendicular (completely unrelated).
- Result of -1.0 = opposite meaning.

We use cosine instead of Euclidean (L2) distance because text embeddings tend to have varying magnitudes but consistent directions. Cosine is invariant to the magnitude of the vectors, making it more robust for semantic similarity.

### 4.6 What is LangChain?

LangChain is an orchestration framework for building applications with LLMs. It provides abstractions for:
- **Document loaders**: Standardized way to load PDFs, Word docs, URLs, etc.
- **Text splitters**: Chunking documents into pieces.
- **Embeddings**: Wrapping embedding model APIs.
- **Vector stores**: Unified interface to Pinecone, ChromaDB, FAISS, etc.
- **Chains**: Composing sequences of LLM calls and retrieval steps.
- **Prompts**: Template management.

Without LangChain, you would write all the glue code yourself — manually calling Pinecone's API, formatting prompts, chaining retrieval and generation. LangChain makes all of this declarative and composable.

---

## 5. Data & Knowledge Base

The knowledge base consists of five professional books covering the full spectrum of fitness and nutrition science:

| Book | Why Chosen |
|------|------------|
| **Advanced Nutrition and Human Metabolism, 7th Ed.** (Gropper, 2017, CENGAGE) | Biochemical foundation — macronutrients, micronutrients, metabolism at a cellular level |
| **Delavier's Women's Strength Training Anatomy Workouts** (2014, Human Kinetics) | Practical exercise instruction with anatomical context |
| **Dietary Guidelines for Americans, 2020-2025** (USDA/HHS) | Official US government dietary recommendations — authoritative reference |
| **Essentials of Strength Training and Conditioning, 4th Ed.** (NSCA, 2021) | The gold standard textbook for certified strength & conditioning coaches |
| **The Complete Guide to Sports Nutrition, 9th Ed.** (Bean, 2022, Bloomsbury) | Practical sports nutrition for athletes — supplements, race-day fueling, hydration |

These books collectively cover: macros, micros, meal planning, exercise physiology, workout programming, recovery, hydration, and supplement science. The diversity ensures the chatbot can answer across a wide range of fitness-related queries.

Total raw data: ~171 MB of PDF content.

---

## 6. Data Pipeline — The Indexing Phase

**File: `store_index.py`** — This is run **once** before the application launches. It ingests all PDF content into Pinecone. Once the index is populated, this script never needs to run again unless the knowledge base changes.

### 6.1 `load_pdf_file()`

**File: `src/helper.py`**

```python
def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents
```

**What it does**: Uses LangChain's `DirectoryLoader` to recursively scan the `data/` directory for all `.pdf` files and load them using `PyPDFLoader`.

**Why `DirectoryLoader`**: Instead of loading each PDF manually in a loop, `DirectoryLoader` handles the glob pattern matching, instantiates the correct loader per file, and returns a unified list. It is the standard LangChain pattern for bulk document ingestion.

**What `PyPDFLoader` does**: Uses the `pypdf` library under the hood to parse PDF page by page. Each page becomes a separate `Document` object containing:
- `page_content`: The extracted raw text of that page.
- `metadata`: A dict including `source` (the file path) and `page` (page number).

After loading all 5 PDFs (~total thousands of pages), we have a list of Document objects, one per page.

### 6.2 `filter_to_minimal_docs()`

**File: `src/helper.py`**

```python
def filter_to_minimal_docs(docs):
    minimal = []
    for doc in docs:
        minimal.append(Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "")}
        ))
    return minimal
```

**What it does**: Creates new Document objects keeping only the `source` field in metadata, discarding the `page` number and any other metadata.

**Why**: When LangChain upserts vectors to Pinecone, the metadata dict gets stored alongside each vector. Large metadata objects inflate the payload size and can hit Pinecone's per-record metadata limits. By stripping everything except the source filename, we keep the payload lean. The `source` is preserved because it lets us log which book each retrieved chunk came from at query time — useful for debugging and traceability.

### 6.3 `text_split()`

**File: `src/helper.py`**

```python
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
```

**What it does**: Splits each document's text into smaller chunks of 500 characters with a 20-character overlap.

**Why chunking is necessary**: Embedding models have a maximum input length (typically 256-512 tokens). A full PDF page can have thousands of characters — far too large. We must split text into sizes the embedding model can handle.

**Why `RecursiveCharacterTextSplitter`**: This splitter tries to split on natural boundaries in order of preference: `\n\n` (paragraphs) → `\n` (lines) → ` ` (words) → individual characters. This means it tries to preserve semantic coherence by not cutting mid-sentence. It is almost always the right default for unstructured text.

**Chunk size = 500 characters**: This is a balance between:
- Too small: Each chunk lacks enough context for the embedding to capture meaning. Retrieved chunks are not helpful enough.
- Too large: Exceeds embedding model limits. Retrieved chunks contain too much irrelevant text mixed with the relevant part.

**Chunk overlap = 20 characters**: If a concept spans the boundary between two chunks, the 20-character overlap ensures neither chunk loses the connective information completely.

### 6.4 `download_hugging_face_embeddings()`

**File: `src/helper.py`**

```python
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
```

**What it does**: Loads the `all-MiniLM-L6-v2` sentence transformer model from HuggingFace Hub. On first call, it downloads the model weights (~22 MB) to a local cache. Subsequent calls load from cache.

**Why this model specifically**:
- **Dimension 384**: Small enough for fast retrieval. Pinecone's free tier supports up to 1536 dims, so 384 is very efficient.
- **MiniLM architecture**: A distilled (compressed) version of a larger BERT-based model. It retains most of the semantic quality with a fraction of the compute cost.
- **L6**: 6 transformer layers — lightweight and fast.
- **Free**: No API key required. The model runs locally, so embedding is free regardless of volume.
- **Battle-tested for retrieval**: The `sentence-transformers` library specifically optimizes these models for semantic similarity tasks, which is exactly what RAG retrieval requires.

**This function is called in both** `store_index.py` (to embed chunks during indexing) **and** `app.py` (to embed user queries at runtime). Both must use the **same model** — the query embedding must be in the same vector space as the stored chunk embeddings for similarity search to work.

### 6.5 Pinecone Index Creation & Upsert

```python
pc = Pinecone(api_key=pinecone_api_key)
index_name = "diet-fitness-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)
```

**Index creation**: Checks if the index exists before creating — idempotent, safe to run repeatedly.

- `dimension=384` must match the embedding model output size exactly.
- `metric="cosine"` — we want semantic similarity, not distance.
- `ServerlessSpec(cloud="aws", region="us-east-1")` — serverless Pinecone auto-scales to zero when not in use, paying only per query. No fixed server cost.

**`from_documents()`**: This is a LangChain convenience method that:
1. Iterates through all text chunks.
2. Calls the embedding model on each chunk to produce a 384-float vector.
3. Upserts each `(vector, metadata, text)` tuple into Pinecone in batches.

After this runs, Pinecone contains a searchable index of every 500-character segment of all 5 books.

---

## 7. Query Pipeline — The Runtime Phase

**File: `app.py`** — This is the live web server. It runs continuously, handling user requests.

### 7.1 Flask App Startup

```python
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
```

**`load_dotenv()`**: Reads the `.env` file and injects all key-value pairs into the process environment. This lets us separate credentials from code. In production (Docker), these are passed as `-e` flags and `load_dotenv()` becomes a no-op (environment already set).

The double assignment (`os.environ.get()` then `os.environ["KEY"] = ...`) is redundant but harmless — it explicitly ensures the keys are available as environment variables for downstream libraries (Pinecone SDK and Google SDK both read from environment variables internally).

### 7.2 The Retriever

```python
embeddings = download_hugging_face_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name="diet-fitness-chatbot",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})
```

**`from_existing_index()`**: Connects to an already-populated Pinecone index. This does NOT re-index. It just sets up a client that knows how to query the existing index.

**`as_retriever()`**: Wraps the vector store in a LangChain `Retriever` interface. This standard interface is what the rest of the chain expects.

- `search_type="similarity"`: Use cosine similarity to find relevant chunks.
- `k=5`: Retrieve the top-5 most similar chunks per query.

**Why k=5**: This is a tunable hyperparameter. With k=5, we retrieve approximately 5 × 500 = 2,500 characters of context. This is enough to give the LLM substantial grounding without being so large that it dilutes focus or exceeds prompt token limits.

**How retrieval works at query time**:
1. User query: "How much protein should I eat to build muscle?"
2. This text is passed through `all-MiniLM-L6-v2` → 384-dimensional vector.
3. Pinecone performs cosine similarity search across all stored vectors.
4. The 5 chunks with highest cosine similarity scores are returned.
5. These are the most semantically relevant passages from the books.

### 7.3 Prompt Engineering

**File: `src/prompt.py`**

```python
system_prompt = (
    "You are an expert Diet and Fitness Coach. Your goal is to provide "
    "practical, actionable advice to help users achieve their health goals. "
    "\n\n"
    "Guidelines:\n"
    "- Give specific, helpful answers with real numbers (calories, reps, sets, etc.)\n"
    "- For exercises, describe the movement and recommend sets/reps\n"
    "- For nutrition, provide calorie estimates and portion guidance\n"
    "- Use your general knowledge combined with the context below\n"
    "- Be encouraging and supportive\n"
    "- If asked about weight status, use BMI as a reference (weight in kg / height in m²)\n"
    "\n\n"
    "Reference context:\n"
    "{context}"
)
```

**Why a system prompt**: System prompts set the **persona, behavior, and constraints** of the LLM. Without it, Gemini defaults to a generic assistant. With it, it becomes a fitness coach that always gives numbers, always structures exercise answers with sets/reps, and always acknowledges BMI as the weight status metric.

**The `{context}` placeholder**: This is a LangChain template variable. At runtime, the `create_stuff_documents_chain` function replaces `{context}` with the concatenated text of all 5 retrieved chunks. The LLM sees the full system prompt including the retrieved book passages, then answers the human's question.

**Prompt structure**:
```
[System]: You are an expert fitness coach... {context filled with retrieved book text}
[Human]: {user's question}
[Assistant]: {LLM generates answer here}
```

This is the standard chat format for instruction-following models. The system message establishes the rules; the human message is the query; the assistant generates the response.

**Temperature = 0.3**: Temperature controls randomness. 0.0 = fully deterministic, 1.0 = highly creative/random. 0.3 is deliberately low because:
- Fitness advice should be consistent and factual, not creative.
- We want the LLM to prioritize the retrieved context over hallucinated content.
- Low temperature makes the model "cling" closer to the input context.

### 7.4 The LLM — Google Gemini 2.5 Flash Lite

```python
chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
```

**Why Gemini over GPT-4 or Claude**:
- Generous free tier from Google AI Studio.
- `gemini-2.5-flash-lite` is the fastest, cheapest variant of Gemini 2.5 — suitable for a demo/learning project.
- `langchain-google-genai` provides native LangChain integration.

**Flash Lite variant**: Optimized for latency and cost. The "lite" designation means smaller parameter count, faster inference, lower cost per token. For a chatbot where response speed matters more than peak accuracy, this is the right choice.

### 7.5 LangChain Chains

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**`ChatPromptTemplate.from_messages()`**: Creates a reusable prompt template. The `{input}` placeholder will be filled with the user's message at invocation time.

**`create_stuff_documents_chain()`**: A LangChain chain that "stuffs" all retrieved documents into the `{context}` of the prompt. "Stuff" means concatenate — it literally concatenates all retrieved chunk texts and inserts them into the prompt. This is the simplest document chain strategy (alternatives include Map-Reduce and Refine, which are better for very long documents but more complex and slower).

**`create_retrieval_chain()`**: Wraps the retriever and the document chain into one end-to-end chain. When invoked with `{"input": user_message}`:
1. Passes `user_message` to the retriever → gets back `context` (list of Document objects).
2. Passes `user_message` + `context` to the `question_answer_chain`.
3. Returns `{"answer": str, "context": [Document, ...]}`.

This composition pattern is the core of LangChain — each component has a clean interface, and they snap together like LEGO blocks.

### 7.6 The /get Endpoint

```python
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"\n{'='*50}")
    print(f"Question: {input}")
    response = rag_chain.invoke({"input": msg})

    # Print retrieved sources for verification
    print(f"\nRetrieved {len(response['context'])} source chunks:")
    for i, doc in enumerate(response['context'], 1):
        source = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i}. {source}")
        print(f"     Preview: {preview}...")

    print(f"\nResponse: {response['answer']}")
    print(f"{'='*50}\n")
    return str(response["answer"])
```

**Flow**:
1. Reads `msg` from the POST form data.
2. Invokes the full RAG chain — retrieval + generation — as one call.
3. Logs each retrieved source chunk to the server console (for debugging/verification — you can see exactly which book passages were used).
4. Returns only `response["answer"]` as a plain string to the browser.

**Why return plain string, not JSON**: The frontend jQuery code uses `.done(function(data) {...})` and directly inserts `data` into the HTML bubble. A plain string is simpler than parsing JSON for a single field.

**The `/` route** serves the static `chat.html` template — the landing page of the application.

```python
app.run(host="0.0.0.0", port=8080, debug=True)
```

- `host="0.0.0.0"`: Binds to all network interfaces. Required inside Docker and EC2 — if you bind to `127.0.0.1`, the container's Flask server cannot be reached from outside the container.
- `port=8080`: Docker exposes port 8080 and the CI/CD pipeline maps 8080:8080.

---

## 8. Frontend — The Chat UI

**File: `templates/chat.html`** + **`static/style.css`**

The frontend is a **single-page application** built with pure CSS and jQuery — no Bootstrap, no React, no external UI framework. Deliberately minimal: the backend is the complexity, the frontend just needs to be clean and fast.

### Design System — Apple Dark Theme

The UI uses a CSS variable-based design token system inspired by Apple's dark palette:

```css
:root {
  --bg:           #000000;
  --surface:      #1C1C1E;   /* iOS card background */
  --surface-2:    #2C2C2E;   /* Bot bubble background */
  --accent:       #30D158;   /* Apple system green */
  --font: -apple-system, BlinkMacSystemFont, "SF Pro Display", ...
}
```

Every colour, spacing, and radius references a variable — changing the theme requires editing only the `:root` block.

### Structure

```
.app (flex column, max-width 700px, centered)
├── .chat-header (frosted glass, backdrop-filter: blur(24px))
│   ├── avatar + online dot (animated pulse)
│   ├── "Fitness & Diet Coach" title + subtitle
│   └── "AI Powered" badge (top right)
├── .messages (flex column, scrollable, gap: 18px)
│   ├── .bot-row: left-aligned, avatar + dark bubble
│   └── .user-row: right-aligned, green accent bubble
├── #typing-indicator (3 bouncing dots, hidden by default)
└── .input-area (frosted glass bar, pill-shaped input)
    └── #send-btn (circular green button, disabled while waiting)
```

### Markdown Rendering

Bot responses from Gemini are markdown. The UI uses `marked.js` to render them:

```javascript
marked.setOptions({ breaks: true, gfm: true });
// In the AJAX done handler:
<div class="md">${marked.parse(data)}</div>
```

The `.md` CSS class styles all markdown elements — headings, bold text, bullet lists, code blocks, horizontal rules — within the bot bubble, so structured answers display correctly.

### AJAX Communication

```javascript
$('#chat-form').on('submit', function(e) {
    e.preventDefault();
    const text = $('#user-input').val().trim();

    // Append user bubble immediately (optimistic UI)
    $('#messages').append(userHtml);
    $('#send-btn').prop('disabled', true);  // Prevent double-send
    showTyping();                            // Show bouncing dots

    $.ajax({ data: { msg: text }, type: 'POST', url: '/get' })
    .done(function(data) {
        hideTyping();
        $('#messages').append(`<div class="md">${marked.parse(data)}</div>`);
    })
    .fail(function() {
        hideTyping();
        // Render red error bubble
    })
    .always(function() {
        $('#send-btn').prop('disabled', false);
    });
});
```

**Key design decisions**:
- **`e.preventDefault()`**: Stops the browser from doing a full-page form POST.
- **Optimistic UI**: User bubble appended before server responds — snappy feel.
- **Send button disabled during request**: Prevents sending duplicate messages while waiting.
- **Typing indicator**: Three bouncing dots shown while the RAG chain runs (can take 2-5 seconds). The middle dot turns green at the bounce peak.
- **Error bubble**: If the AJAX call fails, a red-tinted bubble appears instead of nothing.
- **`marked.parse(data)`**: Converts Gemini's markdown output to HTML so bullet lists, bold text, and headers render properly.

---

## 9. Package Structure & Setup

**File: `setup.py`**

```python
setup(
    name="fitness_diet_chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[]
)
```

**Why `setup.py`**: The `src/` directory contains `helper.py` and `prompt.py` with the package init `src/__init__.py`. Without `setup.py`, Python cannot resolve `from src.helper import ...` when running from the project root in some environments.

**`-e .` in requirements.txt**: The `-e` flag means "editable install." It installs the current directory as a Python package in development mode, creating a `.egg-link` file that points back to the source directory. This means:
- `from src.helper import download_hugging_face_embeddings` resolves correctly.
- Any changes to `src/helper.py` take effect immediately without reinstalling.
- Works identically inside Docker (full install) and in local development.

---

## 10. Containerization with Docker

**File: `Dockerfile`**

```dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "app.py"]
```

**Line-by-line explanation**:

- `FROM python:3.10-slim-buster`: Base image. `slim-buster` is a minimal Debian Buster image with Python 3.10. "Slim" means it strips unnecessary OS utilities, keeping the image small (~150 MB base vs ~900 MB for the full image). Python 3.10 specifically was chosen for compatibility — some ML libraries had issues with 3.11+ at time of development.

- `WORKDIR /app`: Sets the working directory inside the container. All subsequent commands run relative to `/app`. Also creates the directory if it doesn't exist.

- `COPY . /app`: Copies the entire project directory into the container's `/app`. This includes `app.py`, `src/`, `templates/`, `static/`, `requirements.txt`, and `setup.py`. Notably, the large `data/` PDFs are also copied — they are not needed at runtime (only during `store_index.py`) but this is an inefficiency in the current setup.

- `RUN pip install -r requirements.txt`: Installs all dependencies during the build phase. This layer is cached by Docker — if `requirements.txt` hasn't changed, Docker reuses the cached layer and skips reinstallation on subsequent builds.

- `CMD ["python3", "app.py"]`: The default command to run when the container starts. Uses the exec form (`["..."]`) rather than shell form (`"..."`) — exec form runs the process directly as PID 1, which means Docker signals (like SIGTERM on container stop) are received by Flask directly.

**Running the container** (from the CI/CD pipeline):
```bash
docker run -d \
  -e PINECONE_API_KEY="..." \
  -e GOOGLE_API_KEY="..." \
  -p 8080:8080 \
  <ecr-registry>/<repo>:latest
```

- `-d`: Detached mode (runs in background).
- `-e KEY=VALUE`: Injects environment variables at runtime (no `.env` file needed in production).
- `-p 8080:8080`: Maps port 8080 on the host EC2 instance to port 8080 inside the container.

---

## 11. CI/CD Pipeline — GitHub Actions + AWS EC2

**File: `.github/workflows/cicd.yaml`**

The pipeline uses a single job that SSHs into the EC2 instance and redeploys the app — no Docker build, no ECR, no self-hosted runner required.

### The Full Pipeline

```yaml
name: Deploy to EC2

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: SSH into EC2 and redeploy
        uses: appleboy/ssh-action@v1.0.0
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /home/ubuntu/fitness-diet-chatbot
            git pull origin main
            source venv/bin/activate
            pip install -r requirements.txt -q
            sudo systemctl restart fitness-chatbot
            echo "deployed"
```

### What happens on every push to `main`

1. GitHub Actions spins up a free `ubuntu-latest` cloud runner.
2. `appleboy/ssh-action` establishes an SSH connection to the EC2 instance using the private key stored in GitHub Secrets.
3. The script runs on EC2:
   - `git pull origin main` — fetches and applies the latest code changes.
   - `source venv/bin/activate` — activates the Python virtual environment.
   - `pip install -r requirements.txt -q` — installs any new dependencies (no-op if requirements unchanged).
   - `sudo systemctl restart fitness-chatbot` — tells systemd to stop the running app and start the updated version.
4. SSH connection closes. App is now running new code.

### Why SSH + systemd instead of Docker + ECR

An ECR-based pipeline requires: building a Docker image, pushing to ECR (storage cost + transfer time), pulling on EC2, running the container. For a Python app without complex system dependencies this is unnecessary overhead.

The SSH approach is simpler: no Docker build, no container registry, no ECR costs. Deployment takes under 30 seconds. The trade-off is no image versioning — to roll back you `git revert` and push.

### Why systemd manages the process

Early attempts used `nohup python3 app.py &` to keep the process alive after SSH disconnects. This consistently failed with exit code 143 (SIGTERM) — the appleboy/ssh-action's Docker container received SIGTERM every time the SSH session ended, even with `nohup`, `disown`, and subshell approaches.

The root cause: background processes started within an SSH session can remain attached to the session's process group. When SSH closes, the OS sends SIGTERM to the group.

systemd solves this definitively. The app process is owned by systemd, not by any SSH session. `sudo systemctl restart fitness-chatbot` hands control to systemd and returns immediately — nothing dangles in the SSH session.

**Additional benefits of systemd**:
- `Restart=always` automatically restarts the app if it crashes.
- App starts on EC2 reboot (`systemctl enable`).
- Logs available via `journalctl -u fitness-chatbot`.

### systemd Service File

`/etc/systemd/system/fitness-chatbot.service`:

```ini
[Unit]
Description=Fitness Diet Chatbot
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/fitness-diet-chatbot
EnvironmentFile=/home/ubuntu/fitness-diet-chatbot/.env
ExecStart=/home/ubuntu/fitness-diet-chatbot/venv/bin/python3 app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

### GitHub Secrets Required

| Secret | Value |
|---|---|
| `EC2_HOST` | `35.173.100.45` (Elastic IP — permanent) |
| `SSH_PRIVATE_KEY` | Full contents of the `.pem` key file |
| `PINECONE_API_KEY` | Pinecone API key |
| `GOOGLE_API_KEY` | Google AI Studio API key |

---

## 12. Challenges, Trade-offs & Decisions

### Challenge 1: Chunking Size Tuning
**Problem**: Initial runs with very large chunks (1000+ chars) produced retrieved context that was too broad — the LLM was getting an entire nutrition chapter when it only needed one paragraph.
**Solution**: Tuned to 500-char chunks with 20-char overlap. This keeps chunks focused enough for semantic similarity to work while still providing enough context for coherent answers.

### Challenge 2: Metadata Bloat in Pinecone
**Problem**: PyPDFLoader adds extensive metadata per page — source path, page number, file stats. When upserting thousands of chunks, this metadata per record was inflating the Pinecone upsert payload and slowing ingestion.
**Solution**: Created `filter_to_minimal_docs()` to strip everything except the source filename before chunking and upserting. Reduces metadata per record from ~500 bytes to ~50 bytes.

### Challenge 3: Free vs Accurate LLM
**Problem**: GPT-4 is more capable but costs per token — for a learning project with potentially heavy usage, this is prohibitive.
**Solution**: Google Gemini 2.5 Flash Lite has a generous free tier and is fast enough for conversational latency. Temperature 0.3 compensates for slightly lower accuracy by making the model more conservative and context-driven.

### Challenge 4: Embedding Model Choice
**Problem**: Using the OpenAI embedding API would cost money per embed call. Local embedding avoids this but requires the right model.
**Solution**: `all-MiniLM-L6-v2` from HuggingFace — free, fast, 384-dim vectors, excellent for retrieval tasks, runs on CPU without GPU requirements.

### Challenge 5: Environment Variables in Docker
**Problem**: The `.env` file should not be committed (security risk), but Docker needs the variables at runtime.
**Solution**: The Docker run command passes variables via `-e` flags. In the CI/CD pipeline, they come from GitHub Secrets. This way, the `.env` file is only used in local development, never baked into the image.

### Challenge 6: Port Binding in Docker
**Problem**: Flask defaults to `127.0.0.1` which means only local loopback — unreachable from outside the container.
**Solution**: `app.run(host="0.0.0.0")` binds to all interfaces, making Flask reachable on the container's published port.

### Challenge 7: EC2 Disk Space — Out of Space During pip install
**Problem**: The default EC2 instance comes with 8GB of EBS storage. During `pip install -r requirements.txt`, the installation failed mid-way with `OSError: [Errno 28] No space left on device`. The OS + Python + partial packages had consumed the entire 8GB.
**Solution**: AWS free tier allows up to 30GB of EBS storage. Expanded the volume to 20GB via the AWS Console (EC2 → Volumes → Modify Volume), then resized the filesystem on the instance:
```bash
sudo growpart /dev/nvme0n1 1
sudo resize2fs /dev/nvme0n1p1
```
Also used `pip install --no-cache-dir` to prevent pip from caching downloaded packages, further reducing disk usage.

### Challenge 8: EC2 SSH Connection Timeout
**Problem**: After launching the EC2 instance with port 22 open in the security group, direct SSH from a local machine timed out consistently. The security group was correctly configured (confirmed visually), so the firewall was not the issue.
**Solution**: Diagnosed the problem using AWS EC2 Instance Connect (browser-based SSH in the AWS console), which bypassed the local network entirely and connected successfully. The root cause was the local ISP blocking outbound port 22. All EC2 setup was done via EC2 Instance Connect. The CI/CD pipeline's SSH (via GitHub Actions' cloud infrastructure) works fine since it originates from GitHub's servers, not the local network.

### Challenge 9: Background Process Killed on SSH Disconnect (exit code 143)
**Problem**: The CI/CD script started the app with `nohup python3 app.py &`, but the GitHub Actions pipeline reported exit code 143 (SIGTERM) on every deploy. Even `disown` and subshell approaches (`(nohup ... &)`) failed.
**Root cause**: Background processes started within an SSH session can remain in the session's process group. When the SSH action's Docker container closes the connection, the OS sends SIGTERM to the process group — the action container itself received this signal.
**Solution**: Replaced the `nohup` approach with systemd. The app is now a managed systemd service. CI/CD runs `sudo systemctl restart fitness-chatbot` — systemd handles the process completely outside any SSH session. The pipeline now exits cleanly with code 0.

### Challenge 10: VPC Subnet Missing Internet Gateway Route
**Problem**: Although the EC2 instance had a public IP and the security group allowed all traffic, the EC2 Instance Connect initially showed the instance as unreachable.
**Diagnosis**: Checked VPC → Route Tables for the subnet. The route table had only a local VPC route (`172.31.0.0/16 → local`) with no `0.0.0.0/0 → igw-xxx` route. Without an Internet Gateway route, the subnet is effectively private — no internet connectivity despite the public IP.
**Solution**: Added a `0.0.0.0/0 → igw-xxxxxxxxx` route to the route table. The Internet Gateway already existed and was attached to the VPC — it just needed a route entry.

---

## 13. Interview Questions & In-Depth Answers

---

### SECTION A: PROJECT FUNDAMENTALS

**Q1: Can you explain what this project does in simple terms?**

This is a web-based chatbot that acts as a personal fitness and diet coach. When a user asks a health question, the system searches through 5 professional textbooks on nutrition and strength training to find the most relevant passages, then feeds those passages to Google's Gemini AI model to generate a specific, grounded answer. The user sees responses with real numbers — like "eat 1.6–2.2g of protein per kg of bodyweight" — backed by referenced source material, not generic internet content.

---

**Q2: What is RAG and why did you use it instead of fine-tuning?**

RAG stands for Retrieval-Augmented Generation. It's a technique where you first retrieve relevant documents from a knowledge base, then use those documents as context for an LLM to generate a response.

I chose RAG over fine-tuning for several reasons:
- Fine-tuning is expensive — it requires labeled training data and GPU compute.
- Fine-tuning is brittle — if the books change, you re-train. With RAG, you just re-index.
- RAG is transparent — I can log exactly which source chunks were used for each answer. Fine-tuning bakes knowledge into weights silently.
- RAG reduces hallucination more effectively for factual queries — the LLM is constrained to the retrieved text.
- My data is in PDF format, not structured Q&A pairs — RAG works directly with raw documents.

---

**Q3: Walk me through what happens when a user sends "How much protein do I need per day?"**

1. The user types in the chat UI and submits the form.
2. jQuery captures the form submit, prevents the default page reload, and sends a `POST` request to `/get` with `{msg: "How much protein do I need per day?"}`.
3. Flask's `chat()` function receives the message.
4. The message text is passed through `all-MiniLM-L6-v2` to produce a 384-dimensional vector.
5. Pinecone performs cosine similarity search and returns the top-5 chunk vectors with their stored text.
6. LangChain's `create_stuff_documents_chain` concatenates those 5 chunks and inserts them into the `{context}` field of the system prompt.
7. The assembled prompt — system prompt with context + user's question — is sent to Gemini 2.5 Flash Lite.
8. Gemini generates a response like "Research suggests 1.6–2.2g of protein per kg of bodyweight per day for muscle growth. For a 70kg person, that's 112–154g of protein daily..."
9. Flask returns this string, jQuery renders it as a left-aligned blue message bubble in the chat.

---

**Q4: Why is `store_index.py` run only once?**

Pinecone is a persistent vector database. Once the PDFs are indexed — chunked, embedded, and upserted — those vectors live in Pinecone's managed storage indefinitely. The app (at startup) connects to the existing index with `PineconeVectorStore.from_existing_index()` — it does NOT re-index. Re-running `store_index.py` would duplicate all vectors in the index, causing redundant results.

You only re-run `store_index.py` when the knowledge base changes — e.g., adding a new book, updating a book edition, or changing the chunk size.

---

### SECTION B: VECTOR EMBEDDINGS & SIMILARITY SEARCH

**Q5: What exactly is a vector embedding?**

A vector embedding is a mathematical representation of text as a list of floating-point numbers (a vector in N-dimensional space). The key property is that the geometric distance between two vectors reflects their semantic similarity.

The `all-MiniLM-L6-v2` model is a neural network trained to map text sentences to 384-dimensional space such that sentences with similar meaning are near each other. For example, "How many calories in an egg?" and "Caloric value of eggs" would produce vectors that are very close (high cosine similarity ~0.9), while "What is a deadlift?" would produce a vector far away.

This allows us to search for semantically similar content, not just keyword matches.

---

**Q6: Why cosine similarity and not Euclidean distance?**

Cosine similarity measures the angle between two vectors, ignoring their magnitude. This is better for text embeddings because:

Text embeddings can have varying magnitudes depending on the length and complexity of the text, but the direction (angle) consistently encodes semantic meaning. A short sentence and a long sentence about the same topic will have similar directions but different magnitudes. Cosine similarity would correctly rate them as similar; Euclidean distance would penalize the magnitude difference.

Mathematically: `cos(θ) = (A · B) / (||A|| × ||B||)`. Values range from -1 to 1, where 1 means identical direction (semantically identical), 0 means orthogonal (unrelated), -1 means opposite meaning.

---

**Q7: Why `all-MiniLM-L6-v2` specifically? Why not use OpenAI's embedding API?**

Several reasons:
1. **Cost**: OpenAI's `text-embedding-ada-002` charges per token. With 171 MB of PDF text, indexing alone would cost money. `all-MiniLM-L6-v2` is free — it runs locally.
2. **Latency**: Local embedding has no network round-trip for every chunk and query.
3. **Quality for retrieval**: The `sentence-transformers` library specifically trains these models for semantic similarity tasks, which is our exact use case. OpenAI embeddings are more general.
4. **Simplicity**: No additional API key needed. The model is downloaded once and cached.

The 384-dimension output is a deliberate trade-off — OpenAI's ada-002 uses 1536 dimensions (more accurate but costs more to store and query). 384 is sufficient for retrieval in a focused domain.

---

**Q8: What happens if the user asks a question that is completely unrelated to fitness — like "What's the capital of France?"**

Pinecone will still return 5 chunks — it always returns the k nearest neighbors, even if those neighbors are not actually similar. For a geography question, the "nearest" fitness book chunks would be about travel nutrition or something tangentially related, but they would have low cosine similarity scores.

The LLM would receive this weak context and its system prompt says to be a fitness coach. It would likely respond with something like "That's outside my domain as a fitness coach, but Paris is the capital of France" — the general knowledge of Gemini would provide the factual answer while the system prompt keeps it in character.

This is a known limitation of basic RAG — there is no similarity threshold filtering. An improvement would be to add a similarity score filter: if all retrieved chunks score below 0.4, do not inject them as context and instead let the LLM answer from its general knowledge or explicitly redirect.

---

### SECTION C: LangChain FRAMEWORK

**Q9: What is LangChain and what specific components did you use?**

LangChain is an orchestration framework for building LLM applications. I used:

- **`DirectoryLoader` + `PyPDFLoader`**: Loading all PDFs from the data directory.
- **`RecursiveCharacterTextSplitter`**: Chunking document text.
- **`HuggingFaceEmbeddings`**: Wrapping the sentence transformer model.
- **`PineconeVectorStore`**: Interface to read/write to Pinecone.
- **`ChatPromptTemplate`**: Structuring the system + human prompt.
- **`ChatGoogleGenerativeAI`**: LangChain wrapper for Gemini.
- **`create_stuff_documents_chain`**: Stuffing retrieved docs into the prompt.
- **`create_retrieval_chain`**: End-to-end chain combining retriever + document chain.

Without LangChain, I would write all this glue code manually — looping through Pinecone results, string-formatting the prompt, making raw HTTP calls to the Gemini API. LangChain's abstractions make all of this declarative and maintainable.

---

**Q10: What is the difference between `create_stuff_documents_chain` and alternatives like Map-Reduce?**

`create_stuff_documents_chain` is the simplest strategy — it "stuffs" (concatenates) all retrieved documents into a single prompt and sends it to the LLM once. One API call, simple, fast.

**Map-Reduce**: For very long document collections, you first send each document to the LLM independently to get a summary (the Map step), then combine the summaries and send to the LLM again (the Reduce step). More API calls, more cost, more latency, but handles arbitrarily long document sets.

**Refine**: Iteratively processes documents, each time refining the answer based on the next document. Very thorough but slow.

We use Stuff because our k=5 with 500-char chunks produces ~2,500 characters of context — well within Gemini's context window and a single API call is fast enough for conversational latency.

---

### SECTION D: FLASK & BACKEND

**Q11: Why Flask and not FastAPI or Django?**

- **Flask** is lightweight, minimal, and has almost no boilerplate. For an application with just 2 routes (`/` and `/get`), Flask is perfect.
- **FastAPI** would be better if this were a REST API with async requirements, request/response validation, and OpenAPI docs. Overkill for a simple chatbot.
- **Django** is a full-stack framework with an ORM, admin panel, auth system, etc. Massive overkill for a two-route application with no database.

Flask's simplicity also makes it easier to containerize and reason about in production.

---

**Q12: Why does the app bind to `host="0.0.0.0"` instead of the default `127.0.0.1`?**

`127.0.0.1` is the loopback interface — it only accepts connections from the same machine (localhost). Inside a Docker container, the Flask process is PID 1. The "same machine" from Flask's perspective is the container's network namespace, not the host EC2.

When you do `docker run -p 8080:8080`, Docker maps port 8080 of the HOST to port 8080 of the CONTAINER. But if Flask is listening on `127.0.0.1:8080` inside the container, connections arriving from the host are rejected because they appear to come from outside the loopback interface.

`0.0.0.0` means "bind to all available network interfaces" — Flask accepts connections from any source, including those routed through Docker's `veth` (virtual ethernet) interface. This is essential for containerized deployment.

---

### SECTION E: DOCKER & DEPLOYMENT

**Q13: Explain your Dockerfile and why `python:3.10-slim-buster`?**

```dockerfile
FROM python:3.10-slim-buster  # Small base image, Python 3.10
WORKDIR /app                   # All commands relative to /app
COPY . /app                    # Copy project files into container
RUN pip install -r requirements.txt  # Install dependencies (cached layer)
CMD ["python3", "app.py"]      # Start Flask when container runs
```

`slim-buster` is a minimal Debian Buster image. "Slim" strips unnecessary OS packages, reducing the image from ~900 MB to ~150 MB base. Python 3.10 is chosen because:
- `sentence-transformers 4.x` and some ML libraries had C extension compilation issues on Python 3.11+ at the time.
- 3.10 is an LTS-like stable release in the ML ecosystem.

A potential improvement: Use multi-stage builds and a `.dockerignore` file to exclude `venv/`, `data/` (PDFs), and `.git/` from the image, significantly reducing the final image size.

---

**Q14: How does your CI/CD pipeline work end-to-end?**

When you push code to the `main` branch on GitHub:

1. **GitHub Actions triggers**: The workflow in `.github/workflows/cicd.yaml` starts on GitHub's free cloud runner.

2. **The deploy job**:
   - Uses `appleboy/ssh-action` to establish an SSH connection to the EC2 instance using a private key stored in GitHub Secrets.
   - Runs this script on EC2:
     ```bash
     cd /home/ubuntu/fitness-diet-chatbot
     git pull origin main
     source venv/bin/activate
     pip install -r requirements.txt -q
     sudo systemctl restart fitness-chatbot
     echo "deployed"
     ```
   - `git pull` applies the latest code. `systemctl restart` tells the OS-level process manager to restart the app with the new code.

3. SSH disconnects. The app is running the updated version. Total time: under 30 seconds.

This is simpler than a Docker + ECR pipeline — no image builds, no container registry, no self-hosted runner to maintain.

---

**Q15: What are GitHub Secrets and why are they used?**

GitHub Secrets are encrypted variables stored at the repository level. They are injected as environment variables into GitHub Actions workflows at runtime — never visible in logs (automatically masked), never visible to anyone viewing the repository.

We store: `EC2_HOST` (the Elastic IP), `SSH_PRIVATE_KEY` (full `.pem` file contents), `PINECONE_API_KEY`, and `GOOGLE_API_KEY`.

This is the correct way to handle credentials in CI/CD. The alternatives (hardcoding in YAML, committing `.env`) are security vulnerabilities — anyone with repo access can see those credentials. A `.env` was accidentally committed to git early in this project, which is why those keys were rotated.

---

### SECTION F: DESIGN DECISIONS & IMPROVEMENTS

**Q16: What would you improve if you had more time?**

1. **Similarity score threshold**: Filter out retrieved chunks with cosine similarity below a threshold (e.g., 0.4) so that off-topic questions don't get injected with irrelevant fitness context.

2. **Conversational memory**: The current chain has no memory — each query is independent. Adding `ConversationBufferMemory` or a sliding window of recent messages would allow follow-up questions like "How about for vegetarians?" referring to the previous protein question.

3. **Streaming responses**: Currently the user waits for the full Gemini response before seeing anything. Streaming (using LangChain's `.stream()`) would show words appearing token by token for a much snappier UX.

4. **Source citation in the UI**: Log and display which books were retrieved for each answer, not just on the server console. This builds user trust.

5. **Async Flask with async Pinecone/LLM calls**: The current setup is synchronous — one request blocks one thread. With `asyncio` and `flask[async]`, multiple users could be served concurrently.

6. **Source citation in the UI**: Currently the retrieved book sources are logged server-side only. Surfacing them in the chat bubble ("Source: The Complete Guide to Sports Nutrition, p.47") would build user trust.

---

**Q17: Why does the system prompt tell the LLM to "use your general knowledge combined with the context"?**

This is a deliberate design choice. Pure RAG — where the LLM can ONLY use retrieved context — can fail when:
- The retrieved chunks don't directly answer the question (the answer might be in the book but phrased differently).
- The user asks a simple, commonsense question that doesn't require book knowledge ("What is a calorie?").
- The retrieval step returns marginally relevant chunks.

By allowing the LLM to combine retrieved context with its own training knowledge, we get more complete answers. The trade-off is slightly more risk of hallucination — but at temperature 0.3, the LLM strongly favors the provided context over generating from scratch.

---

**Q18: How would you scale this if it had 10,000 concurrent users?**

The current synchronous Flask app would struggle at 10,000 concurrent users. A scaling path:

1. **Async**: Switch to FastAPI with async request handling and async Pinecone/Gemini clients.
2. **Multiple workers**: Use Gunicorn with multiple worker processes (`gunicorn -w 4 app:app`).
3. **Load balancer**: Put an Application Load Balancer (AWS ALB) in front of multiple EC2 instances or ECS tasks.
4. **Pinecone scales automatically**: Serverless Pinecone handles query volume elastically.
5. **LLM rate limits**: Gemini API has per-minute token limits. At scale, implement request queuing or upgrade to a higher rate limit tier.
6. **Caching**: For common questions, cache LLM responses in Redis to avoid redundant API calls.

---

**Q19: What is the difference between the Indexing phase and the Query phase? Why separate them?**

**Indexing phase** (`store_index.py`):
- Runs ONCE (or whenever books change).
- Time-intensive: parsing 171 MB of PDFs, embedding thousands of chunks, upserting to Pinecone.
- Creates persistent state in Pinecone.

**Query phase** (`app.py`):
- Runs continuously for every user request.
- Must be fast (sub-second response ideally).
- Only connects to existing Pinecone index and embeds one query.

Separating them means the user never waits for PDF parsing. If indexing were done inside `app.py` at startup, the app would take minutes to start and the first user would wait through the entire ingestion process. By pre-indexing, the app starts in seconds (just a Pinecone connection) and each query only takes the time for one embedding + one vector search + one LLM call.

---

**Q20: Someone asks you: "Isn't this just a fancy search engine?" How do you respond?**

A search engine returns documents. This system understands questions and generates answers.

When you search Google for "how much protein to build muscle," you get a list of links — you still have to read and synthesize them yourself.

When you ask this chatbot the same question, it:
1. Searches the textbooks (the search engine part, yes).
2. Extracts the 5 most relevant passages.
3. Uses an LLM to read and synthesize those passages.
4. Generates a direct, specific, natural-language answer: "For hypertrophy, the research in our nutrition textbook recommends 1.6–2.2g of protein per kg of bodyweight per day..."

The LLM adds the layer of comprehension, synthesis, and natural language generation that a search engine cannot do. A search engine is a component of our system (the retriever), but the end product is a reasoning assistant, not a list of links.

---

**Q21: What is the `RecursiveCharacterTextSplitter` and why not a simple fixed-size split?**

A naive fixed-size split would split text at every Nth character regardless of content:
- "The optimal protein intake for muscle hypertrophy is 1.6g per kg bod" / "yweight per day. This is because..."

The word "bodyweight" is cut in half — the meaning is fragmented.

`RecursiveCharacterTextSplitter` tries to split at natural language boundaries in order:
1. `\n\n` (paragraph breaks) — preserves paragraph structure.
2. `\n` (newlines) — preserves line structure.
3. ` ` (spaces) — at worst, splits between words, never mid-word.
4. `""` (characters) — last resort.

This means chunks almost always contain complete sentences and coherent ideas. Better chunks produce better embeddings, which produce better retrieval results. The quality of RAG retrieval is fundamentally limited by the quality of chunking.

---

**Q22: How do you ensure the chatbot doesn't give dangerous medical advice?**

Currently, there is no hard safety filter beyond the system prompt. The system prompt positions the bot as a "fitness coach," which is a lower-risk domain than a medical doctor — but there are still risks (e.g., extreme calorie restriction advice, supplement interactions).

Improvements for a production health application:
1. Add an explicit guideline in the system prompt: "Always recommend consulting a licensed physician or registered dietitian for medical decisions."
2. Implement a content moderation layer before the LLM call to detect and block dangerous queries.
3. Add a disclaimer UI element on the chat interface.
4. Use a safety-tuned model or fine-tune with RLHF to avoid harmful content.

For a demo/educational project, the current scope (fitness and general nutrition, not clinical medicine) is reasonably safe, but these safeguards would be essential for a public-facing product.

---

---

## 14. Generic Project Interview Questions — Behavioral & Situational

> These are the "soft" but critical questions interviewers ask about any project. Answer these with the STAR format: **Situation → Task → Action → Result**.

---

### SECTION G: PROBLEM-SOLVING & CHALLENGES

**Q23: Tell me about the biggest challenge you faced in this project and how you solved it.**

The most significant challenge was the **data ingestion and chunking pipeline**. When I first loaded the 5 PDF textbooks and indexed them naively, the chatbot answers were poor — either too generic or retrieving completely wrong passages.

The root problem was in chunking. My initial chunk size was 1000 characters with no overlap. This caused two issues:
1. Chunks were too large, so the embedding captured an average of multiple concepts rather than one focused idea. Retrieval quality suffered because a chunk about "protein metabolism" also contained text about "carbohydrate absorption," muddying the vector.
2. No overlap meant that sentences at chunk boundaries were split, losing the connecting context between adjacent chunks.

I debugged this by adding the source logging in the `/get` endpoint — printing each retrieved chunk's source and a preview to the console. This let me see in real-time what the retriever was actually pulling. I then tuned chunk size down to 500 characters with 20-character overlap, re-ran `store_index.py`, and observed significantly more focused, relevant retrievals.

The lesson: **RAG quality is bottlenecked by chunking quality**. You can't fix bad chunking with a better LLM.

---

**Q24: What went wrong in your project that you had to fix?**

Several concrete issues across the RAG pipeline and deployment:

**Issue 1 — Metadata bloat causing slow indexing**: When I first ran `store_index.py`, upserting to Pinecone was extremely slow. PyPDFLoader attaches extensive metadata per page — file stats, page numbers, author info. Each Pinecone record carried ~500 bytes of metadata across thousands of chunks.
Fix: wrote `filter_to_minimal_docs()` to strip everything except the source filename. Reduced metadata per record to ~50 bytes.

**Issue 2 — EC2 disk space exhausted mid-install**: The default EC2 8GB EBS volume was fully consumed during `pip install`. sentence-transformers and its transitive dependencies are large. The install failed with `OSError: [Errno 28] No space left on device`.
Fix: expanded the EBS volume to 20GB via the AWS Console, then resized the filesystem with `growpart` and `resize2fs`. Also added `--no-cache-dir` to pip install to prevent caching.

**Issue 3 — SSH connection timeout from local machine**: SSH to the EC2 public IP consistently timed out from my local network despite the security group correctly allowing port 22. The issue was the local ISP blocking outbound SSH.
Fix: used AWS EC2 Instance Connect (browser-based terminal in the AWS Console) to access the instance. All server setup was done via this browser shell. The CI/CD pipeline's SSH works fine since it originates from GitHub's cloud infrastructure.

**Issue 4 — VPC subnet had no Internet Gateway route**: Even with a public IP and open security group, the EC2 instance was not reachable. The route table for the subnet only had a local VPC route with no `0.0.0.0/0 → igw-xxx` entry.
Fix: added the Internet Gateway route to the VPC route table. The IGW existed and was attached — it just needed a routing entry.

**Issue 5 — CI/CD pipeline reporting exit code 143 on every deploy**: Using `nohup python3 app.py &` in the SSH deploy script kept failing with SIGTERM. Even `disown` and subshell approaches failed. The appleboy/ssh-action Docker container received SIGTERM when the SSH session closed, regardless of how the process was backgrounded.
Fix: replaced the nohup approach with systemd. The app is now a managed service (`fitness-chatbot.service`). CI/CD just calls `sudo systemctl restart fitness-chatbot` — systemd owns the process, completely outside the SSH session. Pipeline now exits cleanly.

---

**Q25: How did you debug issues in this project?**

Several strategies, depending on the layer:

1. **Console logging in the `/get` endpoint**: I added the print statements that log each retrieved chunk's source, preview, and the final LLM answer. This was the primary debugging tool for RAG quality — you can see exactly what the retriever is returning and whether the LLM answer aligns with those sources.

2. **Running scripts in isolation**: `store_index.py` has print statements at each major step (`"Loaded N pages"`, `"Split into N chunks"`, `"Uploading vectors..."`, `"Done!"`). This lets you monitor the pipeline progress and catch failures at the exact step.

3. **Pinecone dashboard**: Pinecone's web console shows index statistics — number of vectors, dimensionality, index health. I used this to confirm vectors were actually being upserted after running the indexing script.

4. **Environment variable debugging**: When the app failed to connect to Pinecone or Gemini, the first thing I checked was whether the keys were loaded. Added a quick `print(os.environ.get('PINECONE_API_KEY')[:5])` to verify the key was present without printing the full secret.

5. **Docker log inspection**: `docker logs <container_id>` to see Flask's output from inside the running container.

---

**Q26: What trade-offs did you make in this project?**

Several deliberate trade-offs:

| Decision | Trade-off Made | Reason |
|----------|---------------|--------|
| `all-MiniLM-L6-v2` (384-dim) over OpenAI embeddings (1536-dim) | Lower accuracy for zero cost | Free tier project; quality is still sufficient for retrieval |
| Gemini 2.5 Flash Lite over GPT-4 | Lower capability for lower cost | Generous free tier; adequate for Q&A tasks |
| Synchronous Flask over async FastAPI | Lower throughput for simpler code | Single developer, demo project, not expecting 1000+ concurrent users |
| "Stuff" chain over Map-Reduce | Less thorough for faster response | Context size (5 × 500 chars) fits in one prompt; latency matters for chat |
| k=5 retrieved chunks | Less coverage vs. smaller, more focused context | Balance between coverage and prompt focus |
| Docker + EC2 over serverless (Lambda) | More infrastructure to manage vs. full Flask compatibility | Lambda cold starts + 15min timeout limits don't suit a long-running ML app |

---

**Q27: How did you test your project?**

Testing was done in layers:

1. **Unit level**: Ran each function in `src/helper.py` independently — `load_pdf_file()`, `text_split()`, etc. — and printed outputs to verify expected behavior (document count, chunk sizes, embedding dimension).

2. **Integration level**: Ran `store_index.py` end-to-end and verified the Pinecone dashboard showed the expected number of vectors.

3. **Manual end-to-end testing**: Ran the Flask app locally, sent questions through the chat UI, and checked:
   - Was the answer specific and grounded (contained actual numbers)?
   - Did the server logs show relevant source chunks being retrieved?
   - Did the chat UI render correctly across different answer lengths?

4. **Docker testing**: Built and ran the Docker container locally before pushing to ECR, verified it was reachable at `localhost:8080`.

5. **Regression testing for chunking**: After changing chunk size, I re-indexed and manually compared answer quality on 5 test questions to confirm improvement.

The project does not have automated test suites (pytest, etc.) — this is a known gap. For production, I would add unit tests for each helper function and integration tests that mock the Pinecone and Gemini APIs.

---

### SECTION H: PROJECT UNDERSTANDING & OWNERSHIP

**Q28: What is the most important thing you learned building this project?**

The most important lesson was understanding the **full data lifecycle** in an ML application — from raw data (PDFs) to a user-facing product. Academic ML courses focus on model accuracy; this project forced me to deal with the messy reality of:

- **Data quality**: PDFs have inconsistent formatting, headers, footers, and tables that confuse text extractors. The extracted text is noisy.
- **Infrastructure**: A model that works locally needs Docker, port binding, environment variables, and cloud services to actually reach users.
- **The gap between "it works" and "it works well"**: The first version technically answered questions. Tuning chunking, temperature, retrieval k, and the system prompt made it actually useful.

I also learned that **RAG is fundamentally a retrieval problem before it's an LLM problem**. The LLM is capable of generating great answers — the bottleneck is giving it the right context. Most of the engineering effort went into the data pipeline, not the AI model itself.

---

**Q29: How would you explain this project to a non-technical person — say, a hiring manager who doesn't code?**

"Imagine you have 5 thick textbooks on fitness and nutrition written by PhD experts. If you asked someone a question, they'd have to flip through all 5 books to find the relevant pages, then explain it to you in plain English. That's exactly what this chatbot does — but in seconds.

When you type a question, the system does a smart search through all 5 books to find the most relevant paragraphs. Those paragraphs are then given to an AI, like a super-smart reading assistant, which reads them and writes you a clear, specific answer — like a personal trainer who has memorized every nutrition textbook.

The chatbot lives on the internet, so you can access it from any browser. I also set it up so that every time I update the code, it automatically deploys to the server without any manual work."

---

**Q30: If you were to redo this project from scratch, what would you do differently?**

1. **Start with a `.dockerignore` and proper project structure** — I wasted time in later stages because the Docker image included unnecessary files (171 MB of PDFs, the local venv). Setting these up from day one saves build time.

2. **Never commit the `.env` file** — I caught this but the keys were already in git history. I would use `git-crypt` or rely entirely on environment variables from the start, with a `.env.example` for documentation.

3. **Design the chunking strategy before ingesting** — I iterated on chunk size after the fact, which meant re-running the entire indexing pipeline multiple times. A small-scale test on one book section would have informed the parameters upfront.

4. **Add streaming from day one** — Users hate waiting. Streaming LLM responses is a huge UX improvement and LangChain supports it natively. It's easier to build in than to add later.

5. **Fix the CI/CD secret naming** (`OPENAI_API_KEY` → `GOOGLE_API_KEY`) before writing the pipeline — a small mistake that would cause silent deployment failures.

6. **Add a similarity threshold** to the retriever — so off-topic questions don't get injected with irrelevant fitness context.

---

**Q31: Was this a solo project or did you work in a team? How did you manage the work?**

This was a solo project. Managing a solo ML project requires disciplined self-organization because there's no teammate to catch mistakes or provide review.

My approach:
- **Phase separation**: I completed the indexing pipeline and confirmed it worked (Pinecone populated correctly) before writing a single line of the Flask app. End-to-end validation at each phase prevents debugging compound failures.
- **Git commits as checkpoints**: Each working milestone got its own commit — folder structure, requirements, web app completion, CI/CD. This gave clean rollback points.
- **Console logging as a second pair of eyes**: The verbose logging in `/get` serves as automated "review" — every request shows exactly what the system retrieved and what it responded with.

---

**Q32: How long did this project take to build? What took the most time?**

The most time-intensive parts were not the ones I initially expected:

1. **PDF data collection and selection**: Finding high-quality, relevant books that cover the domain comprehensively without overlap took significant curation time. A bad knowledge base produces a bad chatbot regardless of the engineering.

2. **Chunking iteration**: Running `store_index.py` takes 10-15 minutes for 171 MB of PDFs. Each chunking parameter change required a full re-run. Three iterations of tuning meant 30-45 minutes just waiting on indexing.

3. **CI/CD debugging**: The GitHub Actions pipeline — specifically the self-hosted runner setup on EC2, IAM permissions for ECR access, and the Docker port binding issue — collectively took longer than writing the application code.

The actual Python code (`app.py`, `helper.py`, `prompt.py`) took perhaps 2-3 hours to write. The infrastructure — Docker, GitHub Actions, Pinecone setup, AWS configuration — took significantly longer.

---

**Q33: What is your favorite part of this project and why?**

The RAG retrieval pipeline — specifically the moment when you ask a nuanced nutrition question and the server logs show exactly the right passage from the NSCA textbook being retrieved. It's the kind of emergent behavior that makes you feel like the system is actually "thinking."

The chain from raw PDFs → text chunks → 384-dimensional vectors → cosine similarity search → grounded LLM answer is conceptually elegant. Each step is individually simple, but the composition creates something genuinely useful. That's the best kind of engineering — simple parts that combine to solve a hard problem.

---

**Q34: What would you add if this became a real product?**

1. **User accounts and history**: Save past conversations so the coach can refer to what the user said before ("Last week you mentioned you were vegetarian...").
2. **Personalization**: Collect user profile data (height, weight, fitness goal, dietary restrictions) at sign-up and inject them into the system prompt so answers are personalized.
3. **Meal and workout plan generation**: Instead of Q&A only, generate structured weekly meal plans and exercise programs.
4. **Source citations in the UI**: Show which book and page each piece of advice comes from. Builds trust.
5. **Mobile app**: A React Native wrapper around the same backend API.
6. **Feedback loop**: Thumbs up/down per response, logged to a database. Over time, analyze which questions produce poor retrievals and improve chunking/prompts for those cases.
7. **Safety guardrails**: A medical disclaimer, dietary allergy warnings, and automatic escalation ("please consult a doctor") for health conditions mentioned in queries.

---

*End of Interview Preparation Guide*

---

> **Document generated**: 2026-02-27
> **Project**: Fitness & Diet Chatbot
> **Stack**: Flask + LangChain + Google Gemini + Pinecone + HuggingFace + Docker + AWS (ECR + EC2) + GitHub Actions
