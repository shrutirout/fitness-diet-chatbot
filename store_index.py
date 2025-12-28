from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Loading PDF files...")
extracted_data=load_pdf_file(data='data/')
print(f"Loaded {len(extracted_data)} pages from PDFs")

filter_data = filter_to_minimal_docs(extracted_data)
print("Filtering documents...")

text_chunks=text_split(filter_data)
print(f"Split into {len(text_chunks)} text chunks")

print("Downloading embeddings model...")
embeddings = download_hugging_face_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "diet-fitness-chatbot"

print("Connecting to Pinecone...")
if not pc.has_index(index_name):
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists")

index = pc.Index(index_name)

print("Uploading vectors to Pinecone (this may take a few minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Done! Vector database is ready.")