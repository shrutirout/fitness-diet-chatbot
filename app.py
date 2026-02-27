from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "diet-fitness-chatbot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)




retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

chatModel = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
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



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
