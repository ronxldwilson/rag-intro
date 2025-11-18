import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

if not CEREBRAS_API_KEY:
    raise ValueError("Please set CEREBRAS_API_KEY in your .env file")

# Cerebras configuration
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
MODEL_NAME = "llama3.1-8b"  # You can change this to available models

def setup_rag_system():
    # Load documents
    loader = DirectoryLoader('data/', glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    vectorstore.persist()

    # Set up LLM
    llm = OpenAI(
        openai_api_key=CEREBRAS_API_KEY,
        openai_api_base=CEREBRAS_BASE_URL,
        model_name=MODEL_NAME,
        temperature=0.7
    )

    # Create retrieval QA chain
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

def main():
    print("Setting up RAG system with Cerebras LLM...")
    qa_chain = setup_rag_system()
    print("RAG system ready! Type 'quit' to exit.")

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            break

        try:
            result = qa_chain.run(query)
            print(f"\nAnswer: {result}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
