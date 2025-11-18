# RAG Introduction Project

This project demonstrates a basic implementation of Retrieval-Augmented Generation (RAG) using Cerebras as the Large Language Model (LLM).

## What is RAG?

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of Large Language Models (LLMs) by combining them with a retrieval system. Instead of relying solely on the model's pre-trained knowledge, RAG retrieves relevant information from a knowledge base and uses it to generate more accurate and contextually relevant responses.

### Key Components of RAG:

1. **Knowledge Base**: A collection of documents or data that serves as the source of information.
2. **Embedding Model**: Converts text into vector representations (embeddings).
3. **Vector Store**: A database that stores and allows efficient search of embeddings.
4. **Retriever**: Searches the vector store for relevant information based on a query.
5. **Generator**: The LLM that uses the retrieved information to generate responses.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Cerebras Inference API key:
  1. Sign up at [Cerebras](https://cerebras.ai/)
  2. Navigate to the Inference section
  3. Generate an API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/ronxldwilson/rag-intro.git
   cd rag-intro
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   CEREBRAS_API_KEY=your_cerebras_api_key_here
   ```

### Usage

1. Prepare your knowledge base by adding documents to the `data/` directory.

2. Run the RAG system:
   ```
   python rag_demo.py
   ```

3. Interact with the system by entering queries. The system will retrieve relevant information and generate responses using Cerebras LLM.

## Implementation Details

- **LLM**: Cerebras Inference API (OpenAI-compatible) - Model: llama3.1-8b (check Cerebras docs for latest models)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: ChromaDB
- **Framework**: LangChain

## Next Steps

- Experiment with different embedding models
- Try larger knowledge bases
- Implement advanced retrieval techniques (e.g., re-ranking)
- Add evaluation metrics for response quality
