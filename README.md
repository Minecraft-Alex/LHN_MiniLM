# Laws of Human Nature Q&A System

A semantic search and question-answering system built on knowledge about human nature laws and psychological principles. This project uses sentence transformers and FAISS to create an intelligent retrieval system that can answer questions about various laws of human nature, personality traits, and psychological transformations.

## 📋 Overview

This project processes a knowledge base of The Laws of Human Nature book by Robert Greene and creates an instruction-style dataset for fine-tuning. It then implements a semantic search system using sentence embeddings and FAISS indexing to retrieve relevant answers to user queries about human psychology, behavior patterns, and personal development.

## ✨ Features

- **Instruction Dataset Generation**: Converts structured knowledge base into instruction-following format
- **Semantic Search**: Uses sentence transformers (`all-MiniLM-L6-v2`) for embedding-based similarity search
- **FAISS Indexing**: Fast and efficient vector similarity search using Facebook AI Similarity Search
- **Question-Answering**: Query the knowledge base with natural language questions
- **Multiple Query Types**: Supports various question formats including:
  - General law explanations
  - Emotionally personalized questions
  - Personality-driven queries
  - Tag-based advice requests

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Colab (for running the notebook) or local Jupyter environment

### Required Packages

```bash
pip install transformers datasets peft accelerate bitsandbytes
pip install sentence-transformers
pip install faiss-cpu
```

## 📁 Project Structure

```
psych/
├── Finetune.ipynb                 # Main Jupyter notebook with the complete pipeline
├── knowledgebase.json            # Source knowledge base with human nature laws
├── human_nature_instruct.jsonl   # Generated instruction-style training dataset
└── README.md                      # This file
```

## 🚀 Usage

### 1. Data Preparation

The notebook processes `knowledgebase.json` which should contain entries with the following structure:

```json
{
  "title": "Law Name",
  "summary": "Description of the law",
  "key_principles": ["Principle 1", "Principle 2"],
  "tags": ["tag1", "tag2"],
  "personality_links": ["Personality Type"],
  "positive_transformation": "Transformation advice",
  "section": "Chapter/Section name"
}
```

### 2. Generate Instruction Dataset

Run the notebook cells to:
- Load the knowledge base
- Generate instruction-style examples (205 examples from the knowledge base)
- Save as JSONL format for fine-tuning

### 3. Build Semantic Search System

The notebook creates:
- Sentence embeddings using `all-MiniLM-L6-v2` model
- FAISS index for fast similarity search
- Query-answer mapping system

### 4. Query the System

```python
def ask_question(query, top_k=1):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    responses = [id_to_output[i] for i in I[0]]
    return responses

# Example queries
query1 = "Explain the concept of 'The Shadow' and how we should interact with it?"
responses = ask_question(query1)
print(responses[0])
```

## 📊 Dataset Generation

The system generates multiple types of instruction examples from each knowledge base entry:

1. **General Law Explanation**: "What is the law of X?"
2. **Emotionally Personalized**: "I struggle with Y. How can I improve?"
3. **Personality-Driven**: "What should someone with Z personality understand?"
4. **Tag-Based Advice**: "What law applies when dealing with W?"

## 🔍 Example Queries

- "Explain the concept of 'The Shadow' and how we should interact with it?"
- "How does the story of Pericles illustrate the Law of Irrationality?"
- "What is the law of Attitude?"
- "I struggle with anxiety and avoidance. How can I improve?"

## 🧠 Technical Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **Search Method**: L2 distance in FAISS IndexFlatL2
- **Dataset Format**: JSONL with `instruction`, `input`, and `output` fields
- **Total Training Examples**: 205 instruction pairs

## 📝 Notes

- The notebook was originally designed for Google Colab (includes Google Drive mounting)
- For local use, modify file paths accordingly
- The semantic search system can be extended to support fine-tuning of language models
- Widget metadata has been removed for GitHub compatibility

## 🔮 Future Enhancements

- Fine-tune a language model on the instruction dataset
- Add support for multi-turn conversations
- Implement confidence scoring for retrieved answers
- Add support for multiple top-k results with ranking
- Create a web interface for easier querying

## 📄 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project processes knowledge about human nature laws and psychological principles. The semantic search system helps retrieve relevant information based on user queries.
