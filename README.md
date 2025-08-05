# Motor Vehicles Act RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that provides friendly, conversational answers to questions about the Indian Motor Vehicles Act. The bot uses advanced AI techniques to understand your questions and provide accurate, easy-to-understand responses based on the official Motor Vehicles Act document.

## 🚀 Features

- **Intelligent Q&A**: Ask natural language questions about the Motor Vehicles Act
- **RAG-Powered**: Combines document retrieval with AI generation for accurate, contextual answers
- **Conversational Style**: Responses in a friendly, engaging tone (inspired by Grok)
- **Fast FAQ Cache**: Common questions get instant cached responses
- **Web Interface**: Clean, user-friendly web interface
- **Real-time Processing**: Processes complex queries using semantic search

## 🏗️ Architecture

The application uses a modern RAG architecture:

- **Frontend**: Static web interface served by Flask
- **Backend**: Flask API server with RAG pipeline
- **Vector Database**: FAISS for fast similarity search
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **LLM**: Ollama with Gemma2:2b model
- **Document Processing**: PDF extraction and text cleaning pipeline

## 🛠️ Tech Stack

- **Python 3.8+**
- **Flask** - Web framework
- **LangChain** - Text processing and chunking
- **SentenceTransformers** - Text embeddings
- **FAISS** - Vector similarity search
- **Ollama** - Local LLM inference
- **pdfplumber** - PDF text extraction

## 📦 Installation

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** installed and running
   ```bash
   # Install Ollama (visit https://ollama.ai for platform-specific instructions)
   # Then pull the required model:
   ollama pull gemma2:2b
   ```

### Setup Steps

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd motor-vehicles-act-rag
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Motor Vehicles Act document**

   ```bash
   # Place your "MV Act English.pdf" file in the backend/ directory
   cd backend
   python extract_pdf.py    # Extract text from PDF
   python data_cleaner.py   # Clean and process the text
   ```

4. **Start the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## 📋 Requirements

Create a `requirements.txt` file with:

```
Flask==2.3.3
sentence-transformers==2.2.2
langchain==0.0.310
faiss-cpu==1.7.4
numpy==1.24.3
ollama==0.1.8
pdfplumber==0.9.0
requests==2.31.0
```

## 📁 Project Structure

```
motor-vehicles-act-rag/
├── backend/
│   ├── app.py                 # Main Flask application
│   ├── rag_pipeline.py        # RAG processing pipeline
│   ├── extract_pdf.py         # PDF text extraction
│   ├── data_cleaner.py        # Text cleaning utilities
│   ├── faq_cache.json         # Cached FAQ responses
│   ├── mv_act_cleaned.txt     # Processed document (generated)
│   ├── raw_mv_act.txt         # Raw extracted text (generated)
│   └── MV Act English.pdf     # Source document (you provide)
├── static/
│   ├── index.html             # Web interface
│   ├── style.css              # Styling
│   └── script.js              # Frontend JavaScript
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚦 Usage

### Web Interface

1. Open `http://localhost:5000` in your browser
2. Type your question about the Motor Vehicles Act
3. Get instant, conversational responses

### Example Queries

- "What is the penalty for driving without a license?"
- "What happens if I don't wear a helmet?"
- "Can a minor get a driving license?"
- "What is the golden hour in the MV Act?"
- "What are the fines for overspeeding?"

### API Endpoints

- `GET /` - Serve the web interface
- `POST /query` - Process questions (JSON: `{"query": "your question"}`)
- `GET /test` - Health check endpoint

## ⚙️ Configuration

### Environment Variables

The application automatically handles Ollama server startup, but you can configure:

- **Port**: Default is 5000 (modify in `app.py`)
- **Model**: Default is `gemma2:2b` (modify in `rag_pipeline.py`)
- **Chunk size**: Default is 500 characters (modify in `rag_pipeline.py`)

### FAQ Cache

Common questions are cached in `faq_cache.json` for instant responses. You can:

- Add new cached responses
- Modify existing responses
- Clear cache by deleting the file (will regenerate with defaults)

## 🔧 Troubleshooting

### Common Issues

1. **"Ollama server not responding"**

   ```bash
   # Ensure Ollama is installed and the model is available
   ollama list
   ollama pull gemma2:2b
   ```

2. **"mv_act_cleaned.txt not found"**

   ```bash
   # Run the data processing pipeline
   python extract_pdf.py
   python data_cleaner.py
   ```

3. **"RAG pipeline failed to initialize"**

   - Check if all dependencies are installed
   - Ensure the cleaned text file exists and is not empty
   - Check system memory (FAISS indexing requires adequate RAM)

4. **Slow responses**
   - First-time model loading takes time
   - Consider using a smaller model or GPU acceleration
   - Check system resources

### Logs

The application provides detailed logging. Check console output for:

- Server startup status
- Query processing steps
- Error details and stack traces

## 🎯 Development

### Adding New Features

1. **Custom prompts**: Modify the prompt template in `rag_pipeline.py`
2. **Different models**: Change the model in `ollama.generate()` call
3. **Enhanced retrieval**: Adjust chunk size, overlap, or retrieval count
4. **New endpoints**: Add routes to `app.py`

### Testing

```bash
# Test the backend health
curl http://localhost:5000/test

# Test a query
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the penalty for driving without a license?"}'
```

## 📝 License

This project is open source. Please ensure you have rights to use the Motor Vehicles Act document for your specific use case.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:

1. Check the troubleshooting section
2. Review application logs
3. Ensure all dependencies are properly installed
4. Verify Ollama is running with the correct model

---

Built with ❤️ using modern RAG techniques to make legal information more accessible.
