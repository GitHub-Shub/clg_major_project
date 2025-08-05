from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama
import os
import requests
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class RAGPipeline:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            self.index = faiss.IndexFlatL2(384)
            self.chunks = []
            self.load_and_index_document()
            logging.info("RAGPipeline initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAGPipeline: {str(e)}")
            raise

    def load_and_index_document(self):
        try:
            if not os.path.exists('mv_act_cleaned.txt'):
                logging.error("mv_act_cleaned.txt not found")
                raise FileNotFoundError("mv_act_cleaned.txt not found. Run data_cleaner.py first.")
            with open('mv_act_cleaned.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            if not text.strip():
                logging.error("mv_act_cleaned.txt is empty")
                raise ValueError("mv_act_cleaned.txt is empty. Ensure extract_pdf.py and data_cleaner.py ran correctly.")
            self.chunks = self.text_splitter.split_text(text)
            if not self.chunks:
                logging.error("No chunks generated from mv_act_cleaned.txt")
                raise ValueError("Failed to generate chunks from mv_act_cleaned.txt.")
            embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)
            self.index.add(np.array(embeddings, dtype='float32'))
            logging.info(f"Document indexed successfully with {len(self.chunks)} chunks")
        except Exception as e:
            logging.error(f"Failed to index document: {str(e)}")
            raise

    def retrieve_chunks(self, query, k=3):
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            distances, indices = self.index.search(np.array([query_embedding], dtype='float32'), k)
            return [self.chunks[i] for i in indices[0]]
        except Exception as e:
            logging.error(f"Failed to retrieve chunks: {str(e)}")
            raise

    def process_query(self, query):
        try:
            response = requests.get('http://localhost:11434', timeout=5)
            if response.status_code != 200:
                logging.error("Ollama server not responding")
                raise Exception("Ollama server is not responding.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama server not running: {str(e)}")
            raise Exception("Ollama server is not running.")
        retrieved_chunks = self.retrieve_chunks(query)
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query}")
        prompt = (
            f"You are Grok, a friendly and helpful AI created by xAI. Answer the following question about the Motor Vehicles Act in a clear, concise, and conversational style, like a knowledgeable friend explaining things simply. Base your answer strictly on the provided information from the Motor Vehicles Act, and cite relevant sections if possible. Avoid formal or legalistic language—keep it engaging and easy to understand!\n\n"
            f"Question: {query}\n\n"
            f"Relevant Information:\n" + "\n".join(retrieved_chunks) + "\n\n"
            f"Provide a concise answer in Grok's style, sticking to the facts from the Motor Vehicles Act."
        )
        try:
            response = ollama.generate(
                model='tinyllama',
                prompt=prompt,
                options={'temperature': 0.7, 'timeout': 30}
            )
            logging.info(f"Generated response for query: {query}")
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Failed to generate response with tinyllama: {str(e)}")
            raise Exception(f"Failed to generate response with tinyllama: {str(e)}")

# backend/faq_cache.json
{
    "What is the penalty for driving without a license?": "Hey! Driving without a valid license? That's a no-go under Section 181. You'll face a fine of ₹5000. Keep that license handy!",
    "What is the golden hour in the MV Act?": "Alright, the 'golden hour' in the MV Act, under Section 2(12A), is that critical one-hour window after a serious accident where quick medical help can save lives. Think of it as the race-against-time moment!",
    "What is the punishment for overspeeding?": "Speeding ticket blues? Section 183 says light vehicles get a ₹1000-₹2000 fine, while medium/heavy ones face ₹2000-₹4000. Repeat offenders might lose their license too. Slow down, champ!",
    "What is the fine for not wearing a helmet?": "No helmet, no bueno! Section 194D slaps a ₹1000 fine for riding a two-wheeler without a helmet, and you could lose your license for three months. Safety first, right?",
    "Can a minor obtain a driving license under the MV Act?": "Kids behind the wheel? Nope! Section 4 says you gotta be 18 for a driving license, though 16-year-olds can snag a learner’s for certain vehicles. Gotta wait a bit!",
    "What happens if I drive a vehicle without a valid registration?": "Driving unregistered? Ouch! Section 192 hits you with a fine up to ₹5000 for the first offense, and up to ₹10,000 or even 7 years in jail for repeats. Get that registration sorted!"
}
