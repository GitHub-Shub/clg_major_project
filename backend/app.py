from flask import Flask, request, jsonify, send_from_directory
import json
import os
import subprocess
import time
import sys
import logging

app = Flask(__name__, static_folder="../static", static_url_path="/static")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def start_ollama_server():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            app.logger.info("Ollama server is already running.")
            return None
        if sys.platform.startswith('win'):
            process = subprocess.Popen(['start', '/B', 'ollama', 'serve'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            process = subprocess.Popen(['nohup', 'ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(5)
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            app.logger.info("Ollama server started successfully.")
            return process
        else:
            app.logger.error("Failed to start Ollama server.")
            return None
    except Exception as e:
        app.logger.error(f"Error starting Ollama server: {str(e)}")
        return None

ollama_process = start_ollama_server()
try:
    from rag_pipeline import RAGPipeline
    rag = RAGPipeline()
except Exception as e:
    app.logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
    rag = None

FAQ_CACHE = 'faq_cache.json'
if os.path.exists(FAQ_CACHE):
    with open(FAQ_CACHE, 'r', encoding='utf-8') as f:
        faq_cache = json.load(f)
else:
    faq_cache = {
        "What is the penalty for driving without a license?": "Hey! Driving without a valid license? That's a no-go under Section 181. You'll face a fine of ₹5000. Keep that license handy!",
        "What is the golden hour in the MV Act?": "Alright, the 'golden hour' in the MV Act, under Section 2(12A), is that critical one-hour window after a serious accident where quick medical help can save lives. Think of it as the race-against-time moment!",
        "What is the punishment for overspeeding?": "Speeding ticket blues? Section 183 says light vehicles get a ₹1000-₹2000 fine, while medium/heavy ones face ₹2000-₹4000. Repeat offenders might lose their license too. Slow down, champ!",
        "What is the fine for not wearing a helmet?": "No helmet, no bueno! Section 194D slaps a ₹1000 fine for riding a two-wheeler without a helmet, and you could lose your license for three months. Safety first, right?",
        "Can a minor obtain a driving license under the MV Act?": "Kids behind the wheel? Nope! Section 4 says you gotta be 18 for a driving license, though 16-year-olds can snag a learner’s for certain vehicles. Gotta wait a bit!",
        "What happens if I drive a vehicle without a valid registration?": "Driving unregistered? Ouch! Section 192 hits you with a fine up to ₹5000 for the first offense, and up to ₹10,000 or even 7 years in jail for repeats. Get that registration sorted!"
    }
    with open(FAQ_CACHE, 'w', encoding='utf-8') as f:
        json.dump(faq_cache, f, indent=2)

@app.route('/')
def serve_index():
    app.logger.info("Serving index.html")
    return send_from_directory('../static', 'index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        query = data.get('query', '').strip()
        app.logger.info(f"Received query: {query}")
        if not query:
            app.logger.warning("Empty query received")
            return jsonify({'response': 'Hey, give me something to work with! Ask a question about the Motor Vehicles Act.'}), 400
        if query in faq_cache:
            app.logger.info(f"Returning cached response for query: {query}")
            return jsonify({'response': faq_cache[query]})
        if rag is None:
            app.logger.error("RAG pipeline is not initialized")
            return jsonify({
                'response': "Oops, my advanced answering system is down. Try a simple question like 'What is the penalty for driving without a license?' or check the server logs!"
            }), 503
        response = rag.process_query(query)
        app.logger.info(f"Returning RAG response for query: {query}")
        return jsonify({'response': response})
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'response': f"Sorry, something went wrong: {str(e)}. Try a question like 'What is the penalty for driving without a license?' or check the server logs."
        }), 503

@app.route('/test', methods=['GET'])
def test_endpoint():
    app.logger.info("Test endpoint accessed")
    return jsonify({'status': 'Backend is running', 'rag_status': 'Initialized' if rag else 'Failed'}), 200

@app.teardown_appcontext
def cleanup(exception=None):
    if ollama_process and sys.platform.startswith('win'):
        app.logger.info("Terminating Ollama process")
        subprocess.run(['taskkill', '/PID', str(ollama_process.pid), '/F'], shell=True)

if __name__ == '__main__':
    app.logger.info("Starting Flask server on port 5000")
    app.run(debug=True, host='0.0.0.0', port=5000)