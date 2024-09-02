from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from vector_db_populate import clear_vector_db,populate_db

from rag_service_query import query_rag,raw_query,query_rag_with_prompt_engineering
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
DATA_PATH = 'knowledge-base'
VECTOR_DB_PATH = 'job-library'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(VECTOR_DB_PATH):
    os.makedirs(VECTOR_DB_PATH)

app.config['DATA_PATH'] = DATA_PATH
app.config['VECTOR_DB_PATH'] = VECTOR_DB_PATH


@app.route('/')
def hello_world():
    return 'Hello World!'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, timeout=10000)




@app.route('/clear-db', methods=['GET'])
def clear_db():
    try:
        clear_vector_db()
        return jsonify({'message': 'DB cleared'}), 200
    except:
        return jsonify({'message': 'Error clearing DB'}), 500


@app.route('/query-rag', methods=['POST'])
def handle_query_rag():
    request_data = request.get_json()
    query_text = request_data.get('query_text', '')
    if query_text:
        response_text = query_rag(query_text)
        return jsonify({'response': response_text}), 200
    else:
        return jsonify({'error': 'No query text provided'}), 400


@app.route('/populate-db',methods=['POST'])
def populate_vector_dv():
    try:
        message = populate_db()
        add_db_message = message[1]

        return jsonify({'success': 'File successfully uploaded', 'db_message': add_db_message}), 200
    except Exception as e:
        app.logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500



@app.route('/clear-db-job', methods=['GET'])
def clear_db_job():
    try:
        clear_vector_db()
        return jsonify({'message': 'DB cleared'}), 200
    except:
        return jsonify({'message': 'Error clearing DB'}), 500

@app.route('/raw-query', methods=['POST'])
def handle_raw_query():
    try:
        request_data = request.get_json()
        query_text = request_data.get('query_text', '')
        if query_text:
            response_text = raw_query(query_text)
            return jsonify({'response': response_text}), 200
        else:
            return jsonify({'error': 'No query text provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/prompt-eng-rag', methods=['POST'])
def handle_query_rag_with_prompt_eng():
    request_data = request.get_json()
    query_text = request_data.get('query_text', '')
    if query_text:
        response_text = query_rag_with_prompt_engineering(query_text)
        return jsonify({'response': response_text}), 200
    else:
        return jsonify({'error': 'No query text provided'}), 400