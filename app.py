from flask import Flask, request, jsonify
from flask_cors import CORS
from stockfish_pool import StockfishPool
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes


# Create a global StockfishPool instance
stockfish_pool = StockfishPool(num_processes=max(1, os.cpu_count() - 1))

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/api/stockfish', methods=['POST'])
def post_example():
    if request.method == 'POST':
        data = request.get_json()
        pgn = data.get('pgn')
        if pgn:
            response = stockfish_pool.analyze_pgn(pgn)
            print(response)
            return jsonify(response), 201
        else:
            return jsonify({'error': 'No data received'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)