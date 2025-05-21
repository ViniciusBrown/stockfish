# Stockfish Analysis API

A Flask-based REST API that provides chess game analysis using Stockfish engine. The API accepts chess games in PGN format and returns detailed analysis using a pool of Stockfish instances for efficient processing.

## Requirements

- Python 3.x
- Stockfish chess engine installed on the system
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
3. Ensure Stockfish is installed and accessible in your system PATH

## API Endpoints

### POST /api/stockfish

Analyzes a chess game in PGN format.

#### Request
- Method: POST
- Content-Type: application/json
- Body:
```json
{
    "pgn": "Your PGN string here"
}
```

#### Response
Returns a JSON object containing the analysis results with HTTP status 201 on success.

#### Example Request
```bash
curl -X POST http://localhost:5000/api/stockfish \
  -H "Content-Type: application/json" \
  -d '{"pgn": "1. e4 e5 2. Nf3 Nc6 3. Bb5"}'
```

## Development

Run the development server:
```bash
python app.py
```
The server will start on http://0.0.0.0:5000

## Production Deployment

For production deployment, use Gunicorn:
```bash
gunicorn app:app -b 0.0.0.0:5000
```

## Error Handling

- 400 Bad Request: Returned when no PGN data is provided
- 201 Created: Successful analysis
