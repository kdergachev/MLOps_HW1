curl -X POST http://127.0.0.1:5000/models/1 -d '{"action":"predict","X":[[0],[1],[2]]}' -H 'Content-Type: application/json'