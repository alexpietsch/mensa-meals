from flask import Flask, request, jsonify
from read import find_similar, find_similar_with_threshold

app = Flask(__name__)

@app.route('/similar', methods=['GET'])
def find_similar_api():
    query = request.args.get('query')
    k = int(request.args.get('k', 5))
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    if k > 200:
        return jsonify({"error": "k must be less than 200"}), 400
    results = find_similar(query, k)
    return jsonify({"results": results})

@app.route('/similar-threshold', methods=['GET'])
def find_similar_threshold_api():
    query = request.args.get('query')
    t = int(request.args.get('t', 9)) / 100
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    if t > 100:
        return jsonify({"error": "t must be 0 < t <= 100"}), 400
    results = find_similar_with_threshold(query, t)
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
