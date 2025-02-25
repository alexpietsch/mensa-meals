from flask import Flask, request, jsonify
from read import find_similar

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

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
