from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

from process import classify
from AppOpener import open, close

app = Flask(__name__)
CORS(app)


@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify_text():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    elif request.method == "POST":
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').replace('\n','')
            print(text)
            if classify(text, app.instance_path) == 1:
                result = 'depression'
            else:
                result = 'no depression'
            response = {
                'result': result
            }
            print(result)
            return jsonify(response), 200
        else:
            print('error')
            return jsonify({"error": "Request must be JSON"}), 400


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response


if __name__ == '__main__':
    open("liwc-22", match_closest=True)
    app.run(debug=True)
    close("liwc-22", match_closest=True)
