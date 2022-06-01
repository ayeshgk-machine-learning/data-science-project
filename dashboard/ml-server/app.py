from flask import Flask ,request, jsonify, make_response

app = Flask(__name__)


@app.route('/')
def hello():
    return make_response(jsonify({
                'message': 'Hello World'
            }), 200)


if __name__ == "__main__":
    app.run(debug=True)