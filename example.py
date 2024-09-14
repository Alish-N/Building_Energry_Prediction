from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample data
students = ['Osama', 'Alishba']

# Routes for different functionalities
@app.route('/profile', methods=['POST'])
def get_profile():
    data = request.get_json(force=True)
    if data:
        return jsonify(data)

@app.route('/')
def home():
    return "Welcome to Student Portal"

if __name__ == '__main__':
    app.run(debug=True)