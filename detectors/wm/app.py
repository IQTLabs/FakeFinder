import pickle
from flask import Flask, request, jsonify
from ensemble import Ensemble

app = Flask(__name__)

submit = Ensemble()


@app.route('/predict', methods=['POST'])
def predict():
    video_pth = str(request.get_json(force=True)['video_path'])
    result = submit.inference(video_pth)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
