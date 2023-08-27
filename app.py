from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import requests
from PIL import Image

app = Flask(__name__)
# define labels
labels = ['Matang', 'Mentah', 'Setengah Matang']


def preprocess(img_path, input_size):
    nimg = img_path.convert('RGB').resize(input_size, resample=0)
    img_arr = (np.array(nimg))/255
    return img_arr


def reshape(imgs_arr):
    return np.stack(imgs_arr, axis=0)


@app.route("/", methods=['GET', 'POST'])
def get_output():
    if request.method == 'GET':
        return render_template("index.html")

    elif request.method == 'POST':
        model = load_model('model1.h5', compile=False)
        img = request.files['photo']
        img_path = 'static/img/predict_img/'+img.filename
        img.save(img_path)
        im = Image.open(img_path)

        # read image
        input_size = (224, 224)
        X = preprocess(im, input_size)
        X = reshape([X])
        y = model.predict(X)

        hasil = labels[np.argmax(y)]
        gambar = 'static/img/predict_img/' + img.filename
        return render_template("predict.html", result=labels[np.argmax(y)], gambar=gambar, hasil=hasil)


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
