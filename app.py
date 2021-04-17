from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

import tensorflow as tf
import numpy as np
from tensorflow import keras

UPLOAD_FOLDER = "static/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class_name = ['n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02087046-toy_terrier']

img_width = 180
img_beight = 180

model = tf.keras.models.load_model('dog-breed.h5')


def get_prediction(file_name):
    img = keras.preprocessing.image.load_img(
        file_name,
        target_size=(img_width, img_beight)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    prediction = model.predict(img_array)
    score = tf.nn.softmax(prediction[0])

    print(f"predictions : {prediction}")
    print(f"Score: {score}")

    prediction_name = class_name[np.argmax(score)]
    per = np.max(score) * 100

    return prediction_name, per


@app.route("/", methods=['GET', 'POST'])
def uploadANDpredict():
    if request.method == "POST":
        file_obj = request.files['file']

        if file_obj:
            f_name = secure_filename(file_obj.filename)
            f_name = os.path.join(UPLOAD_FOLDER, f_name)
            file_obj.save(f_name)

            name, per = get_prediction(f_name)

            return render_template('index.html', img=f_name, img_class_name=name, accuracy=per)
            # return "Upload success!"

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
