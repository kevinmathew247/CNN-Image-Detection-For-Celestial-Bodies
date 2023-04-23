import os
from base64 import b64encode
import pandas as pd
from flask import render_template, request, redirect, url_for, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, validators
import matplotlib as plt
import keras
import numpy as np
from keras.utils import load_img, img_to_array
from app.__init__ import app
from hub.examples.image_retraining.label_image import get_labels, wiki
from hub.examples.image_retraining.reverse_image_search import reverseImageSearch
import keras.utils as image
from werkzeug.datastructures import FileStorage
import wikipedia
from yaml import load, SafeLoader
import tensorflow as tf

# no secret key set yet
SECRET_KEY = os.urandom(32)
app.config["SECRET_KEY"] = SECRET_KEY
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(APP_ROOT, 'uploads')
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class SelectImageForm(FlaskForm):
    image_url = StringField(
        "image_url",
        validators=[validators.Optional(), validators.URL()],
        render_kw={"placeholder": "Enter a URL"},
    )
    image_file = FileField(
        "file",
        validators=[
            validators.Optional(),
            FileAllowed(["jpg", "jpeg", "png"], "Invalid File"),
        ],
        render_kw={"class": "custom-file-input"},
    )

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route("/upload")
def upload():
        form = SelectImageForm()
        if form.validate_on_submit():
            image_file = FileStorage(
            stream=form.image_file.data.stream,
            filename=form.image_file.data.filename,
            content_type=form.image_file.data.content_type,
            content_length=form.image_file.data.content_length,
            headers=form.image_file.data.headers,
            )

            # save the image to the UPLOADS_DEFAULT_DEST folder
            filename = photos.save(image_file)
            file_url = url_for('get_file', filename=filename)
            predict_answer()

@app.route("/", methods=["GET", "POST"])
def index():
    # image in memory will be used on reload
    global imageBytes
    form = SelectImageForm()
    if form.validate_on_submit():
        if request.files.get(form.image_file.name):
         upload()
        return redirect(url_for("result"))    
    print("returning to index")
    return render_template("index.html", form=form)


@app.route("/about")
def about():
    return render_template("about.html")



@app.route("/result")
def result():
    cwd = os.path.join(app.root_path, "..", "hub", "examples", "image_retraining")

    celestial_object, labels = predict_answer()
    title, properties, description = wiki(celestial_object, cwd)
    print(celestial_object)
    print(labels)
    print(title)
    print(properties)
    return render_template(
        "result.html",
        labels=labels,
        title=title,
        description=description,
        properties=properties,
    )


@app.route("/redirectToGoogle")
def redirectToGoogle():
    searchUrl = reverseImageSearch(imageBytes)
    return redirect(searchUrl, 302)

@app.route("/predict")
def predict_answer():
    import matplotlib.image as mpimg
    # Read Test Images Dir and their labels
    test_upload_dir = '/Users/kevinmathew/Documents/UoL CNN/app/uploads'
    image_list = ['asteroids', 'earth','elliptical', 'jupiter', 'mars', 'mercury','moon', 'neptune', 'saturn', 'spiral', 'uranus', 'venus']

    # Get a list of all the files in the directory
    files = os.listdir(test_upload_dir)

    # Get the first file in the directory
    first_file = files[0]
    print("first_file")
    first_file_path = os.path.join(test_upload_dir, first_file)

    img_width=256; img_height=256
 
    img = image.load_img(first_file_path, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0   # normalize pixel values to [0, 1]
    print("Pre processign done")
    model = keras.models.load_model('/Users/kevinmathew/Documents/UoL CNN/hub/examples/image_retraining/CNN_Model.h5')
    # Make the prediction
    predictions = model.predict(x)

    # Print the predicted class label and probability
    class_names = ['asteroids', 'earth', 'jupiter', 'mars', 'moon', 'neptune', 'saturn', 'spiral', 'uranus', 'venus']
    predicted_class_index = np.argmax(predictions)
    print(predictions)
    predicted_class_label = class_names[predicted_class_index]
    predicted_probability = predictions[0][predicted_class_index]
    print(f"The predicted class is {predicted_class_label} with probability {predicted_probability:.2f}")

    cwd = os.path.join(app.root_path, "..", "hub", "examples", "image_retraining")
    title, properties, description = wiki(predicted_class_label, cwd)
    top_k = predictions[0].argsort()[-len(predictions[0]) :][::-1]

    label_lines = [
        line.rstrip() for line in tf.io.gfile.GFile(cwd + "/retrained_labels.txt")
    ]
    labels_and_scores = [
            (label_lines[node_id], predictions[0][node_id]) for node_id in top_k
        ]
    print("results page")
    return predicted_class_label, labels_and_scores


# return title, statistics and summary
def wiki(celestial_object,cwd):
    ans = celestial_object
    with open(os.path.join(cwd, "display_info.yml"), "r") as stream:
        all_display_statistics = load(stream, Loader=SafeLoader)

    req_statistics = all_display_statistics.get(ans, {})
    statistics = None
    title = None
    summary = None
    if ans in ["spiral", "elliptical"]:
        title = ("Classified Celestial Object is {} Galaxy : ".format(ans.capitalize()))
        summary = (wikipedia.WikipediaPage(title="{} galaxy".format(ans)).summary)
    elif ans in [
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ]:
        title = ("Classified Celestial Object is {} Planet : ".format(ans.capitalize()))
        statistics = req_statistics.items()
        summary = (wikipedia.WikipediaPage(title="{} (planet)".format(ans)).summary)
    elif ans == "moon":
        statistics = req_statistics.items()
        summary = (wikipedia.WikipediaPage(title="{}".format(ans)).summary)
        title = ("Classified Celestial Object is the {} : ".format(ans.capitalize()))
    elif ans == "asteroids":
        statistics = req_statistics.items()
        summary = (wikipedia.WikipediaPage(title="{}".format(ans)).summary)
        title = ("Classified Celestial Object is the {} : ".format(ans.capitalize()))
    return title,statistics,summary