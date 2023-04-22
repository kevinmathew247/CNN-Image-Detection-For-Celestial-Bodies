import os
from io import BytesIO
from base64 import b64encode
import pandas as pd
from flask import render_template, request, redirect, url_for, send_from_directory
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed
from wtforms import StringField, validators
import matplotlib.pyplot as plt
import keras
import numpy as np
from keras.utils import load_img, img_to_array
from app.__init__ import app
from hub.examples.image_retraining.label_image import get_labels, wiki
from hub.examples.image_retraining.reverse_image_search import reverseImageSearch
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
        filename = photos.save (form.image_file.data)
        file_url = url_for('get_file', filename=filename)
        predict_answer()
        #return render_template ('index.html', form=form, file_url=file_url)


@app.route("/", methods=["GET", "POST"])
def index():
    # image in memory will be used on reload
    global imageBytes
    form = SelectImageForm()
    if form.validate_on_submit():
        upload()
        # if request.files.get(form.image_file.name):
        #     # from file
        #     imageBytes = request.files[form.image_file.name].read()
        #     im = Image.open(BytesIO(imageBytes))
        #     # cant save RGBA as JPEG
        #     im_resize=im.convert('RGB')
        #     # resize using PIL
        #     max_width = 1000
        #     if(im.size[0]>max_width):
        #         im_resize=im.resize((max_width,int(max_width/im.size[0]*im.size[1])))
        #     buf = BytesIO()
        #     filext = request.files[form.image_file.name].filename.split('.')[-1].upper()
        #     if(filext=='JPG'):
        #         filext='JPEG'
        #     im_resize.save(buf, format=filext)
        #     # get back bytes
        #     imageBytes = buf.getvalue()
        #     print("using form")

        # elif form.image_url.data:
        #     # from url
        #     response = requests.get(form.image_url.data)
        #     imageBytes = BytesIO(response.content).read()
        #     print("using url")
        # else:
        #     # empty form
        #     return render_template("index.html", form=form)
    # print(form.errors)

    return render_template("index.html", form=form)


@app.route("/about")
def about():
    return render_template("about.html")


# Route for displaying the prediction results
@app.route("/result")
def result():
    try:
        with open(os.path.join(APP_ROOT, 'uploads', 'prediction.txt'), 'r') as file:
            labels = file.read().splitlines()
        title, properties, description = wiki(labels[0], APP_ROOT)
        with open(os.path.join(APP_ROOT, 'uploads', 'image.png'), 'rb') as file:
            imageBytes = file.read()
            image = b64encode(imageBytes).decode("utf-8")
    except Exception as e:
        return render_template("error.html", detail=str(e))
    return render_template(
        "result.html",
        image=image,
        labels=labels,
        title=title,
        description=description,
        properties=properties,
    )

# Route for handling Google reverse image search
@app.route("/redirectToGoogle")
def redirectToGoogle():
    with open(os.path.join(APP_ROOT, 'uploads', 'image.png'), 'rb') as file:
        imageBytes = file.read()
    searchUrl = reverseImageSearch(imageBytes)
    return redirect(searchUrl, 302)



# @app.route("/redirectToGoogle")
# def redirectToGoogle():
#     searchUrl = reverseImageSearch(imageBytes)
#     return redirect(searchUrl, 302)

@app.route("/predict")
def preprocess_image(path):
    img = load_img(path, target_size = (256, 256))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a
def predict_answer():
    import matplotlib.image as mpimg
    # Read Test Images Dir and their labels
    test_upload_dir = '/Users/krishna/Desktop/CNN-Image-Detection-For-Celestial-Bodies-main/app/uploads'
    image_list = ['asteroids', 'earth','elliptical', 'jupiter', 'mars', 'mercury','moon', 'neptune', 'saturn', 'spiral', 'uranus', 'venus']

    # Get a list of all the files in the directory
    files = os.listdir(test_upload_dir)

    #Get the first file in the directory
    first_file = files[0]

    # Print the first file
    # print(first_file)
    first_file_path = os.path.join(test_upload_dir, first_file)
    # print(first_file_path)

    # preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        first_file_path, target_size=(256, 256)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # add a batch dimension
    
    # normalize the image
    img_array = img_array / 255.0
    
    # Load the trained model
    model = keras.models.load_model('/Users/krishna/Desktop/CNN-Image-Detection-For-Celestial-Bodies-main/hub/examples/image_retraining/CNN_Model.h5')

    # make predictions
    predictions = model.predict(img_array)
    
    # get the predicted class name
    class_idx = np.argmax(predictions[0])
    class_name = image_list[class_idx]
    
    # display the image and predicted class name
    plt.imshow(img)
    plt.title(class_name)
    plt.show()

test_images_dir = '/Users/krishna/Desktop/CNN-Image-Detection-For-Celestial-Bodies-main/hub/examples/image_retraining/lets_test/'
test_df = pd.read_csv('/Users/krishna/Desktop/CNN-Image-Detection-For-Celestial-Bodies-main/hub/examples/image_retraining/test.csv')

# put them in a list
test_dfToList = test_df['Image_id'].tolist()
test_ids = [str(item) for item in test_dfToList]

test_images = [test_images_dir+item for item in test_ids]
test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in test_images])
np.save('/Users/krishna/Desktop/CNN-Image-Detection-For-Celestial-Bodies-main/hub/examples/image_retraining/test_preproc_CNN.npy', test_preprocessed_images)

def preprocess_image(path):
    img = load_img(path, target_size = (256, 256))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a
app.run(debug=True)