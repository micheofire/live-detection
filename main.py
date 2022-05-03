from flask import Flask, request, jsonify
from flask_restful import Api, Resource
import pickle
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import argparse
from tensorflow import keras

#START THE FLASK APP
app = Flask(__name__)
api = Api(app)

    
def format_image(img_path):
  img = Image.open(img_path)
  img_array = np.array(img)
  image = tf.image.resize(img_array, (224, 224))/255.0
  image = tf.expand_dims(image, 0)
  return image

def make_inference(model, image):
  prediction = model(image,training=False).numpy()
  predicted_batch = tf.squeeze(prediction).numpy()
  predicted_result = np.argmax(predicted_batch, axis=-1)
  return predicted_result


#CLASS FOR PREPROCESSING TEXT AND MAKING PREDICTION
class Predict(Resource):
    def post(self):
        img_path = request.json['image_path']
        model_path = request.json['model_path']
        print(model_path)
        model = keras.models.load_model(model_path)
        image = format_image(img_path)
        result = make_inference(model,image)
        print(result)
        if result == 0:
            return jsonify({'response':'This image is a Live Image'})
        else:
            return jsonify({'response':'This image is pic of pic'})


#ENDPOINT FOR MAKING API PREDICTION
api.add_resource(Predict, '/predict')


if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0')

