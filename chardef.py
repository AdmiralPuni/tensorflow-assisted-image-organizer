import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow.keras as keras
import json
import numpy as np
from keras.preprocessing import image

selected_model = ""
character_names = []
model_filename = ""
settings_filename = "settings.json"
settings_json = json.load(open(settings_filename, 'r'))
input_directory = 'input'
output_directory = 'output'
model_directory = 'models'

prediction_list = []

print('TAIO v0.1 TERMINAL')
print('==============================================')

for index, model_name in enumerate(settings_json['models']):
  print(model_name)

print('==============================================')

selected_model = input('Select model : ')

character_names = settings_json['models'][selected_model]['names']
model_filename = settings_json['models'][selected_model]['filename']

print('==============================================')
print('Loading Keras...')

#create output folder based on names
for name in character_names:
  if not os.path.exists(output_directory + '/' + selected_model + '/' + name):
    os.makedirs(output_directory + '/' + selected_model + '/' + name)

#load keras model
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(len(character_names), activation='softmax')
])

#load saved model
model.load_weights(model_directory + "/" + model_filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def get_name(cropped_image):
  img = image.load_img(cropped_image, target_size=(150, 150))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)

  largest = 0

  for x in range(0, len(classes[0])):
    if(classes[0][x] > largest):
      largest = x
  
  return character_names[largest]

#for filename in os.listdir(input_directory):
  #path = os.path.join(input_directory, filename)
  #img = image.load_img(path, target_size=(150, 150))
  #print(get_name(img))