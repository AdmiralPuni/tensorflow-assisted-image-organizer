print('loading...')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow.keras as keras
import json
from shutil import copyfile
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt


selected_model = ""
character_names = []
model_filename = ""
settings_filename = "settings.json"
settings_json = json.load(open(settings_filename, 'r'))
input_directory = 'input'

print('TAIO v0.1 TERMINAL')
print('==============================================')

for index, model_name in enumerate(settings_json['models']):
  print(model_name)

selected_model = input('Select model : ')
character_names = settings_json['models'][selected_model]['names']
model_filename = settings_json['models'][selected_model]['filename']

#create output folder based on names
if not os.path.exists('output/' + selected_model + '/false'):
    os.makedirs('output/' + selected_model + '/false')

for name in character_names:
  if not os.path.exists('output/' + selected_model + '/' + name):
    os.makedirs('output/' + selected_model + '/' + name)

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
model.load_weights("models/" + model_filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

def user_decision(detected_class, filename, accuracy):
  print("Detected  :", character_names[detected_class])
  print("Accuracy  :", round(accuracy)*100)
  if input("y/n       : ") == "y":
    copyfile(path, "output/" + selected_model + '/' + character_names[detected_class] + "/" + filename)
  else:
    copyfile(path, "output/" + selected_model + '/false/' + filename)

#load saved model
plt.ion()
plt.show()
for filename in os.listdir(input_directory):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    path = os.path.join(input_directory, filename)
    img = image.load_img(path, target_size=(150, 150))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    print('==============================================')
    print("File      :", filename)

    largest = 0

    for x in range(0, len(classes[0])):
      if(classes[0][x] > largest):
        largest = x

    user_decision(largest, filename, classes[0][largest])

  else:
      continue