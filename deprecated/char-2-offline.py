print('loading...')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow.keras as keras
import json
from shutil import move
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tqdm import tqdm


selected_model = ""
character_names = []
model_filename = ""
settings_filename = "settings.json"
settings_json = json.load(open(settings_filename, 'r'))
input_directory = 'input'
output_directory = 'output'
model_directory = 'models'

prediction_list = []

class prediction:
  def __init__(self, filename, classes, image):
    self.filename = filename
    self.classes = classes
    self.image = image

plt.ion()
plt.show()
plt.imshow(image.load_img('logpu.png', target_size=(150, 150)))


print('TAIO v0.1 TERMINAL')
print('==============================================')

for index, model_name in enumerate(settings_json['models']):
  print(model_name)

print('==============================================')

selected_model = input('Select model : ')

character_names = settings_json['models'][selected_model]['names']
model_filename = settings_json['models'][selected_model]['filename']

print('==============================================')
print('Predicting images...')




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

def user_decision(detected_class, filename):
  print("Detected  :", character_names[detected_class])
  plt.title(character_names[detected_class].upper(), fontsize=24)
  if input("3=true    : ") == "3":
    move(os.path.join(input_directory, filename), output_directory + "/" + selected_model + '/' + character_names[detected_class] + "/" + filename)
#load saved model

for filename in tqdm(os.listdir(input_directory)):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    path = os.path.join(input_directory, filename)
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    prediction_list.append(prediction(filename, classes, img))
  else:
      continue

for list in prediction_list:
  plt.imshow(list.image)

  print('==============================================')
  print("File      :", list.filename)

  largest = 0

  for x in range(0, len(list.classes[0])):
    if(list.classes[0][x] > largest):
      largest = x

  user_decision(largest, list.filename)
