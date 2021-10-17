import tensorflow as tf
import os
from shutil import copyfile
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

filename = "model_weights_saved_char.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')
print(model.summary())
print(model.losses)

directory = 'input'
for filename in os.listdir(directory):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    path = os.path.join(directory, filename)
    img = image.load_img(path, target_size=(150, 150))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    plt.show(block=False)
    plt.pause(1)
    plt.close()

    print('==============================================')
    print("File      :", filename)

    largest = 0

    for x in range(0, len(classes[0])):
      if(classes[0][x] > largest):
        largest = x

    if largest == 0:
      print("Detected  :", 'cirno')
      if input("y/n       : ") == "y":
        copyfile(path, "output/cirno/" + filename)
      else:
        copyfile(path, "output/false/" + filename)
    elif largest == 1:
      print("Detected  :", 'gab')
      if input("y/n       : ") == "y":
        copyfile(path, "output/gab/" + filename)
      else:
        copyfile(path, "output/false/" + filename)
    elif largest == 2:
      print("Detected  :", 'karen')
      if input("y/n       : ") == "y":
        copyfile(path, "output/karen/" + filename)
      else:
        copyfile(path, "output/false/" + filename)
  else:
      continue
