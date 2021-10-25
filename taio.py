import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
import getopt
import json
import gc

import tkinter as tk
from tkinter.constants import BOTH, BOTTOM, LEFT, TOP, W, X, YES

import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing import image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from shutil import move



def load_model(character_count, model_filename):
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
    keras.layers.Dense(character_count, activation='softmax')
  ])

  #load saved model
  model.load_weights("models/" + model_filename)
  model.compile(loss='categorical_crossentropy', optimizer='adam')

  return model

def main(argv):
  settings_json = json.load(open('settings.json', 'r'))
  input_directory = ''
  output_directory = ''
  selected_model = ''
  supervision = True
  prediction_list = []

  class prediction:
    def __init__(self, path, names, image):
      self.path = path
      self.names = names
      self.image = image

  try:
    opts, args = getopt.getopt(argv, "hli:o:m:sa", ["i=", "o=", "m="])
  except getopt.GetoptError:
    print('test.py -i <input_directoryfile> -o <output_directory>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
      print('test.py -i <input_directoryfile> -o <output_directory>')
      sys.exit()
    elif opt in ("-l"):
      for model_name in settings_json['models']:
          print(model_name)
      sys.exit()
    elif opt in ("-i"):
      input_directory = arg
    elif opt in ("-o"):
      output_directory = arg
    elif opt in ("-m"):
      selected_model = arg
    elif opt in ("-s"):
      supervision = True
    elif opt in ("-a"):
      supervision = False
  
  print('Input Directory     :', input_directory)
  print('Output Directory    :', output_directory)
  print('Model               :', selected_model)
  print('Supervision         :', supervision)

  character_names = settings_json['models'][selected_model]['names']
  model_filename = settings_json['models'][selected_model]['filename']

  #for name in character_names:
  #  if not os.path.exists(output_directory + '/' + selected_model + '/' + name):
  #    os.makedirs(output_directory + '/' + selected_model + '/' + name)

  print('Loading model...')
  model = load_model(len(character_names), model_filename)
  print("Loading Mobilenet...")
  #module_handle = "models/rcnn/"
  module_handle = "models/mobilenet/"
  detector = hub.load(module_handle).signatures['default']

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

  def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top),(right, bottom)], width=thickness, fill=color)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
              (left, top)],
              width=thickness,
              fill=color)
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
      text_bottom = top
    else:
      text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
      text_width, text_height = font.getsize(display_str)
      margin = np.ceil(0.05 * text_height)
      draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                      (left + text_width, text_bottom)],
                    fill=color)
      draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
      text_bottom -= text_height - 2 * margin

  def crop_and_detect(image, ymin, xmin, ymax, xmax):
    image_pil = image
    im_width, im_height = image_pil.size
    (left, top, right, bottom) = (xmin * im_width,ymin * im_height, xmax * im_width,
                                  ymax * im_height)
    image_pil = image_pil.crop((left, top, right, bottom))
    image_pil.save(output_directory + "/temp.png")
    return get_name(output_directory + "/temp.png")

  def draw_boxes(loaded_image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    color = ImageColor.getrgb('#6AE670')
    font = ImageFont.truetype("arial.ttf", 35)
    image_boxes = loaded_image
    detected_names = []

    for i in range(min(boxes.shape[0], max_boxes)):
      if scores[i] >= min_score:
        if class_names[i].decode("ascii") == "Human face":
          ymin, xmin, ymax, xmax = tuple(boxes[i])
          
          image_pil = Image.fromarray(np.uint8(loaded_image)).convert("RGB")
          detected_names.append(crop_and_detect(image_pil, ymin, xmin, ymax, xmax))

          display_str = "{}: {}%".format(detected_names[-1] + ' | Face', int(100 * scores[i]))
          
          draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
          
          np.copyto(image_boxes, np.array(image_pil))
    #Abomination
    #todo load image preserve aspect ratio
    #Image.fromarray(np.uint8(image_boxes)).convert("RGB").save(output_directory + '/' + 'temp-detected' + "/temp_box.png")
    return image_boxes, detected_names
  
  def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

  def run_detector(detector, path):
    img = load_img(path)
    detected_names = []

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    result = {key:value.numpy() for key,value in result.items()}

    #detect faces>draw box around them> detect the face<repeat >return image with box and detected names
    image_with_boxes, detected_names = draw_boxes(
        img.numpy(), result["detection_boxes"],
        result["detection_class_entities"], result["detection_scores"])

    #Cache the image into a folder to ease matplotlib and tkinter
    Image.fromarray(np.uint8(image_with_boxes)).convert("RGB").save(output_directory + '/' + 'temp-detected' + '/' + os.path.basename(path))
    
    return prediction(path, np.unique(detected_names), output_directory + '/' + 'temp-detected' + '/' + os.path.basename(path))

  print("Running detections...")
  for filename in tqdm(os.listdir(input_directory)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
      path = os.path.join(input_directory, filename)
      prediction_list.append(run_detector(detector, path))
    else:
        continue

  #Attempting to free up RAM, hub ram in still in shackles
  print('Clearing model...')
  del model
  print('Clearing mobilenet...')
  del detector
  print('Clearing keras session')
  keras.backend.clear_session()
  print('Collecting garbage...')
  gc.collect()

  def user_decision(path, detected_names):
    if supervision:
      print("Detected            :", detected_names)
      plt.title(' '.join(detected_names), fontsize=24)
      choice = int(input("Decision            : "))
      if choice == 1:
        move_file(path, detected_names, True)
      elif choice == 2:
        move_file(path, detected_names, False)
      else:
        return
    else:
      if len(detected_names) == 1:
        move_file(path, detected_names, True)
      elif len(detected_names) > 1:
        move_file(path, detected_names, False)

  def move_file(path, detected_names, single):
    if single:
      if not os.path.exists(output_directory + "/" + selected_model + '/' + detected_names[0]):
        os.makedirs(output_directory + "/" + selected_model + '/' + detected_names[0])
      move(path, output_directory + "/" + selected_model + '/' + detected_names[0] + "/" + os.path.basename(path))
    else:
      if not os.path.exists(output_directory + "/" + selected_model + '/' + '-'.join(detected_names)):
        os.makedirs(output_directory + "/" + selected_model + '/' + '-'.join(detected_names))
      move(path, output_directory + "/" + selected_model + '/' + '-'.join(detected_names) + "/" + os.path.basename(path))

  #standard terminal with image using matplotlib(incredibly slow)
  def terminal():
    print('=============================================================')
    print('Decision answers    :')
    print('1. Save in single folder')
    print('2. Save in grouped folder')
    print('3. False detection')
    plt.ion()
    plt.show()

    for list in prediction_list:
      if len(list.names) != 0:
        print('=============================================================')
        #print('Path      : ', list.path)
        plt.imshow(load_img(list.image))
        user_decision(list.path, list.names)

  
  #Tkinter gui stuff, miles faster than matplotlib
  def gui():
    root = tk.Tk()

    root.title("Tensorflow Assisted Image Organizer | TAIO v0.1")

    def cycle_prediction(index, single, save=True):
      
      current_prediction = prediction_list[index]
      if save:
        move_file(current_prediction.path, current_prediction.names, single)
      #attempting to treat out of index on last prediction
      try:
        current_prediction = prediction_list[index+1]
        #Skip if prediction doesn't detect any names
        if len(current_prediction.names) == 0:
          cycle_prediction(get_index(), False, False)
          return
        change_pic(vlabel, current_prediction.image)
        change_name(detection_text, current_prediction.names)
      except:
        change_name(detection_text, 'Task completed')

    def get_index(index = []):
      index.append(0)
      return len(index)-1

    def change_pic(labelname, file_path, max_width = 500):
        loaded_image = Image.open(file_path)
        width, height = loaded_image.size
        ratio_height = height/width
        photo1 = ImageTk.PhotoImage(loaded_image.resize((max_width, round(max_width*ratio_height))))
        labelname.configure(image=photo1)
        labelname.photo = photo1

    def change_name(labelname, text):
        labelname.configure(text=text)

    vlabel = tk.Label(root)
    photo = ImageTk.PhotoImage(Image.open(output_directory + "/temp.png").resize((500,500)))
    vlabel.configure(image=photo)

    fm = tk.Frame(root)
    button_single = tk.Button(fm, text="Single", width=15,command=lambda: cycle_prediction(get_index(), True), font="Calibri 20")
    button_multi = tk.Button(fm, text="Multi", width=15,command=lambda: cycle_prediction(get_index(), False), font="Calibri 20")
    button_false = tk.Button(fm, text="False", width=15,command=lambda: cycle_prediction(get_index(), True, False), font="Calibri 20")
    button_single.pack(side=LEFT, anchor=W, fill=X, expand=YES)
    button_multi.pack(side=LEFT, anchor=W, fill=X, expand=YES)
    button_false.pack(side=LEFT, anchor=W, fill=X, expand=YES)
    fm.pack(side=TOP, fill=BOTH, expand=YES)

    detection_text = tk.Label(root, text="Detected names", font="Calibri 20")
    detection_text.pack(side=TOP, fill=BOTH, expand=YES)
    vlabel.pack(side=BOTTOM)

    current_prediction = prediction_list[0]
    change_pic(vlabel, current_prediction.image)
    change_name(detection_text, current_prediction.names)

    root.mainloop()

  gui()


if __name__ == "__main__":
    main(sys.argv[1:])
