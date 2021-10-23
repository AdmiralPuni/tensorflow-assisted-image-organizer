#@title Imports and function definitions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

import threading

from tqdm import tqdm

import matplotlib.pyplot as plt

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont

from random import randrange

import chardef

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

def crop_image(image, ymin, xmin, ymax, xmax):
  image_pil = image
  im_width, im_height = image_pil.size
  (left, top, right, bottom) = (xmin * im_width,ymin * im_height, xmax * im_width,
                                ymax * im_height)
  image_pil = image_pil.crop((left, top, right, bottom))
  image_pil.save("cropper/temp.png")
  return chardef.get_name("cropper/temp.png")
  #print(chardef.get_name("cropper/temp.png"))
  #image_pil.save("/content/crop.png")

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.2):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())
  #print('==============================================')
  font = ImageFont.truetype("arial.ttf", 35)
  image_boxes = image
  detected_names = []

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      if class_names[i].decode("ascii") == "Human face":
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        
        color = colors[hash(class_names[i]) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        detected_names.append(crop_image(image_pil, ymin, xmin, ymax, xmax))

        display_str = "{}: {}%".format(detected_names[-1], int(100 * scores[i]))
        
        draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
        
        np.copyto(image_boxes, np.array(image_pil))
  return image_boxes, detected_names


print("Loading Mobilenet...")
#module_handle = "models/rcnn/"
module_handle = "models/mobilenet/"

detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.grid(False)
  plt.imshow(image)

class prediction:
  def __init__(self, path, names, image):
    self.path = path
    self.names = names
    self.image = image

prediction_list = []

def run_detector(detector, path):
  img = load_img(path)
  detected_names = []

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}

  image_with_boxes, detected_names = draw_boxes(
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])


  return prediction(path, detected_names, image_with_boxes)
  #display_image(image_with_boxes)

for filename in tqdm(os.listdir('input')):
  if filename.endswith(".jpg") or filename.endswith(".png"):
    path = os.path.join('input', filename)
    prediction_list.append(run_detector(detector, path))
  else:
      continue

def user_decision(detected_names):
  print("Detected  :", set(detected_names))
  plt.title(set(detected_names), fontsize=24)
  input("3=true    : ")

plt.ion()
plt.show()
plt.imshow(chardef.image.load_img('logpu.png', target_size=(150, 150)))

for list in prediction_list:
  print('==============================================')
  print('Path      : ', list.path)
  plt.imshow(list.image)
  user_decision(list.names)

