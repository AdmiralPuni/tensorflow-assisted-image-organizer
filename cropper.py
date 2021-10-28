#@title Imports and function definitions
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# For running inference on the TF-Hub module.
import tensorflow as tf

import tensorflow_hub as hub

import threading

from tqdm import tqdm

# For drawing onto the image.
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageFont

def crop_image(output, image, ymin, xmin, ymax, xmax):
  image_pil = image
  im_width, im_height = image_pil.size
  (left, top, right, bottom) = (xmin * im_width,ymin * im_height, xmax * im_width,
                                ymax * im_height)
  image_pil = image_pil.crop((left, top, right, bottom))
  image_pil.save(output)
  #image_pil.save("/content/crop.png")

def draw_boxes(output, image, boxes, class_names, scores, max_boxes=10, min_score=0.3):
  """Overlay labeled boxes on an image with formatted scores and label names."""
  colors = list(ImageColor.colormap.values())

  font = ImageFont.load_default()

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      if class_names[i].decode("ascii") == "Human face" or class_names[i].decode("ascii") == "Human head":
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        
        display_str = "{}: {}%".format(class_names[i].decode("ascii"),
                                      int(100 * scores[i]))
        color = colors[hash(class_names[i]) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        crop_image(output, image_pil, ymin, xmin, ymax, xmax)
  return image


#module_handle = "models/rcnn/"
module_handle = "models/mobilenet/"

detector = hub.load(module_handle).signatures['default']

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img


def run_detector(detector, path, output):
  try:
    img = load_img(path)
  except:
    return

  converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)

  result = {key:value.numpy() for key,value in result.items()}

  image_with_boxes = draw_boxes(output,
      img.numpy(), result["detection_boxes"],
      result["detection_class_entities"], result["detection_scores"])

  #os.remove(path)
  #display_image(image_with_boxes)

path_thread_0 = []
path_thread_1 = []
path_thread_2 = []
path_thread_3 = []

image_paths_list = []

path_count = 0

input_folder = 'images'
master_folder = 'nijigasaki'
output_folder = 'cropper'

class image_paths:
  def __init__(self, input, output):
    self.input = input
    self.output = output

for master in os.listdir(input_folder + '/' + master_folder):
    for slave in os.listdir(input_folder + '/' + master_folder + '/' + master):
        if not os.path.exists(output_folder + '/' + master_folder + '/' + master + '/' + slave):
            os.makedirs(output_folder + '/' + master_folder + '/' + master + '/' + slave)
        for files in os.listdir(input_folder + '/' + master_folder + '/' + master + '/' + slave):
          path_thread_0.append(image_paths(input_folder + '/' + master_folder + '/' + master + '/' + slave + '/' + files,output_folder + '/' + master_folder + '/' + master + '/' + slave + '/' + files))


class myThread (threading.Thread):
   def __init__(self, path_collection_number):
      threading.Thread.__init__(self)
      self.path_collection_number = path_collection_number
   def run(self):
    if self.path_collection_number == 0:
      for files in tqdm(path_thread_0, leave=False):
        run_detector(detector, files.input, files.output)
    if self.path_collection_number == 1:
      for files in tqdm(path_thread_1, leave=False):
        run_detector(detector, files.input, files.output)
    if self.path_collection_number == 2:
      for files in tqdm(path_thread_2, leave=False):
        run_detector(detector, files.input, files.output)
    if self.path_collection_number == 3:
      for files in tqdm(path_thread_3, leave=False):
        run_detector(detector, files.input, files.output)
    #for files in tqdm(image_paths_list[self.path_collection_number], leave=False):
    #    run_detector(detector, files.input, files.output)


myThread(0).start()