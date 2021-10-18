# Tensorflow Assisted Image Organizer
 Reducing the amount of work needed to sort out jumbled images in a folder.
 
 

https://user-images.githubusercontent.com/42926364/137637063-355fa517-d5e0-45ab-a31d-87f5583c0b33.mp4


 
 Files to run:
 1. char_2_offline.py | Terminal only
 2. char-gui-2.py | Simple GUI but it still uses terminal for input
 
 The model provided is trained to be used on three anime character and the program is hardcoded to do so:
  1. Cirno
  2. Gabriel Tenma White
  3. Kujou Karen

You will need to modify the program and change the model to it according to your custom category.

Imports
* tensorflow
* PIL
* tkinter (char-gui-2)
* os
* shutil
* numpy
* keras.preprocessing
* matplotlib.pyplot

Knows issues:
* GUI Stretched image
* GUI not fully working
* GUI Code and terminal code is different, gui needs to somehow load the other file
* Class naming is currently manual
