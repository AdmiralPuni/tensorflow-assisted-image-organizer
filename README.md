# Tensorflow Assisted Image Organizer | TAIO
 Reducing the amount of work needed to sort out jumbled images in a folder.

https://user-images.githubusercontent.com/42926364/137741853-e7d37463-7f1a-4c77-862f-fd607a3f6900.mp4

 Files to run:
 1. char-2-offline.py | Terminal only displays image with plt/mathplotlib
 2. char-2-gui.py | Currently the terminal achieves the same goal as the gui, more work will be needed
 
 This program comes with two pre trained model:
  - sample
    - Cirno, Gabriel, Karen
    - models/model_weights_saved_char.hdf5
  - hololive-en : 
    - Ame, Gura, Ina, Mori, Kiara
    - models/model-hololive-en.hdf5

 However you can add your own custom model, add a new model decription in settings.json follow the examples, the training notebook is included in  char-2.ipynb simply download the model and move it to /models.

Important Files | Description
------------ | -------------
char-2-offline.py | Main Program
char-2-gui.py | The program but with easier to operate GUI
char-2.ipynb | Jupyter notebook to train the models

Imports
* tensorflow.keras
* json
* tkinter (char-gui-2)
* os
* shutil
* keras.preprocessing
* matplotlib.pyplot
