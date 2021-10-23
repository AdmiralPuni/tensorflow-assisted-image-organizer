# Tensorflow Assisted Image Organizer | TAIO
 Reducing the amount of work needed to sort out jumbled images in a folder.



https://user-images.githubusercontent.com/42926364/138214467-9bfe112d-d3ad-49f1-851b-d0ef09e95649.mp4

 ## Running The Program
 Example : taio.py -i input -o output -m myusu -s
 
 ### Mandatory arguments
 Option | Description | Example
 ------------ | ------------- | ------------- 
 -i | Input directory | input
 -o | Output directory | output
 -m | Model name | myusu
 Choose one
 -s | Supervised decision, a choice will appear when character is detected | 
 -a | Unsupervised, detections are automatically moved to output folder | 
 
 ### Information arguments
 Option | Description
 ------------ | -------------
 -h | Show help
 -l | List models in settings.json
 
 ## Models
  - sample
    - Cirno, Gabriel, Karen
    - models/model_weights_saved_char.hdf5
  - hololive-en : 
    - Ame, Gura, Ina, Mori, Kiara
    - models/model-hololive-en-faces.hdf5
  - μ's/myusu : 
    - Eli, Hanayo, Honoka, Kotori, Maki, Nico, Nozomi, Rin, Umi
    - models/model-myusu.hdf5

 You can add your own custom model, add a new model decription in settings.json follow the examples, the training notebook is included in  char-2.ipynb simply download the model and move it to /models. Tutorial will be added soon into the jupyter notebook(char-2.ipynb)

## Files

Important Files | Function | Description
------------ | ------------- | ------------- 
char-2-offline.py | Main Program | Terminal only displays image with plt/mathplotlib
char-2-gui.py | The program but with easier to operate GUI | Currently the terminal achieves the same goal as the gui, more work will be needed
char-2.ipynb | Jupyter notebook to train the models | images needs to be separated in a zip file and structured "modelname/train/character" and "modelname/test/character"

## Imports
* tensorflow.keras
* json
* tkinter (char-gui-2)
* os
* shutil
* keras.preprocessing
* matplotlib.pyplot
