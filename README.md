<p align="center">
  <img src="https://files.catbox.moe/r9l8px.png" />
</p>

<p align="center">
  <img src="https://files.catbox.moe/oxq5q9.png" />
</p>

https://user-images.githubusercontent.com/42926364/138804997-638e03f7-0be0-4f7a-87ec-899ecad1ab65.mp4


## Running The Program

Example : taio.py -i input -o output -m myusu -s

### Installation

For windows you can run this command on powershell : c:/Windows/py.exe -m pip install #modulename

Name | Method | Linux
------------ | ------------- | -------------
python 3.9 | Windows installation | python is included
tensorflow | pip install tensorflow | pip install tensorflow
tensorflow_hub | pip install tensorflow-hub | pip install tensorflow-hub
matplotlib | pip install matplotlib | pip install matplotlib
keras | pip install keras | pip install keras
tqdm | pip install tqdm | pip install tqdm
numpy | pip install numpy | pip install numpy
PIL | pip install pillow | pip install pillow
Tkinter | pip install tkinter/tk | pacman -S tk

### Arguments

#### Mandatory arguments
Option | Description | Example
------------ | ------------- | ------------- 
-i | Input directory | input
-o | Output directory | output
-m | Model name | myusu
Choose one | - | -
-s | Supervised decision, a choice will appear when character is detected | 
-a | Unsupervised, detections are automatically moved to output folder. Not recommended since false positives is quite often with large model | 

#### Information arguments
Option | Description
------------ | -------------
-h | Show help
-l | List models in settings.json

## Models

Name | Characters | File
------------ | ------------- | ------------- 
μ's | models/model-myusu-v2.hdf5 | Eli, Hanayo, Honoka, Kotori, Maki, Nico, Nozomi, Rin, Umi
Hololive | models/model-hololive-v2.hsf5 | Akai haato, Aki rosenthal, Amane kanata, Gawr gura, Himemori luna, Hoshimachi suisei, Houshou marine, Inugami korone, Kiryu coco, Minato aqua, Momosuzu nene, Mori calliope, Murasaki shion, Nakiri ayame, Natsuiro matsuri, Nekomata okayu, Ninomae ina'nis, Ookami mio, Oozora subaru, Sakura miko, Shirakami fubuki, Shiranui flare, Shirogane noel, Takanashi kiara, Tokino sora, Tokoyami towa, Tsunomaki watame, Uruha rushia, Usada pekora, Watson amelia, Yozora mel, Yukihana lamy, Yuzuki choco

You can add your own custom model, add a new model decription in settings.json follow the examples, the training notebook is included in  char-2.ipynb simply download the model and move it to /models. Tutorial will be added soon into the jupyter notebook(char-2.ipynb)

Or you can request a model to be made and included in the repository, open an issue or contact me about it. It would be better if you can provide 150 or more solo images of the characters you requested.

## Files

Important Files | Function | Description
------------ | ------------- | -------------
taio.py | Main program | Detect faces categorize it
cropper.py | Face cropper | Crop faces in images and saves it, used for making models
char-5.ipynb | Jupyter notebook to train the models | images needs to be separated in a zip file and structured "modelname/character"

## TODO
 - CBIR method discovered, I'm going to try it and if it's more accurate and easier it will be implemented to a separate program

## Performance

Test on myusu model with 12 input image

GPU/CPU | Operating System |  Time
------------ | ------------- | -------------
Intel Core I5 3320M | Manjaro 21.1.4 | 37.37 s
AMD Athlon X4 860K | Windows 10 1903 | 53.56 s
