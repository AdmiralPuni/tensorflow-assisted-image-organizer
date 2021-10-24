# TAIO | Tensorflow Assisted Image Organizer 
Reducing the amount of work needed to sort out jumbled images in a folder.

![unknown_2021 10 24-15 59](https://user-images.githubusercontent.com/42926364/138587802-1dab8452-e8d6-462e-823d-8c0a417b9b99.png)

## Running The Program

Example : taio.py -i input -o output -m myusu -s

### Installation

Name | Method 
------------ | -------------
python 3.9 | Windows installation
tensorflow | pip
tensorflow_hub | pip
matplotlib | pip 
keras | pip
tqdm | pip
numpy | pip
PIL | pip
shutil | pip

### Arguments

#### Mandatory arguments

Option | Description | Example
------------ | ------------- | ------------- 
"-i" | Input directory | input
"-o" | Output directory | output
"-m" | Model name | myusu
Choose one | "-" | "-"
"-s" | Supervised decision, a choice will appear when character is detected | 
"-a" | Unsupervised, detections are automatically moved to output folder. Not recommended since false positives is quite often with large model | 

#### Information arguments

Option | Description
------------ | -------------
"-h" | Show help
"-l" | List models in settings.json

## Models

Name | Characters | File
------------ | ------------- | ------------- 
μ's | models/model-myusu.hdf5 | Eli, Hanayo, Honoka, Kotori, Maki, Nico, Nozomi, Rin, Umi
Hololive EN Gen-1 | models/model-hololive-en-faces.hdf5 | Ame, Gura, Ina, Mori, Kiara
Hololive | models/model-hololive.hsf5 | Akai haato, Amane kanata, Gawr gura, Himemori luna, Hoshimachi suisei, Houshou marine, Inugami korone, Kiryu coco, Minato aqua, Momosuzu nene, Mori calliope, Murasaki shion, Nakiri ayame, Natsuiro matsuri, Nekomata okayu, Ninomae_inanis, Ookami mio, Oozora subaru, Ouro kroni, Sakura miko, Shirakami fubuki, Shiranui flare, Shirogane noel, Shishiro botan, Takanashi kiara, Tokino sora, Tokoyami towa, Tsunomaki watame, Uruha rushia, Usada pekora, Watson amelia, Yukihana lamy

You can add your own custom model, add a new model decription in settings.json follow the examples, the training notebook is included in  char-2.ipynb simply download the model and move it to /models. Tutorial will be added soon into the jupyter notebook(char-2.ipynb)

Or you can request a model to be made and included in the repository, open an issue or contact me about it. It would be better if you can provide 150 or more solo images of the characters you requested.

## Files

Important Files | Function | Description
------------ | ------------- | -------------
taio.py | Main program | Detect faces categorize it
cropper.py | Face cropper | Crop faces in images and saves it, used for making models
char-2.ipynb | Jupyter notebook to train the models | images needs to be separated in a zip file and structured "modelname/train/character" and "modelname/test/character"

## TODO
 - Model generalization / submodel detection
   - New detection method, run the image to general model then to the submodel
 - Help argument
 - Handling error when running without argument
 - Matplotlib image load slows down decision making process
 - Overall program optimization

