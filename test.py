import os

input_folder = 'images'
master_folder = 'model-hololive-en'
output_folder = 'cropper'

for master in os.listdir(input_folder + '/' + master_folder):
    print(input_folder + '/' + master_folder + '/' + master)
    for slave in os.listdir(input_folder + '/' + master_folder + '/' + master):
        print(input_folder + '/' + master_folder + '/' + master + '/' + slave)
        if not os.path.exists(output_folder + '/' + master_folder + '/' + master + '/' + slave):
            os.makedirs(output_folder + '/' + master_folder + '/' + master + '/' + slave)

