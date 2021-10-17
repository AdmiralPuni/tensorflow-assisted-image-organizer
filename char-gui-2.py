import tkinter as tk

from PIL import ImageTk, Image

import tensorflow as tf
import os
from shutil import copyfile
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

root = tk.Tk()
wait_for_click = tk.IntVar()


root.title("Assisted Image Categorization | AIC v0.1")
window_height = 500
window_width = 540
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width,
              window_height, x_cordinate, y_cordinate))


def change_pic(labelname, file_path):
    photo1 = ImageTk.PhotoImage(Image.open(file_path).resize((500, 700)))
    labelname.configure(image=photo1)
    labelname.photo = photo1


def change_name(labelname, text):
    labelname.configure(text=text)


def start_categorization():
    b3.pack_forget()
    
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

    def category_selection(name):
        print("Detected  :", name)
        change_name(detected_name, name)
        if input("3 = true  : ") == "3":
            copyfile(path, "output/" + name + "/" + filename)
        else:
            copyfile(path, "output/false/" + filename)
        
            
    print("[INFO] To be used with the terminal. GUI incomplete")
    print("[INFO] When asked for input press 3 then enter if the detection is correct")
    directory = 'input'
    for filename in os.listdir(directory):
        wait_for_click = 3
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(directory, filename)
            img = image.load_img(path, target_size=(150, 150))
            imgplot = plt.imshow(img)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)

            change_pic(vlabel, path)

            print('=============================================================')
            print("File      :", filename)

            largest = 0

            for x in range(0, len(classes[0])):
                if(classes[0][x] > largest):
                    largest = x

            if largest == 0:
                category_selection('cirno')
            elif largest == 1:
                category_selection('gab')
            elif largest == 2:
                category_selection('karen')
        else:
            continue
    change_name(detected_name, "Completed, check the output folder.")


vlabel = tk.Label(root)
photo = ImageTk.PhotoImage(Image.open("logpu.png").resize((500,700)))
vlabel.configure(image=photo)
vlabel.pack()
detected_name = tk.Label(root, text="Press START to begin", font="Calibri 20")
detected_name.pack()
#b1 = tk.Button(root, text="FALSE", command=lambda: wait_for_click.set(0), width=15, font="Calibri 20")
#b1.pack(side="left")
#b2 = tk.Button(root, text="TRUE", command=lambda: wait_for_click.set(1), width=15, font="Calibri 20")
#b2.pack(side="right")

b3 = tk.Button(root, text="START", command=lambda: start_categorization(), width=15, font="Calibri 20")
b3.pack()

root.mainloop()
