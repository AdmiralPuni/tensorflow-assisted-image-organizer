import tkinter as tk
from tkinter.constants import BOTH, BOTTOM, LEFT, TOP, W, X, YES

from PIL import ImageTk, Image
import os

root = tk.Tk()
wait_for_click = tk.IntVar()

root.title("Tensorflow Assisted Image Organizer | TAIO v0.1")

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
photo = ImageTk.PhotoImage(Image.open("logpu.png").resize((500,700)))
vlabel.configure(image=photo)

fm = tk.Frame(root)
button_single = tk.Button(fm, text="Single", width=15, font="Calibri 20")
button_multi = tk.Button(fm, text="Multi", width=15, font="Calibri 20")
button_false = tk.Button(fm, text="False", width=15, font="Calibri 20")
button_single.pack(side=LEFT, anchor=W, fill=X, expand=YES)
button_multi.pack(side=LEFT, anchor=W, fill=X, expand=YES)
button_false.pack(side=LEFT, anchor=W, fill=X, expand=YES)
fm.pack(side=TOP, fill=BOTH, expand=YES)

detection_text = tk.Label(root, text="Detected names", font="Calibri 20")
detection_text.pack(side=TOP, fill=BOTH, expand=YES)
vlabel.pack(side=BOTTOM)

root.mainloop()
