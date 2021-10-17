import tkinter as tk

from PIL import ImageTk, Image


root = tk.Tk()
wait_for_click = tk.IntVar()



root.title("Centering windows")
window_height = 500
window_width = 540
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


def change_pic(labelname):
    photo1 = ImageTk.PhotoImage(Image.open("input/1533146286016.jpg").resize((300,300)))
    labelname.configure(image=photo1)
    labelname.photo = photo1
    

def change_name(labelname):
    labelname.configure(text="asdasdasd")

vlabel = tk.Label(root)
photo = ImageTk.PhotoImage(Image.open("input/1534069659568.png").resize((300,300)))
vlabel.configure(image=photo)
vlabel.pack()
detected_name = tk.Label(root, text="gab", font="Calibri 20")
detected_name.pack()
b2 = tk.Button(root, text="True", command=lambda: change_pic(vlabel), width=15, font="Calibri 20")
b2.pack(side="left")
b3 = tk.Button(root, text="False", command=lambda: change_name(detected_name), width=15, font="Calibri 20")
b3.pack(side="right")

root.mainloop()
