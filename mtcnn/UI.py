from tkinter import*
import os
from subprocess import call
import pathlib
import cv2
import numpy as np
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed 
import shutil

# building the UI for the app 
root = Tk()
root.title('Sign in system')


def takepic():
    call(["python", "project_1.py"])

    return

def predict():
    call(["python", "project_2.py"])

    return

def show_img():
    call(["python",'show_pic.py'])

def new_pic():
	call(['python','new_user_pics.py'])

def train_new():
        call(['python','split_pics.py'])


my_button = Button(root,text='Take pic',width =40, padx =20, pady=10, fg='black', command=takepic)
my_button2 = Button(root,text='Sign In',width = 40, padx =20, pady=10, fg='black',command=predict)
my_button3 = Button(root,text='show image ',width = 40, padx =20, pady=10, fg='black',command=show_img)
my_button4 = Button(root,text='Add member ',width = 40, padx =20, pady=10, fg='black',command=new_pic)
my_button5 = Button(root,text='train',width = 40, padx =20, pady=10, fg='black',command=train_new)


my_button.grid(row=0,column=0)
my_button2.grid(row=1,column=0)
my_button3.grid(row=3,column=0)
my_button4.grid(row=4,column=0)
my_button5.grid(row=5,column=0)




root.mainloop()
