from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename, b =  filedialog.asksaveasfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"), ("PDF", "*.pdf"), ("all files","*.*")))
print (root.filename)