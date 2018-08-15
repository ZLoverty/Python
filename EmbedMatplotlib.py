from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2TkAgg)
from matplotlib.backends.backend_agg import FigureCanvasAgg
import tkinter as tk
import numpy as np
import pdb

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.create_widgets()
    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")
        self.initCanvas()
        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=root.destroy)
        self.quit.pack(side="bottom")
    def say_hi(self):
        print("hi there, everyone!")
    def initCanvas(self):
        self.fig = Figure(figsize=(7,7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self) ## Use matplotlib.backend to generate GUI widget
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
    def updateCanvas(self, ax=None):
        X = np.linspace(0, 2 * np.pi, 50)
        Y = np.sin(X)
        ax = self.fig.add_axes([0, 0, 1, 1])
        ax.plot(X, Y)
        self.canvas.draw()
    def on_press(self, event):
        print('you pressed', event.button, event.xdata, event.ydata)
        
root = tk.Tk(className="Embedding matplotlib widget")
app = Application(master=root)
app.updateCanvas()
root.mainloop()
