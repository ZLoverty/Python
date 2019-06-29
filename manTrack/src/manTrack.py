import tkinter as tk
import tkinter.filedialog as TFD
from matplotlib.figure import Figure
from matplotlib.image import imread
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import numpy as np
import pandas as pd
import math
import pdb
class mplApp(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.createVars()
        self.create_widgets()        
    """
    Components
    """
    def createVars(self):
        self.mousePos = [0, 0]
        self.mousePosStringVar = tk.StringVar()
        self.mousePosStringVar.set(str(self.mousePos[0]) + ', ' + str(self.mousePos[1]))
        self.dID = -1
        self.tID = -1
    def create_widgets(self):
        self.loadButton = tk.Button(self, text='Load', command=self.imgOpenDialog)
        self.loadButton.pack()
        self.loadDataButton = tk.Button(self, text='Load data', command=self.dataOpenDialog)
        self.loadDataButton.pack()
        self.drawDataButton = tk.Button(self, text='Draw data', command=self.drawData)
        self.drawDataButton.pack()
        self.reloadButton = tk.Button(self, text='Reload', command=self.reloadButtonCallback)
        self.reloadButton.pack()
        self.PPUEntry = tk.Entry(self)
        self.PPUEntry.pack()
        self.mousePositionLabel = tk.Label(self, textvariable=self.mousePosStringVar)
        self.mousePositionLabel.pack()
        self.mouseDeleteModeButton = tk.Button(self, text='Delete mode', command=self.mouseDeleteModeButtonCallback)
        self.mouseDeleteModeButton.pack()
        
        self.mouseTrackModeButton = tk.Button(self, text='Track mode', command=self.mouseTrackModeButtonCallback)
        self.mouseTrackModeButton.pack()
        self.mergeDataButton = tk.Button(self, text='Merge data', command=self.mergeDataButtonCallback)
        self.mergeDataButton.pack()
        self.saveDataButton = tk.Button(self, text='Save data', command=self.saveDataButtonCallback)
        self.saveDataButton.pack()

        
        self.testButton = tk.Button(self, text='test', command=self.testButtonCallback)
        self.testButton.pack()
        
    def initCanvas(self, ax):        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self) ## Use matplotlib.backend to generate GUI widget
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()
        self.pID = self.canvas.mpl_connect('motion_notify_event', self.mousePosCallback)
    """
    Callbacks
    """
    def imgOpenDialog(self):
        try:
            self.canvas.get_tk_widget().destroy()
        except:
            pass
        imgDir = TFD.askopenfilename()
        img = imread(imgDir)
        self.img = img
        h, w = img.shape
        dpi = 100
        hcanvas = h
        wcanvas = w
        self.compressRatio = 1
        if wcanvas > 800:
            wcanvas = 800
            hcanvas = h/w*wcanvas
            self.compressRatio = 800/w
        if hcanvas > 600:
            hcanvas = 600
            wcanvas = w/h*hcanvas
            self.compressRatio = 600/h
        self.fig = Figure(figsize=(wcanvas/dpi,hcanvas/dpi),dpi=dpi)
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.imshow(img, cmap='gray')
        self.initCanvas(self.ax)
        
    def dataOpenDialog(self):
        dataDir = TFD.askopenfilename()
        if dataDir.endswith('.dat'):
            self.data = pd.read_csv(dataDir, delimiter='\t')
        else:
            ValueError('Wrong data type, please open *.dat text file')
          
    def drawData(self):
        try:
            self.PPU = float(self.PPUEntry.get())          
        except:
            print('Enter pixel per unit in the blank')
            return
        PPU = self.PPU
        h, w = self.img.shape
        for name, value in self.data.iterrows():
            xy = (value.X*PPU, h - value.Y*PPU)
            width = value.Major*PPU
            height = value.Minor*PPU
            angle = value.Angle
            elli = mpatches.Ellipse(xy, width, height, angle)
            elli.set_fill(False)
            self.ax.add_patch(elli)
            elli.set_picker(True)         
        self.canvas.draw()

    def reloadButtonCallback(self):
        self.canvas.get_tk_widget().destroy()
        img = self.img
        h, w = img.shape
        dpi = 100
        hcanvas = h
        wcanvas = w
        self.compressRatio = 1
        if wcanvas > 800:
            wcanvas = 800
            hcanvas = h/w*wcanvas
            self.compressRatio = 800/w
        if hcanvas > 600:
            hcanvas = 600
            wcanvas = w/h*hcanvas
            self.compressRatio = 600/h
        self.fig = Figure(figsize=(wcanvas/dpi,hcanvas/dpi),dpi=dpi)
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.imshow(img, cmap='gray')
        self.initCanvas(self.ax)
    def mouseDeleteModeButtonCallback(self):
        self.dID = self.canvas.mpl_connect('pick_event', self.mouseDeleteCallback)
        self.canvas.mpl_disconnect(self.tID)
    def mouseTrackModeButtonCallback(self):
        self.tID = self.canvas.mpl_connect('button_press_event', self.mouseTrackPressCallback) #####
        
        self.canvas.mpl_disconnect(self.dID)    
    def mouseDeleteCallback(self, event):
        def common_member(a, b): 
            a_set = set(a) 
            b_set = set(b) 
            if (a_set & b_set): 
                common = a_set & b_set
                return list(common)
            else: 
                print("No common elements")
                return -1
        h, w = self.img.shape
        artist = event.artist
        artist.set_visible(False)
        self.canvas.draw()
        xy = artist.center
        x = xy[0]
        y = h - xy[1]
        xData = self.data.X*self.PPU 
        yData = self.data.Y*self.PPU 
        index = self.data[xData==x].index.tolist()     
        index_y = self.data[yData==y].index.tolist()
        index = common_member(index, index_y)      
        self.old_data = self.data
        self.data = self.data.drop(axis=0, index=index)
        print(artist, index, ' deleted!')
        print(len(self.data), ' particles left')
        
    
        
    def mouseTrackPressCallback(self, event):
        print('you pressed', event.button, event.xdata, event.ydata)
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.releaseID = self.canvas.mpl_connect('button_release_event', self.mouseTrackReleaseCallback)
        

        
    def mouseTrackReleaseCallback(self, event):
        h, w = self.img.shape
        print('you released', event.button, event.xdata, event.ydata)
        self.x2 = event.xdata
        self.y2 = event.ydata
        self.ax.plot([self.x1, self.x2], [self.y1, self.y2])
        self.canvas.mpl_disconnect(self.releaseID)
        self.canvas.draw()
        No = -1;
        Area = -1;
        X = (self.x1 + self.x2) / 2 / self.PPU
        Y = (h - (self.y1 + self.y2) / 2) / self.PPU
        Major = ((self.x1 - self.x2)**2+(self.y1 - self.y2)**2)**.5 / self.PPU
        Minor = Major / 6
        Angle = math.atan((self.y1 - self.y2)/(self.x1 - self.x2))
        if Angle < 0:
            Angle = Angle + math.pi
        Angle = math.degrees(Angle)
        Slice = 1
        data = np.array([[No, Area, X, Y, Major, Minor, Angle, Slice]])
        header = self.data.columns.tolist()
        addedDataFrame = pd.DataFrame(data=data, columns=header)
        try:
            self.addedData = self.addedData.append(addedDataFrame)
        except:
            self.addedData = addedDataFrame  
    def mousePosCallback(self, event):
        h, w = self.img.shape
        self.mousePos = [event.x/self.compressRatio, h - event.y/self.compressRatio]
        self.mousePosStringVar.set(str(self.mousePos[0]) + ', ' + str(self.mousePos[1]))
    def mergeDataButtonCallback(self):
        self.data = self.data.append(self.addedData, ignore_index=False)
        self.addedData = None
    def saveDataButtonCallback(self):
        saveName = TFD.asksaveasfilename(initialdir=os.getcwd(), title='Select file')
        self.data.to_csv(saveName, index=False, sep='\t',float_format='%.3f')
    def testButtonCallback(self):
        pass
    
if __name__ == '__main__':
    root = tk.Tk(className="manTrack")
    app = mplApp(master=root)
    app.pack()
    root.mainloop()
