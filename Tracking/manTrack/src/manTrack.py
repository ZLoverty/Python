import tkinter as tk
import tkinter.filedialog as TFD
import tkinter.messagebox as TMB
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib import colors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ctypes
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
        self.mode = tk.StringVar()
        self.colorButtonText = tk.StringVar()
        self.dataStatStringVar = tk.StringVar()
        self.deleteTmpStringVar = tk.StringVar()
        self.addTmpStringVar = tk.StringVar()        
        self.PPUStringVar = tk.StringVar()
        self.minorAxisStringVar = tk.StringVar()
        self.PPULabelStringVar = tk.StringVar()
        self.minorAxisLabelStringVar = tk.StringVar()
        self.PPU = 100
        self.minorAxis = 10
        self.mousePosStringVar.set(str(self.mousePos[0]) + ', ' + str(self.mousePos[1]))
        self.mode.set('I')
        self.colorButtonText.set('Color plot')
        self.workingDir = os.getcwd()
        
        self.dID = -1
        self.tID = -1
        self.deletedArtist = []
        self.addedArtist = []
        self.artistList = []
        self.data = pd.DataFrame()
        self.addedData = pd.DataFrame()
        self.deletedData = pd.DataFrame()
        self.PPUStringVar.set(str(self.PPU))
        self.minorAxisStringVar.set(str(self.minorAxis))
        
    def create_widgets(self):
        # DATA LOADING buttons
        self.buttonFrame = tk.Frame(self)
        self.buttonFrame.pack(side='left')
        loadLabel = tk.Label(self.buttonFrame, text='DATA LOADING', font=('Helvetica', 10, 'bold'))
        loadLabel.pack(fill='x')
        self.loadButton = tk.Button(self.buttonFrame, text='Load', command=self.imgOpenDialog)
        self.loadButton.pack(fill='x')
        self.loadDataButton = tk.Button(self.buttonFrame, text='Load data', command=self.dataOpenDialog)
        self.loadDataButton.pack(fill='x')
        self.drawDataButton = tk.Button(self.buttonFrame, text='Draw data', command=self.drawData)
        self.drawDataButton.pack(fill='x')
        self.reloadButton = tk.Button(self.buttonFrame, text='Reload', command=self.reloadButtonCallback)
        self.reloadButton.pack(fill='x')
        spaceFrame1 = tk.Frame(self.buttonFrame, height=30)
        spaceFrame1.pack()
        
		# DATA SAVING buttons
        saveLabel = tk.Label(self.buttonFrame, text='DATA SAVING', font=('Helvetica', 10, 'bold'))
        saveLabel.pack(fill='x')
        self.mergeDataButton = tk.Button(self.buttonFrame, text='Merge data', command=self.mergeDataButtonCallback)
        self.mergeDataButton.pack(fill='x')
        self.saveDataButton = tk.Button(self.buttonFrame, text='Save data', command=self.saveDataButtonCallback)
        self.saveDataButton.pack(fill='x')
        self.saveFigButton = tk.Button(self.buttonFrame, text='Save figure', command=self.saveFigButtonCallback)
        self.saveFigButton.pack(fill='x')
        
        # MODE buttons
        spaceFrame2 = tk.Frame(self.buttonFrame, height=30)
        spaceFrame2.pack()  
        modeLabel = tk.Label(self.buttonFrame, text='MODE', font=('Helvetica', 10, 'bold'))
        modeLabel.pack(fill='x')
        MODES = [('Idle mode', 'I'), ('Delete mode', 'D'), ('Track mode', 'T')]
        for text, mode in MODES:
            rb = tk.Radiobutton(self.buttonFrame, text=text, variable=self.mode, value=mode, indicatoron=0,
                                command=self.modeCallback)
            rb.pack(fill='x')

        # Miscellulose buttons and other widgets
        spaceFrame2 = tk.Frame(self.buttonFrame, height=30)
        spaceFrame2.pack()
        PPULabel = tk.Label(self.buttonFrame, text='INPUT PPU & MA', font=('Helvetica', 10, 'bold'))
        PPULabel.pack(fill='x')
        inputFrame1 = tk.Frame(self.buttonFrame, height=12)
        inputFrame1.pack()
        PPULabel = tk.Label(inputFrame1, text='PPU', width = 7)
        PPULabel.pack(side='left', fill='y')
        self.PPUEntry = tk.Entry(inputFrame1, textvariable=self.PPUStringVar, width=8)
        self.PPUEntry.pack(side='left', fill='y')
        self.PPUEntry.bind('<Return>', self.PPUEnterCallback)
        inputFrame2 = tk.Frame(self.buttonFrame, height=12)
        inputFrame2.pack()
        MALabel = tk.Label(inputFrame2, text='MinAxis', width = 7)
        MALabel.pack(side='left', fill='y')
        self.minorAxisEntry = tk.Entry(inputFrame2, textvariable=self.minorAxisStringVar, width=8)
        self.minorAxisEntry.pack(fill='y')
        self.minorAxisEntry.bind('<Return>', self.MAEnterCallback)                     
        self.backwardButton = tk.Button(self.buttonFrame, text='Backward', state='disabled', command=self.backwardButtonCallback)
        self.backwardButton.pack(fill='x')       
        self.colorButton = tk.Button(self.buttonFrame, text='Color/Mono plot', command=self.colorButtonCallback)
        self.colorButton.pack(fill='x')
        
        # Status block, tracking status of background data
        spaceFrame2 = tk.Frame(self.buttonFrame, height=30)
        spaceFrame2.pack()
        statLabel = tk.Label(self.buttonFrame, text='STATUS BLOCK', font=('Helvetica', 10, 'bold'))
        statLabel.pack(fill='x')
        self.mousePositionLabel = tk.Label(self.buttonFrame, textvariable=self.mousePosStringVar)
        self.mousePositionLabel.pack(fill='x') 
        self.dataStatLabel = tk.Label(self.buttonFrame, textvariable=self.dataStatStringVar)
        self.dataStatLabel.pack(fill='x')
        self.deleteTmpLabel = tk.Label(self.buttonFrame, textvariable=self.deleteTmpStringVar)
        self.deleteTmpLabel.pack(fill='x')
        self.addTmpLabel = tk.Label(self.buttonFrame, textvariable=self.addTmpStringVar)
        self.addTmpLabel.pack(fill='x')
        self.PPULabel = tk.Label(self.buttonFrame, textvariable=self.PPULabelStringVar)
        self.PPULabel.pack(fill='x')
        self.minorAxisLabel = tk.Label(self.buttonFrame, textvariable=self.minorAxisLabelStringVar)
        self.minorAxisLabel.pack(fill='x')
        
        # self.testButton = tk.Button(self.buttonFrame, text='test', command=self.testButtonCallback)
        # self.testButton.pack(fill='x')
        self.updateStatus()
    def initCanvas(self):        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self) ## Use matplotlib.backend to generate GUI widget
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='left')
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
        folder, filename = os.path.split(imgDir)
        if filename.endswith('.tif') is not True:
            TMB.showerror('File type error', 'Please open *.tif file')
            return
        self.workingDir = folder
        img = imread(imgDir)        
        self.img = img
        h, w = img.shape
        dpi = 100
        hcanvas = h
        wcanvas = w
        self.compressRatio = 1
        user32 = ctypes.windll.user32
        wmax = math.floor(0.92*user32.GetSystemMetrics(0))
        hmax = math.floor(0.92*user32.GetSystemMetrics(1))
        if wcanvas > wmax:
            wcanvas = wmax
            hcanvas = h/w*wcanvas
            self.compressRatio = wmax/w
        if hcanvas > hmax:
            hcanvas = hmax
            wcanvas = w/h*hcanvas
            self.compressRatio = hmax/h
        self.fig = Figure(figsize=(wcanvas/dpi,hcanvas/dpi),dpi=dpi)
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.imshow(img, cmap='gray')
        self.initCanvas()
        
    def dataOpenDialog(self):
        dataDir = TFD.askopenfilename(initialdir = self.workingDir)
        if dataDir.endswith('.dat') or dataDir.endswith('.txt'):
            folder, filename = os.path.split(dataDir)
            self.workingDir = folder
            self.data = pd.read_csv(dataDir, delimiter='\t')
            self.updateStatus()
        else:
            TMB.showerror('File type error', 'Please open *.dat or *.txt file')

         
    def drawData(self):
        PPU = self.PPU
        h, w = self.img.shape
        for name, value in self.data.iterrows():
            xy = (value.X*PPU, h - value.Y*PPU)
            width = value.Major*PPU
            height = value.Minor*PPU
            angle = value.Angle
            elli = mpatches.Ellipse(xy, width, height, angle)
            elli.set_fill(False)
            elli.set_picker(True)
            elli.set_color('green')
            self.ax.add_patch(elli)            
            self.artistList.append(elli)
        self.canvas.draw()

    def reloadButtonCallback(self):
        try:
            self.canvas.get_tk_widget().destroy()
        except:
            TMB.showerror('Error', 'No image is open')
            return
        img = self.img
        h, w = img.shape
        dpi = 100
        hcanvas = h
        wcanvas = w
        self.compressRatio = 1
        user32 = ctypes.windll.user32
        wmax = math.floor(0.92*user32.GetSystemMetrics(0))
        hmax = math.floor(0.92*user32.GetSystemMetrics(1))
        if wcanvas > wmax:
            wcanvas = wmax
            hcanvas = h/w*wcanvas
            self.compressRatio = wmax/w
        if hcanvas > hmax:
            hcanvas = hmax
            wcanvas = w/h*hcanvas
            self.compressRatio = hmax/h
        self.fig = Figure(figsize=(wcanvas/dpi,hcanvas/dpi),dpi=dpi)
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.imshow(img, cmap='gray')
        self.initCanvas()
        self.artistList = []
        if self.mode.get() == 'I':
            self.canvas.mpl_disconnect(self.dID)
            self.canvas.mpl_disconnect(self.tID)
        elif self.mode.get() == 'D':
            self.dID = self.canvas.mpl_connect('pick_event', self.mouseDeleteCallback)
            self.canvas.mpl_disconnect(self.tID)
        elif self.mode.get() == 'T':
            self.tID = self.canvas.mpl_connect('button_press_event', self.mouseTrackPressCallback)
            self.canvas.mpl_disconnect(self.dID)
    def mouseDeleteModeButtonCallback(self):
        self.dID = self.canvas.mpl_connect('pick_event', self.mouseDeleteCallback)
        self.canvas.mpl_disconnect(self.tID)
    def mouseTrackModeButtonCallback(self):
        self.tID = self.canvas.mpl_connect('button_press_event', self.mouseTrackPressCallback)
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
        # pdb.set_trace()
        artist.set_visible(False)
        artist.set_picker(None)
        self.canvas.draw()
        xy = artist.center
        index = self.artistList.index(artist)
        print('Delete an ellipse at (%.1f, %.1f) ...' % (xy[0], xy[1]))      
        deletedDataFrame = self.data.iloc[index].to_frame().transpose()

        try:
            self.deletedData = self.deletedData.append(deletedDataFrame, ignore_index=False)
        except:
            self.deletedData = deletedDataFrame
        self.backwardButton.config(state='normal')
        self.deletedArtist.append(artist)
        self.updateStatus()
    def mouseTrackPressCallback(self, event):
        # print('you pressed', event.button, event.xdata, event.ydata)
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.releaseID = self.canvas.mpl_connect('button_release_event', self.mouseTrackReleaseCallback)      
        
    def mouseTrackReleaseCallback(self, event):
        self.canvas.mpl_disconnect(self.releaseID)
        PPU = self.PPU
        h, w = self.img.shape
        # print('you released', event.button, event.xdata, event.ydata)
        self.x2 = event.xdata
        self.y2 = event.ydata               
        No = -1;
        Area = -1;
        X = (self.x1 + self.x2) / 2 / PPU
        Y = (h - (self.y1 + self.y2) / 2) / PPU
        Major = ((self.x1 - self.x2)**2+(self.y1 - self.y2)**2)**.5 / PPU
        Minor = self.minorAxis/PPU
        Angle = math.atan((self.y1 - self.y2)/(self.x1 - self.x2))
        xy = (X*PPU, h - Y*PPU)
        width = Major*PPU
        height =  Minor*PPU
        angle = Angle/math.pi*180
        elli = mpatches.Ellipse(xy, width, height, angle)
        elli.set_fill(False)
        elli.set_color('red')
        self.ax.add_patch(elli)
        self.canvas.draw()
        print('Add an ellipse at (%.1f, %.1f) ...' % (xy[0], xy[1]))
        if Angle < 0:
            Angle = Angle + math.pi
        Angle = math.degrees(Angle)
        Slice = 1
        data = np.array([[No, Area, X, Y, Major, Minor, Angle, Slice]])
        header = self.data.columns.tolist()
        if len(header) != 8:
            header = ['No', 'Area', 'X', 'Y', 'Major', 'Minor', 'Angle', 'Slice']
            
        # pdb.set_trace()
        addedDataFrame = pd.DataFrame(data=data, columns=header)
        try:
            self.addedData = self.addedData.append(addedDataFrame, ignore_index=True)
        except:  
            self.addedData = addedDataFrame
        self.addedArtist.append(elli)
        self.backwardButton.config(state='normal')
        self.updateStatus()
    def mousePosCallback(self, event):
        h, w = self.img.shape
        self.mousePos = [event.x/self.compressRatio, h - event.y/self.compressRatio]
        self.mousePos = [math.floor(self.mousePos[0]), math.floor(self.mousePos[1])]
        self.mousePosStringVar.set(str(self.mousePos[0]) + ', ' + str(self.mousePos[1]))
    def mergeDataButtonCallback(self):        
        if self.addedData.empty == False:
            self.data = self.data.append(self.addedData, ignore_index=True)
            self.addedData = pd.DataFrame()    
        if self.deletedData.empty == False:
            # if len(self.deletedData)==1
            for index, value in self.deletedData.iterrows():                    
                self.data.drop(index=index, inplace=True)
            self.deletedData = pd.DataFrame()
        # self.mode.set('I')
        # self.modeCallback()
        self.updateStatus()
    def saveDataButtonCallback(self):
        saveName = TFD.asksaveasfilename(initialdir=self.workingDir, title='Select file')
        if saveName != '':
            folder, filename = os.path.split(saveName)
            self.workingDir = folder
            self.data.to_csv(saveName, index=False, sep='\t',float_format='%.3f')
    def modeCallback(self):
        if self.mode.get() == 'I':
            self.canvas.mpl_disconnect(self.dID)
            self.canvas.mpl_disconnect(self.tID)
        elif self.mode.get() == 'D':
            self.dID = self.canvas.mpl_connect('pick_event', self.mouseDeleteCallback)
            self.canvas.mpl_disconnect(self.tID)
        elif self.mode.get() == 'T':
            self.tID = self.canvas.mpl_connect('button_press_event', self.mouseTrackPressCallback)
            self.canvas.mpl_disconnect(self.dID)
            
    def backwardButtonCallback(self):
        if self.mode.get() == 'D':
            # draw ellipse according to the last row of self.deletedArtist
            artist = self.deletedArtist.pop()
            artist.set_visible(True)
            artist.set_picker(True)
            self.canvas.draw()       
            # delete the last row of self.deletedData
            idx = self.deletedData.last_valid_index()
            self.deletedData.drop(axis=0, index=idx, inplace=True)
            # when self.deletedData is empty, set "Backward" button to DISABLED
            if self.deletedData.empty == True:
                self.backwardButton.config(state='disabled')
                # self.deletedData = None
        if self.mode.get() == 'T':
            # Delete ellipse according to the last row of self.deletedData
            artist = self.addedArtist.pop()
            artist.set_visible(False)
            artist.set_picker(None)
            self.canvas.draw()       
            # delete the last row of self.addedData
            idx = self.addedData.last_valid_index()
            # pdb.set_trace()
            self.addedData.drop(axis=0, index=idx, inplace=True)
            # when self.deletedData is empty, set "Backward" button to DISABLED
            if self.addedData.empty == True:
                self.backwardButton.config(state='disabled')
                # self.addedData = None
        self.updateStatus()
    def colorButtonCallback(self):
        if self.colorButtonText.get() == 'Color plot':
            for artist in self.artistList:
                angle = artist.angle
                cohsv = (angle/180, 1, 1)
                corgb = colors.hsv_to_rgb(cohsv)
                artist.set_color(corgb)            
            self.canvas.draw()
            self.colorButtonText.set('Mono plot')
        elif self.colorButtonText.get() == 'Mono plot':
            for artist in self.artistList:
                artist.set_color('green')            
            self.canvas.draw()
            self.colorButtonText.set('Color plot')
    def saveFigButtonCallback(self):
        saveName = TFD.asksaveasfilename(initialdir=self.workingDir, title='Select file', defaultextension='.png', \
                    filetypes = (("jpeg files","*.jpg"), ("PDF", "*.pdf"), ("all files","*.*")))
        if saveName != '':
            folder, filename = os.path.split(saveName)
            self.workingDir = folder
            if filename.endswith('.png'):
                self.fig.savefig(saveName, format='png', dpi=300)
            elif filename.endswith('.pdf'):
                self.fig.savefig(saveName, format='pdf')
            else:
                TMB.showerror('File type error', 'Please input a valid extension (pdf or png)')
    def submitButtonCallback(self):
        self.updateStatus()
        self.PPUStringVar.set('')
        self.minorAxisStringVar.set('')        
    def PPUEnterCallback(self, event):
        self.PPU = float(self.PPUStringVar.get())
        self.updateStatus()
        self.PPUStringVar.set('')
    def MAEnterCallback(self, event):
        self.minorAxis = float(self.minorAxisStringVar.get())
        self.updateStatus()
        self.minorAxisStringVar.set('')
    def testButtonCallback(self):
        pdb.set_trace()
    
        
    """
    Miscellulose
    """
    def updateStatus(self):
        try:
            self.dataStatStringVar.set('Data | ' + str(len(self.data)))
        except:
            self.dataStatStringVar.set('Data | 0')
        try:
            self.deleteTmpStringVar.set('DelTmp | ' + str(len(self.deletedData)))
        except:
            self.deleteTmpStringVar.set('DelTmp | 0')
        try:
            self.addTmpStringVar.set('AddTmp | ' + str(len(self.addedData)))
        except:
            self.addTmpStringVar.set('AddTmp | 0')
        try:
            self.PPULabelStringVar.set('PPU | ' + str(self.PPU))
        except:
            self.PPULabelStringVar.set('PPU | 0')
        try:
            self.minorAxisLabelStringVar.set('Minor axis | ' + str(self.minorAxis))
        except:
            self.minorAxisLabelStringVar.set('Minor axis | 0')
            
if __name__ == '__main__':
    root = tk.Tk(className="manTrack")
    app = mplApp(master=root)
    app.pack()
    root.mainloop()