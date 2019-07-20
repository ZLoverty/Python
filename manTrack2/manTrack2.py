import skimage.io
import numpy as np
import pandas as pd
import tkinter as tk
import ctypes
import math, cmath
from PIL import Image, ImageTk
import time
import threading
import tkinter.messagebox as TMB
import tkinter.filedialog as TFD
from queue import Queue
import os
import pdb
"""
- Zoom in and out
- Draw ellipse dynamic preview
"""    
class Canvas(tk.Canvas):
    def __init__(self, master, width=600, height=600):
        super().__init__(master, width=width, height=height)
    def create_ellipse(self, cent, major, minor, angle, state='normal'):
        a = major/2
        b = minor/2
        x1 = a
        y1 = 0
        x2 = 0.85*x1
        y2 = b*(1-(x2/a)**2)**0.5
        x3 = 0.4*x1
        y3 = b*(1-(x3/a)**2)**0.5
        x4 = 0
        y4 = b
        xyi = [(x1,y1),(x2,y2),(x3,y3), \
            (x4,y4),(-x3,y3),(-x2,y2), \
            (-x1,-y1),(-x2,-y2),(-x3,-y3), \
            (x4,-y4),(x3,-y3),(x2,-y2)]
        ccent = complex(cent[0], cent[1])
        cangle = cmath.exp(angle*1j)
        xy = []
        for x, y in xyi:
            cxy = cangle * complex(x, y) + ccent
            xy.append((cxy.real, cxy.imag))        
        handle = super().create_polygon(xy, fill='', outline='green', width=1.5, state=state)
        return handle

class ProgressBar(tk.Canvas):
    def __init__(self, master, width=100, height=30):
        super().__init__(master, width=width, height=height)
        self.w = width
        self.h = height
        self.config(bg='red')
    def set_progress(self, progress):
        """
        progress is float (0, 1)
        """
        # Draw canvas according to progress
        try:
            self.delete(self.bar)
        except:
            pass        
        self.bar = self.create_rectangle(0,0,progress*self.w,self.h,fill='black')

class manTrack2(tk.Frame):
    def __init__(self, master):
        super().__init__(master)        
        self.createVars()
        self.createWidget()    
    def createVars(self):
        self.w = 600
        self.h = 600
        self.fps = 30
        self.compress = 1
        self.frameNo = 1
        self.minorAxis = 10
        self.playButtonWidth = 12
        self.isPlaying = 0
        self.workingDir = os.getcwd()
        self.displaypNo = tk.BooleanVar()
        self.displaypNo.trace('w', self.displaypNoTrace)
        self.fpsStringVar = tk.DoubleVar()
        self.frameStringVar = tk.StringVar()
        self.frameStringVar.trace('w', self.displayTrackedFrameCallback)
        self.dataStatStringVar = tk.StringVar()
        self.mousePosStringVar = tk.StringVar()
        self.deleteTmpStringVar = tk.StringVar()
        self.addTmpStringVar = tk.StringVar()
        self.PPULabelStringVar = tk.StringVar()
        self.minorAxisLabelStringVar = tk.StringVar()
        self.photoimgList = []
        self.deletedData = pd.DataFrame()
        self.addedData = pd.DataFrame()
        self.deletedArtist = []
        self.addedArtist = []
        self.daOrderList = []
    def createWidget(self):
        buttonFrame = tk.Frame(self)
        buttonFrame.pack(side='left')
        mainFrame = tk.Frame(self)
        mainFrame.pack(side='left')
        self.canvas = Canvas(mainFrame, width=self.w, height=self.h)
        self.canvas.pack()        
        self.imgpbar = ProgressBar(buttonFrame)
        self.imgpbar.pack(side='top')
        loadImgButton = tk.Button(buttonFrame, text='Load image', command=self.loadImgCallback)
        loadImgButton.pack(side='top')
        loadDataButton = tk.Button(buttonFrame, text='Load data', command=self.loadDataCallback)
        loadDataButton.pack(side='top')
        saveDataButton = tk.Button(buttonFrame, text='Save data', command=self.saveDataCallback)
        saveDataButton.pack(side='top')
        frameEntry = tk.Entry(buttonFrame, textvariable=self.frameStringVar)
        frameEntry.pack(side='top')        
        self.datapbar = ProgressBar(buttonFrame)
        self.datapbar.pack(side='top')        
        # directEditButton = tk.Button(buttonFrame, text='Direct edit', command=self.directEditCallback)
        # directEditButton.pack(side='left')
        mergeButton = tk.Button(buttonFrame, text='Merge', command=self.mergeData)
        mergeButton.pack(side='top')
        backButton = tk.Button(buttonFrame, text='Backward', command=self.backCallback)
        backButton.pack(side='top')       
        self.playButton = tk.Button(buttonFrame, text='>', command=self.playCallback, width=2)
        self.playButton.pack(side='top')
        pNoButton = tk.Button(buttonFrame, text='Particle #', command=self.pNoCallback)
        pNoButton.pack(side='top')
        
        testButton = tk.Button(buttonFrame, text='test', command=self.testCallback)
        testButton.pack(side='top')
        self.scbar = tk.Scrollbar(mainFrame, orient='horizontal', command=self.scbarCallback)
        
        # STATUS FRAME
        statusFrame = tk.Frame(self)
        statusFrame.pack(side='top')
        self.mousePositionLabel = tk.Label(statusFrame, textvariable=self.mousePosStringVar)
        self.mousePositionLabel.pack(fill='x') 
        self.dataStatLabel = tk.Label(statusFrame, textvariable=self.dataStatStringVar)
        self.dataStatLabel.pack(fill='x')
        self.deleteTmpLabel = tk.Label(statusFrame, textvariable=self.deleteTmpStringVar)
        self.deleteTmpLabel.pack(fill='x')
        self.addTmpLabel = tk.Label(statusFrame, textvariable=self.addTmpStringVar)
        self.addTmpLabel.pack(fill='x')
        self.PPULabel = tk.Label(statusFrame, textvariable=self.PPULabelStringVar)
        self.PPULabel.pack(fill='x')
        self.minorAxisLabel = tk.Label(statusFrame, textvariable=self.minorAxisLabelStringVar)
        self.minorAxisLabel.pack(fill='x')
        
    def displaypNoTrace(self, event, a, b):
        # pdb.set_trace()
        if self.displaypNo.get() == False:
            # items = self.item_withtag(('frame-'+str(self.frameNo), 'text'))
            # for item in items:
            self.canvas.itemconfig('all', state='hidden')
        else:
            items = self.item_withtag(('frame-'+str(self.frameNo), 'text'))
            for item in items:
                self.canvas.itemconfig(item, state='normal')
    def pNoCallback(self):
        if self.displaypNo.get() == False:
            self.displaypNo.set(True)
        else:
            self.displaypNo.set(False)
    def loadImg(self):
        def resizeCanvas(wimg, himg, windowSize = .8):
            w = wimg
            h = himg            
            user32 = ctypes.windll.user32
            wmax = math.floor(windowSize*user32.GetSystemMetrics(0))
            hmax = math.floor(windowSize*user32.GetSystemMetrics(1))
            if wimg > wmax:
                w = wmax  
                h = himg/wimg*wmax              
            if h > hmax:  
                h = hmax        
                w = wimg/himg*hmax
            compress = h/himg
            return w, h, compress
        try:
            items = self.canvas.find_withtag('frame')
            for item in items:
                delete(item)
        except:
            pass
        imgDir = TFD.askopenfilename(initialdir=self.workingDir,  title='Select file')
        if imgDir != '':
            folder, filename = os.path.split(imgDir)
            self.workingDir = folder
        else:
            return
        # imgDir = r'F:\Google Drive\Code\Python\Particle-tracking\bacTrack\small.tif'
        # imgDir = r'C:\Users\liuzy\Google Drive\Code\Python\Particle-tracking\bacTrack\small.tif'
        self.imgStack = skimage.io.imread(imgDir)
        himg = self.imgStack.shape[1]
        wimg = self.imgStack.shape[2]
        self.w, self.h, self.compress = resizeCanvas(wimg, himg, windowSize = .88)
        self.canvas.config(width=self.w, height=self.h)
        self.d = len(self.imgStack)
        self.scbarLength = 1/self.d               
        self.scbar.pack(fill='x', side='top')
        self.scbar.set(0, self.scbarLength)
        for num, img in enumerate(self.imgStack):           
            imPIL = Image.fromarray(img)
            imPIL = imPIL.resize((int(self.w), int(self.h)), Image.ANTIALIAS)
            imPho = ImageTk.PhotoImage(imPIL)
            self.photoimgList.append(imPho)
            self.imgpbar.set_progress((num+1)/self.d)
            frame = self.canvas.create_image(0, 0, anchor='nw', image=imPho, state='hidden')
            self.canvas.itemconfig(frame, tag=('frame-' + str(num+1), 'frame'))
        
        
        self.layerCorrection()
    def loadData(self):
        # dataDir = r'C:\Users\liuzy\Google Drive\Code\Python\Particle-tracking\bacTrack\small_data.csv'
        # dataDir = r'F:\Google Drive\Code\Python\Particle-tracking\bacTrack\small_data.csv'
        dataDir = TFD.askopenfilename(initialdir=self.workingDir,  title='Select file', filetypes=(('','*.csv'),('','*.dat')))
        if dataDir != '':
            folder, filename = os.path.split(dataDir)
            self.workingDir = folder
        else:
            return
        self.data = pd.read_csv(dataDir, delimiter=',')
        try:
            self.data.rename(columns={' ': 'No'}, inplace=True)
        except:
            pass
        self.addNo = int(self.data.No.max())
    def scbarCallback(self, *args):
        if len(args) == 2:
            fraction = args[1]
            self.frameNo = math.floor(float(fraction)/self.scbarLength+0.5)+1
            self.frameStringVar.set(str(self.frameNo))
            self.scbar.set((self.frameNo-1)*self.scbarLength, self.frameNo*self.scbarLength)
            self.displayTrackedFrame(self.frameNo)
        else:
            if args[2] == 'pages':
                turnPageNo = math.floor(self.d / 10) + 1
                turnPage = turnPageNo * int(args[1])
                self.frameNo = max(min(self.frameNo + turnPage, self.d), 1)
                self.frameStringVar.set(self.frameNo)
            else:
                turnPageNo = 1
                turnPage = turnPageNo * int(args[1])
                self.frameNo = min(max(self.frameNo + turnPage, 1), self.d)
                self.frameStringVar.set(self.frameNo)            
    def loadImgCallback(self):
        t = threading.Thread(target=self.loadImg)
        t.start()
    def loadDataCallback(self):
        self.loadData()
        t = threading.Thread(target=self.renderData)
        t.start()    
    def renderData(self):
        max = self.data.Slice.max()
        for frame in self.data.drop_duplicates(subset='Slice').Slice:
            data = self.data.where(self.data.Slice==frame).dropna()
            for index, particle in data.iterrows():
                # pdb.set_trace()
                cent = (particle.X*self.compress, particle.Y*self.compress)
                major = particle.Major*self.compress
                minor = particle.Minor*self.compress
                angle = math.pi - math.radians(particle.Angle)
                elli = self.canvas.create_ellipse(cent, major, minor, angle, state='hidden')
                self.canvas.itemconfig(elli, tag=('frame-'+str(int(frame)), 'particle', 'p' + str(int(particle.No))))
                try:
                    text = self.canvas.create_text(cent, text=str(int(particle.particle)), anchor='n', state='hidden', fill='red')
                    self.canvas.itemconfig(text, tag=('frame-'+str(int(frame)), 'text', 'p' + str(int(particle.No))))
                except:
                    pass
            self.datapbar.set_progress(frame/max)
    def displayTrackedFrameCallback(self, event, a, b):
        if self.frameStringVar.get() == '':
            return
        self.frameNo = int(self.frameStringVar.get())
        self.displayTrackedFrame(self.frameNo)
        self.scbar.set((self.frameNo-1)*self.scbarLength, self.frameNo*self.scbarLength)
        self.directEditCallback()
    def displayTrackedFrame(self, frameNo):
        if self.displaypNo.get() == True:
            self.canvas.itemconfig('all', state='hidden')
            for item in self.canvas.find_withtag('frame-'+str(self.frameNo)):
                self.canvas.itemconfig(item, state='normal')
        else:
            self.canvas.itemconfig('all', state='hidden')
            for item in self.canvas.find_withtag('frame-'+str(self.frameNo)):
                if 'text' not in self.canvas.gettags(item):
                    self.canvas.itemconfig(item, state='normal')
        self.canvas.update_idletasks()    
    def item_withtag(self, tags):
        def find_same(list_of_lists):                
            length = len(list_of_lists)
            if length < 2:
                return 0
            elif length == 2:
                tup1 = list_of_lists.pop(0)
                tup2 = list_of_lists.pop(0)
                same = tuple([value for value in tup1 if value in tup2])
                return same
            else:
                tup1 = list_of_lists.pop(0)
                tup2 = list_of_lists.pop(0)
                same = [value for value in tup1 if value in tup2]
                list_of_lists.append(same)
                return find_same(list_of_lists)
        itemList = []           
        for tag in tags:
            listTmp = self.canvas.find_withtag(tag)
            itemList.append(listTmp)
        return find_same(itemList)       
    def mergeData(self):        
        try:
            if self.deletedData.empty == False:
                for index, particle in self.deletedData.iterrows():
                    No = particle.No
                    self.data = self.data.where(self.data.No!=No).dropna()
                    # self.canvas.delete(self.canvas.find_withtag(str(index)))
                for item in self.deletedArtist:
                    self.canvas.delete(item)
                self.deletedArtist = []
                self.deletedData = pd.DataFrame()
            if self.addedData.empty == False:
                self.data = self.data.append(self.addedData, ignore_index=True)
                for index, particle in self.addedData.iterrows():
                    cent = (particle.X*self.compress, particle.Y*self.compress)
                    major = particle.Major*self.compress
                    minor = particle.Minor*self.compress
                    angle = math.pi - math.radians(particle.Angle)
                    elli = self.canvas.create_ellipse(cent, major, minor, angle, state='hidden')
                    self.canvas.itemconfig(elli, tag=('frame-'+str(int(particle.Slice)), 'particle', 'p' + str(int(particle.No))))
                self.addedArtist = []
                self.addedData = pd.DataFrame()
            self.updateStatus()
        except:
            TMB.showerror('','Error merging')
    def deleteCallback(self, event):
        for item in self.canvas.find_overlapping(event.x, event.y, event.x, event.y):
            if 'frame' in self.canvas.gettags(item):
                continue
            self.canvas.itemconfig(item, state='hidden')
            # self.deletedArtist.append(item)
            tags = self.canvas.gettags(item)
            self.deletedArtist.append(item)
            self.daOrderList.append(0)
            for tag in tags:
                if tag.startswith('frame') or tag.startswith('particle') or tag.startswith('current'):
                    continue
                try:
                    No = int(tag[1:])
                    self.deletedData = self.deletedData.append(self.data.where(self.data.No==No).dropna())
                except:
                    pass
        self.updateStatus()
    def addPressCallback(self, event):
        self.x1 = event.x
        self.y1 = event.y
        self.canvas.bind('<ButtonRelease-1>', self.addReleaseCallback)
    def addReleaseCallback(self, event):
        self.addNo += 1
        PPU = self.compress
        self.x2 = event.x
        self.y2 = event.y             
        No = self.addNo;        
        X = (self.x1 + self.x2) / 2 / PPU
        Y = (self.y1 + self.y2) / 2 / PPU
        Major = ((self.x1 - self.x2)**2+(self.y1 - self.y2)**2)**.5 / PPU
        Minor = self.minorAxis
        Area = math.pi*Major*Minor/4
        if self.x1 == self.x2:
            Angle = math.pi/2
            return
        else:            
            Angle = math.atan((self.y1 - self.y2)/(self.x1 - self.x2))
        if Angle < 0:
            Angle = Angle + math.pi
        x = X * PPU
        y = Y * PPU
        major = Major * PPU
        minor = Minor * PPU
        angle = Angle
        elli = self.canvas.create_ellipse((x, y), major, minor, angle, state='normal')
        self.canvas.itemconfig(elli, outline='red')
        print('Add an ellipse at (%.1f, %.1f) ...' % (X, Y))
        self.addedArtist.append(elli)        
        Angle = math.degrees(Angle)
        Slice = self.frameNo
        data = np.array([[No, Area, X, Y, Major, Minor, 180-Angle, Slice]])
        header = self.data.columns.tolist()
        if len(header) != 8:
            header = ['No', 'Area', 'X', 'Y', 'Major', 'Minor', 'Angle', 'Slice']
        addedDataFrame = pd.DataFrame(data=data, columns=header)
        self.addedData = self.addedData.append(addedDataFrame, ignore_index=True)
        self.daOrderList.append(1)
        self.updateStatus()        
    def backCallback(self):
        if len(self.daOrderList) != 0:
            da = self.daOrderList.pop()
            if da == 0:
                item = self.deletedArtist.pop()
                self.canvas.itemconfig(item, state='normal')
                idx = self.deletedData.last_valid_index()
                self.deletedData.drop(axis=0, index=idx, inplace=True)
            else:
                item = self.addedArtist.pop()
                self.canvas.itemconfig(item, state='hidden')
                idx = self.addedData.last_valid_index()
                self.addedData.drop(axis=0, index=idx, inplace=True)
        self.updateStatus()
    def directEditCallback(self):
        for item in self.item_withtag(('frame-' + str(self.frameNo), 'frame')):
            self.canvas.bind('<Button-1>', self.addPressCallback)
        for item in self.item_withtag(('frame-' + str(self.frameNo), 'particle')):
            self.canvas.itemconfig(item, activefill='red')
            self.canvas.tag_bind(item, '<Button-3>', self.deleteCallback)
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
    def getCurrentFrameNo(self):
        return self.frameNo
    def getCurrentImg(self):
        return self.imgStack[self.frameNo-1,:,:]
    def getCurrentData(self):
        return self.data.where(self.data.Slice==self.frameNo).dropna()
    def setData(self, newData):
        newDataFrameNo = newData.drop_duplicates(subset='Slice').Slice.values[0]
        leftover = self.data.where(self.data.Slice!=newDataFrameNo).dropna()
        self.data = leftover.append(newData, ignore_index=True)
    def getData(self):
        return self.data
    def fpsCallback(self, event):
        self.fps = self.fpsStringVar.get()
    def layerCorrection(self):
        for item in self.canvas.find_withtag('frame'):
            self.canvas.tag_lower(item)
    def playCallback(self):
        if self.isPlaying == True:
            self.isPlaying = False
        else:
            self.isPlaying = True
            self.playVideo()
    def playVideo(self):
        if self.isPlaying == True:
            
            # tags = ('frame', 'frame-1')
            # items = self.item_withtag(tags)
            # while self.isPlaying:                
                # self.frameStringVar.set(str(self.frameNo))
                # self.canvas.update_idletasks()  
                # self.frameNo += 1
                # print(self.frameNo)
                
                # self.after(math.floor(1000/self.fps))
            self.frameStringVar.set(str(self.frameNo))
            self.frameNo += 1
            if self.frameNo > self.d:
                    self.frameNo = 1
            self.after(math.floor(1000/self.fps), self.playVideo)
    def saveDataCallback(self):
        saveDir = TFD.asksaveasfilename(initialdir=self.workingDir, \
                            title='Select file', filetypes=(('','*.csv'),('','*.dat')))
        self.data.to_csv(saveDir, float_format='%.2f', sep=',', index=False)
    def testCallback(self):
        pdb.set_trace()   
    
        

if __name__ == '__main__':
    t1 = time.monotonic()
    root = tk.Tk()
    app = manTrack2(root)
    app.pack()    
    root.mainloop()
    t2 = time.monotonic()
    print(t2-t1)