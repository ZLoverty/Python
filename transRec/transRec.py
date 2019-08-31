import pandas as pd
import pdb
import tkinter as tk
import tkinter.filedialog as TFD
import os
from datetime import date

def gen_report(data, year, month):            
    summ = {}
    count = {}
    report = pd.DataFrame()
    report_count = pd.DataFrame()
    data = data.loc[(data.yyyy==year)&(data.mm==month)]
    artistList = data.artist.drop_duplicates()
    for artist in artistList:
        summ['artist'] = artist
        summ['month'] = month
        summ['year'] = year
        count['artist'] = artist + '(次数)'
        count['month'] = month
        count['year'] = year
        subdata = data.loc[data.artist==artist]
        print(subdata)
        for key in subdata.keys():
            if key in ['artist', 'mm', 'dd', 'yyyy']:
                continue
            summ[key] = subdata[key].sum()
            count[key] = len(subdata[key].where(subdata[key]!=0).dropna())
        report1 = pd.DataFrame.from_dict(summ, orient='index').transpose()
        report2 = pd.DataFrame.from_dict(count, orient='index').transpose()
        print(report1)
        report = report.append(report1).append(report2)
    return report
def main():
    root = tk.Tk(className='transRec')
    app = transRec(master=root)
    app.pack()
    root.mainloop()
class Transaction(dict):
    def __init__(self, **kwargs):
        columns = ['artist', 'mm', 'dd', 'yyyy', 'a', 'b',
                                 'c', 'd', 'e', 'f', 'g', 'h', 'i']
        for key in columns:
            self[key] = 0        
    def set_date(self, date_string):
        self['mm'] = int(date_string[0:2])
        self['dd'] = int(date_string[2:4])
        self['yyyy'] = int(date_string[4:8])
    def set_values(self, value_in):
        for key in value_in:
            if key in value_in.keys():
                self[key] = value_in[key]
class transRec(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.create_vars()
        self.create_widget()        
    def create_widget(self):
        frame = tk.Frame(self, width=600, height=300)
        frame.pack()
        frame2 = tk.Frame(frame, width=200, height=300)
        frame2.pack(side='left')
        items = {'artist': ('发型师', 10, '周老师'), 'date': ('日期', 10, date_string())}
        self.vb = {}
        for key in items:
            self.vb[key] = value_box(frame2, boxLabel=items[key][0], entryWidth=items[key][1],
                               default=items[key][2])
            self.vb[key].pack(side='top', padx=10, pady=5)
        items = {'a': '剪发', 'b': '烫染', 'c': '指定剪发', 'd': '指定染发', \
                'e': '洗剪','f': '护发', 'g': '营养', 'h': '卡金', 'i': '外卖'}
        self.vbo = {}
        frame1 = tk.Frame(self, width=600, height=300)
        frame1.pack()
        for key in items:
            self.vbo[key] = value_box_opt(frame1, boxLabel=items[key], default='0')
            self.vbo[key].pack(side='left')
        submitButton = tk.Button(frame1, text='提交交易', command=self.submitTransac)
        submitButton.pack()
        frame3 = tk.Frame(frame, width=400, height=300)
        frame3.pack(side='left')
        saveButton = tk.Button(frame3, text='保存文件', width=6, height=3,
                               command=self.saveFile)
        saveButton.pack(side='left', padx=10, pady=5)
        loadButton = tk.Button(frame3, text='读取文件', width=6, height=3,
                               command=self.loadFile)
        loadButton.pack(side='left', padx=10, pady=5)
        loadButton = tk.Button(frame3, text='生成报表', width=6, height=3,
                               command=self.gen_reportCallback, bg='#e48fff')
        loadButton.pack(side='left', padx=10, pady=5)
        frame4 = tk.Frame(frame, width=400, height=300, bg='#e48fff')
        frame4.pack(side='left', expand=True)
        items = {'year': ('报表年份', 5, '2019'), 'month': ('报表月份', 5, '8')}
        self.vbr = {}
        for key in items:
            self.vbr[key] = value_box(frame4, boxLabel=items[key][0], entryWidth=items[key][1],
                               default=items[key][2], bg='#e48fff')
            self.vbr[key].pack(side='top', padx=10, pady=5)
    def create_vars(self):
        self.workingDir = os.getcwd()
        self.artistList = {'a': 'A', 'b': 'B'}
        self.data = pd.DataFrame()
    def saveFile(self):
        saveName = TFD.asksaveasfilename(initialdir=self.workingDir, title='Select file', defaultextension='.csv')
        if saveName != '':
            folder, filename = os.path.split(saveName)
            self.workingDir = folder
            if os.path.exists(saveName) == False:                
                self.data.to_csv(saveName, index=False, header=True, encoding='utf-8', mode='w')
#                 header=['发型师', '月', '日', '年', '剪发', '烫染', '指定剪发', '指定染发', '洗剪',
#                                         '护发', '营养', '卡金', '外卖']
            else:
                self.data.to_csv(saveName, index=False, header=False, encoding='utf-8', mode='a')
            self.data = pd.DataFrame()
    def loadFile(self):
        fileName = TFD.askopenfilename(initialdir=self.workingDir, title='select')
        if fileName != '':
            folder, filename = os.path.split(fileName)
            self.workingDir = folder
            self.data = pd.read_csv(fileName)
    def submitTransac(self):
        transac = Transaction()
        for key in self.vb:            
            if key == 'date':
                transac.set_date(self.vb[key].get_value())
            else:                
                transac.set_values({key: self.vb[key].get_value()})
        for key in self.vbo:
            if self.vbo[key].get_stat() == 1:
                transac.set_values({key: self.vbo[key].get_value()})
        pdTransac = pd.DataFrame.from_dict(transac, orient='index').transpose()
        self.data = self.data.append(pdTransac)
    def gen_reportCallback(self):       
        year = int(self.vbr['year'].get_value())
        month = int(self.vbr['month'].get_value())
        report = gen_report(self.data, year, month)
#         print(self.data)
#         print(report)
        saveName = TFD.asksaveasfilename(initialdir=self.workingDir, title='Select file', defaultextension='.xlsx')
        if saveName != '':
            folder, filename = os.path.split(saveName)
            self.workingDir = folder
            report.to_excel(saveName, index=False)
class value_box_opt(tk.Frame):
    def __init__(self, master=None, boxLabel='label', entryWidth=5, default='00000000'):
        super().__init__(master)
        self.s = tk.IntVar()
        self.s.trace('w', self.showEntry)
        self.tv = tk.StringVar()
        self.tv.set(default)
        c1 = tk.Checkbutton(self, text=boxLabel, variable=self.s)
        c1.pack()
        self.e = tk.Entry(self, textvariable=self.tv, width=entryWidth)  
    def showEntry(self, *args):
        if self.s.get() == 1:
            self.e.pack()
        else:
            self.e.pack_forget()
    def get_value(self):
        return self.tv.get()
    def get_stat(self):
        return self.s.get()
class value_box(tk.Frame):
    def __init__(self, master=None, boxLabel='label', entryWidth=10, default='0', **kwargs):
        super().__init__(master, kwargs)
        self.s = tk.IntVar()
        self.tv = tk.StringVar()
        self.tv.set(default)
        c1 = tk.Label(self, text=boxLabel)
        c1.pack(side='left')
        self.e = tk.Entry(self, textvariable=self.tv, width=entryWidth)  
        self.e.pack(side='left')
    def get_value(self):
        return self.tv.get()
def date_string():
    yyyy = date.today().year
    dd = date.today().day
    mm = date.today().month
    return ('%02d%02d%04d' % (mm, dd, yyyy))
    
if __name__ == '__main__':
    main()