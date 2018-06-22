# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:34:05 2018

To get the horizonal lines of images in the Viewnyx dataset

@author: Wen Wen
"""
import wx
import os
import json
import matplotlib.pyplot as plt
import scipy.misc as scimisc


class MyCanvas(wx.ScrolledWindow):
    def __init__(self, parent, id = -1, size = wx.DefaultSize, filepath = None):
        wx.ScrolledWindow.__init__(self, parent, id, (0, 0), size=size, style=wx.SUNKEN_BORDER)

        self.image = wx.Image(filepath)
        self.w = self.image.GetWidth()
        self.h = self.image.GetHeight()
        self.bmp = wx.BitmapFromImage(self.image)
        self.pos_y=0

        self.SetVirtualSize((self.w, self.h))
        self.SetScrollRate(20,20)
        self.SetBackgroundColour(wx.Colour(0,0,0))

        self.buffer = wx.EmptyBitmap(self.w, self.h)
        dc = wx.BufferedDC(None, self.buffer)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        self.DoDrawing(dc)

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_UP, self.OnClick)

    def OnClick(self, event):
        pos = self.CalcUnscrolledPosition(event.GetPosition())
        print('x=',pos.x,' y=', pos.y)        
        self.pos_y=pos.y
        
    def OnPaint(self, event):
        # device content above image
        dc = wx.BufferedPaintDC(self, self.buffer, wx.BUFFER_VIRTUAL_AREA)

    def DoDrawing(self, dc):
        dc.DrawBitmap(self.bmp, 0, 0)

class MyFrame(wx.Frame): 
    def __init__(self, parent=None, id=-1, filepath = None): 
        wx.Frame.__init__(self, parent, id, title=filepath)
        self.canvas = MyCanvas(self, -1, filepath = filepath)

        self.canvas.SetMinSize((self.canvas.w, self.canvas.h))
        self.canvas.SetMaxSize((self.canvas.w, self.canvas.h))
        self.canvas.SetBackgroundColour(wx.Colour(0, 0, 0))
        vert = wx.BoxSizer(wx.VERTICAL)
        horz = wx.BoxSizer(wx.HORIZONTAL)
        vert.Add(horz,0, wx.EXPAND,0)
        vert.Add(self.canvas,1,wx.EXPAND,0)
        self.SetSizer(vert)
        vert.Fit(self)
        self.Layout()

def GetJPGDict(filepath):
    jpglist=os.listdir(filepath) 
    jpgdict={}
    count=0
    for i in jpglist:
        if 'jpg' in i and 'o' not in i: # exclude the horizon and bottom img
            jpgdict[count]=i
            count+=1
    return jpgdict

def DrawLineNSave(filepath,filename,pos_y):
    img=scimisc.imread(filepath+filename)
    for i in range(0,640):
       img[pos_y][i][0]=255
       img[pos_y][i][1:2]=0
    scimisc.imsave(filepath+filename.split('.')[0]+'_horizon.jpg',img)

if __name__ == '__main__':
    app = wx.App()
    app.SetOutputWindowAttributes(title='stdout')  
    wx.InitAllImageHandlers()
    
    imgpath = 'Frame Images/FRAMES/'
    jpgdict=GetJPGDict(imgpath)
    horizondict={}
    
    ''' if annotate it by line regression '''
    '''
    myframe = MyFrame(filepath=imgpath+jpgdict[0])
    myframe.Center()
    myframe.Show()
    app.MainLoop()
    y0=myframe.canvas.pos_y
    
    myframe = MyFrame(filepath=imgpath+jpgdict[len(jpgdict)-1])
    myframe.Center()
    myframe.Show()
    app.MainLoop()
    y1=myframe.canvas.pos_y
    
    for i in jpgdict:
        pos_y=int(y0+float(y1-y0)/(len(jpgdict)-1)*i)
        horizondict[jpgdict[i]]=pos_y
        DrawLineNSave(imgpath,jpgdict[i],pos_y)
    '''
    ''' line regression code ended '''
    
    
    ''' if annotate it mannually'''
    
    for i in jpgdict:
        #if i>5:  break # for debug
        #print(imgpath+jpgdict[i]+' done saving horizon line')
        myframe = MyFrame(filepath=imgpath+jpgdict[i])
        myframe.Center()
        myframe.Show()
        app.MainLoop()
        # save the horizon line of current image
        horizondict[jpgdict[i]]=myframe.canvas.pos_y
        DrawLineNSave(imgpath,jpgdict[i],myframe.canvas.pos_y)
    
    ''' mannually annotation code ended '''
    
    chart=[]
    for i in horizondict:
        chart.append(horizondict[i])
    plt.plot(chart)
    
    with open(imgpath+'horizon.json', 'w') as outfile:
        json.dump(horizondict, outfile)
   
    
     
''' End of File '''