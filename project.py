#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import glob
import math
import tkinter as tk 
from tkinter import filedialog
import threading
from PIL import Image as Img
from PIL import ImageTk
from tkinter import messagebox


# In[2]:


thumbnail_size=350
def resize_img(img,size):
    if img.width> img.height:
        img_resize= img.resize((size, int(img.height/img.width*size)))
    elif img.width< img.height:
        img_resize= img.resize((int(img.width/img.height*size),size))
    else:
        img_resize= img.resize((size,size))
    return img_resize
def loadFile():
    global img_tk,file_path,img,img_filter,img_1,img_c

    if loadFile_en.get() is None:
        file_path = filedialog.askopenfilename(filetypes = (("JPG files","*.jpg"),("all files","*.*")),
                                              title='請選擇想要的圖片')
        loadFile_en.insert(0,file_path) 
    else:
        file_path = filedialog.askopenfilename(filetypes = (("JPG files","*.jpg"),("all files","*.*")),
                                              title='請選擇想要的圖片')
        loadFile_en.delete(0,'end')
        loadFile_en.insert(0,file_path)
    
    img_122=cv2.imread(file_path)
    img_1=cv2.cvtColor(img_122,cv2.COLOR_BGR2RGB)
    img=Img.open(file_path).convert('RGB') 
    img_filter=None
    img_resize=resize_img(img, thumbnail_size)
    img_tk= ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_tk)
    image_label.place(x=560 ,y=20)
    width, height = img_resize.size
    img_c=np.zeros((width, height))
    img_d=np.zeros((width, height))
def thread_it(func,*args):
    t= threading.Thread(target=func,args=args)
    t.setDaemon(True)
    t.start()

def save_button():
    global img_filter
    
    file_path=filedialog.asksaveasfilename(filetypes = (("JPG files","*.jpg"),("PNG","*.png")),
                                              title='儲存圖片')
    img_filter.save(str(file_path)+'.jpg')
def pop_up():
    messagebox.showinfo("說明書", "'親愛的使用者，請點選左上方「...」按鈕，選擇您想要修改的照片後匯入照片，請務必點選「開始/重新」按鈕，之後您就可以盡情修改照片了!如果您修改完一項後，記得點選「確定」保存此次更改，如果不滿意您可以按下「開始/重新」按鈕，您就可以重新修改圖片!最後等您滿意後，請按「輸出照片」按鈕，您就可以保存您精心修改的圖片了!'")
    


# In[3]:



#旋轉圖片--------------------------------------------------------------
def rotate_img(img1,s_11):
    (h,w,d) = img1.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心
    
    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center,s_11, 1.0)
    
    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img1, M, (w, h))
    return rotate_img
#---------------------------------------------------------------------
#亮度亮度+/-幾%---------------------------------------------------------
def modify_lightness(img1,s_22):
    origin_img = img1

    # 圖像歸一化，且轉換為浮點型
    fImg = img1.astype(np.float32)
    fImg = fImg / 255.0
    
    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    lightness = s_22 # lightness 調整為  "1 +/- 幾 %"

    # 亮度調整
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
    
    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img
#---------------------------------------------------------------------
#對比度+/-幾%------------------------------------------------------------
def modify_saturation(img1,s_33):
    origin_img = img1

    # 圖像歸一化，且轉換為浮點型
    fImg = img1.astype(np.float32)
    fImg = fImg / 255.0
    
    saturation = s_33 # saturation 調整為 "1 +/- 幾 %"
    
    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1
    
    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img
#---------------------------------------------------------------------
#冷暖色調-------------------------------------------------------------
def modify_cold_temperature(img1,s_55):

    # ---------------- 冷色調 ---------------- #  
    
#     height = img.shape[0]
#     width = img.shape[1]
#     dst = np.zeros(img.shape, img.dtype)

    # 1.計算三個通道的平均值，並依照平均值調整色調
    imgB = img1[:, :, 0] 
    imgG = img1[:, :, 1]
    imgR = img1[:, :, 2] 

    # 調整色調請調整這邊~~ 
    # 白平衡 -> 三個值變化相同
    # 冷色調(增加b分量) -> 除了b之外都增加
    # 暖色調(增加r分量) -> 除了r之外都增加
    bAve = cv2.mean(imgB)[0]
    gAve = cv2.mean(imgG)[0]+s_55
    rAve = cv2.mean(imgR)[0]+s_55
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2. 計算各通道增益係數，並使用此係數計算結果
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve
    imgB = np.floor((imgB * bCoef))  # 向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    # 3. 變換後處理
#     for i in range(0, height):
#         for j in range(0, width):
#             imgb = imgB[i, j]
#             imgg = imgG[i, j]
#             imgr = imgR[i, j]
#             if imgb > 255:
#                 imgb = 255
#             if imgg > 255:
#                 imgg = 255
#             if imgr > 255:
#                 imgr = 255
#             dst[i, j] = (imgb, imgg, imgr)

    # 將原文第3部分的演算法做修改版，加快速度
    imgb = imgB
    imgb[imgb > 255] = 255
    
    imgg = imgG
    imgg[imgg > 255] = 255
    
    imgr = imgR
    imgr[imgr > 255] = 255
        
    img1 = np.dstack((imgb, imgg, imgr)).astype(np.uint8) 
            

    return img1
#---------------------------------------------------------------------
#噪點------------------------------------------------------------------
def gaussian_noise(img1, s_55,mean=0):
    sigma=s_55
    # int -> float (標準化)
    img1 = img1 / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img1.shape)
    # noise + 原圖
    gaussian_out = img1 + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    
    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)
    return gaussian_out
#---------------------------------------------------------------------
#對比度----------------------------------------------------------------
def modify_contrast_and_brightness2(img1, s_66,brightness=0):
    contrast=s_66
    brightness=0

    B = brightness / 255.0
    c = contrast / 255.0 
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    img1 = (img1 - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
      
    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img1 = np.clip(img1, 0, 255).astype(np.uint8)

    return img1
#---------------------------------------------------------------------
#銳化------------------------------------------------------------------
def sharpen(img1, s_77): 

    sigma=s_77
    # sigma = 5、15、25
    blur_img = cv2.GaussianBlur(img1, (0, 0), sigma)
    usm = cv2.addWeighted(img1, 1.5, blur_img, -0.5, 0)
    
    return usm  
#---------------------------------------------------------------------
#灰化圖片-------------------------------------------------------------
def grayize(img1):

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # start from BGR -> gray
    return gray
#---------------------------------------------------------------------
#模糊圖片---------------------------------------------------------------
def do_GaussianBlur(img1,s_88):
    kernel_size = s_88
    blur_gray = cv2.GaussianBlur(img1,(kernel_size, kernel_size), 0)
    return blur_gray
#---------------------------------------------------------------------
#邊緣檢測-------------------------------------------------------------
def do_Canny(img1):
    gray_img= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # start from BGR -> gray
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray_img,(kernel_size, kernel_size), 0)
    Canny_img=blur_gray
    low_threshold = 90
    high_threshold = 90
    edges = cv2.Canny(Canny_img, low_threshold, high_threshold)
    return edges
#----------------------------------------------------------------------
#黑白圖片--------------------------------------------------------------
def black_and_white(img1):

    gray= cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) # start from BGR -> gray
    img1=cv2.GaussianBlur(gray,(5,5),0)
    ret,th = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th
#------------------------------------------------------------------------


# In[4]:


def button1(s1):
    global img_filter,img_filter_tk,img_c,img_d,rot
    s_1 = int(s1)
    rot=rotate_img(img_c,s_1)
    img_filter= Img.fromarray(rot.astype('uint8'), mode='RGB')
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk =ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button2(s2):
    global img_filter,img_filter_tk,s_2,img_c,img_d,mod_l
    s_2=int(s2)
    mod_l=modify_lightness(img_c,s_2)
    img_filter=Img.fromarray(mod_l.astype('uint8'), 'RGB')    
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

        
def button3(s3):
    global img_filter,img_filter_tk,img_c,img_d,mod_s
    s_3=int(s3)

    mod_s=modify_saturation(img_c,s_3)
    img_filter=Img.fromarray(mod_s.astype('uint8'), 'RGB')   
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button4(s4):
    global img_filter,img_filter_tk,img_c,img_d,tem
    s_4=int(s4)
    tem=modify_cold_temperature(img_c,s_4)
    img_filter=Img.fromarray(tem.astype('uint8'), 'RGB')  
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button5(s5):
    global img_filter,img_filter_tk,img_c,img_d,noi
    s_5=float(s5)
    noi=gaussian_noise(img_c, s_5,0)
    img_filter=Img.fromarray(noi.astype('uint8'), 'RGB') 
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button6(s6):
    global img_filter,img_filter_tk,img_c,img_d,con
    s_6=int(s6)
    con=modify_contrast_and_brightness2(img_c,s_6,0)
    img_filter=Img.fromarray(con.astype('uint8'), 'RGB')  
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button7(s7):
    global img_filter,img_filter_tk,img_c,img_d,sha
    s_7=int(s7)
    if s_7 % 2==0:
        s_7=s_7+1
    sha=sharpen(img_c, s_7)
    img_filter=Img.fromarray(sha.astype('uint8'), 'RGB') 
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button8(s8):
    global img_filter,img_filter_tk,img_c,gau
    s_8=int(s8)
    if s_8 % 2==0:
        s_8=s_8+1
    gau=do_GaussianBlur(img_c,s_8)
    img_filter= Img.fromarray(gau.astype('uint8'), 'RGB') 
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button_b2():
    global img_filter,img_filter_tk,img_c
    bla=black_and_white(img_c)
    img_filter=Img.fromarray(bla.astype('uint8'), 'L')
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button_b3():
    global img_filter,img_filter_tk,img_c
    can=do_Canny(img_c) 
    img_filter=Img.fromarray(can.astype('uint8'), 'L')
    img_resize=resize_img(img_filter, thumbnail_size)
    img_filter_tk=ImageTk.PhotoImage(img_resize)
    image_label=tk.Label(win,height=350,width=400,bg ='gray94',fg='blue',image=img_filter_tk)
    image_label.place(x=950 ,y=20)

def button_0_0():
    global img_filter,img_filter_tk,img_c
    img_c=img_1
def button_1_1():
    global img_filter,img_filter_tk,img_c
    img_c=rot
def button_2_2():
    global img_filter,img_filter_tk,img_c
    img_c=mod_l
def button_3_3():
    global img_filter,img_filter_tk,img_c
    img_c=mod_s
def button_4_4():
    global img_filter,img_filter_tk,img_c
    img_c=tem
def button_5_5():
    global img_filter,img_filter_tk,img_c
    img_c=noi
def button_6_6():
    global img_filter,img_filter_tk,img_c
    img_c=con
def button_7_7():
    global img_filter,img_filter_tk,img_c
    img_c=sha
def button_8_8():
    global img_filter,img_filter_tk,img_c
    img_c=gau


# In[6]:


win=tk.Tk()
win.title('photo modifier')
win.geometry('1350x600')
win.resizable(False, False)


s_2=0
#Lebel
lb = tk.Label(win,text="請選取檔案",bg ="grey",fg="white",height=1)
lb.place(x=0 ,y=0)
lb1= tk.Label(win,text="原圖:",height=1)
lb1.place(x=750 ,y=0)
lb2= tk.Label(win,text="修改後:",height=1)
lb2.place(x=1125 ,y=0)
lb_s1=tk.Label(win,text="照片旋轉幾度:",height=1)
lb_s2=tk.Label(win,text="亮度+/-幾%:",height=1)
lb_s3=tk.Label(win,text="飽和度+/-幾%:",height=1)
lb_s4=tk.Label(win,text="冷暖色調(+為暖,-為冷):",height=1)
lb_s5=tk.Label(win,text="增加噪點:",height=1)
lb_s6=tk.Label(win,text="對比度:",height=1)
lb_s7=tk.Label(win,text="銳化:",height=1)
lb_s8=tk.Label(win,text="模糊圖片:",height=1)
lb_s1.place(x=0,y=60)
lb_s2.place(x=0,y=120)
lb_s3.place(x=0,y=180)
lb_s4.place(x=0,y=240)
lb_s5.place(x=0,y=300)
lb_s6.place(x=0,y=360)
lb_s7.place(x=0,y=420)
lb_s8.place(x=0,y=480)
#scale
s1=tk.Scale(win,from_=0,to_=360,tickinterval=30,resolution=5,orient='horizontal',length=350,command=button1)
s2=tk.Scale(win,from_=-100,to_=100,tickinterval=30,resolution=1,orient='horizontal',length=350,command=button2)
s3=tk.Scale(win,from_=-100,to_=100,tickinterval=30,resolution=1,orient='horizontal',length=350,command=button3)
s4=tk.Scale(win,from_=-100,to_=100,tickinterval=30,resolution=1,orient='horizontal',length=350,command=button4)
s5=tk.Scale(win,from_=0,to_=1,tickinterval=0.2,resolution=0.1,orient='horizontal',length=350,command=button5)
s6=tk.Scale(win,from_=-200,to_=200,tickinterval=50,resolution=1,orient='horizontal',length=350,command=button6)
s7=tk.Scale(win,from_=1,to_=100,tickinterval=30,resolution=5,orient='horizontal',length=350,command=button7)
s8=tk.Scale(win,from_=0,to_=100,tickinterval=230,resolution=1,orient='horizontal',length=350,command=button8)
s1.place(x=150,y=40)
s2.place(x=150,y=100)
s3.place(x=150,y=160)
s4.place(x=150,y=220)
s5.place(x=150,y=280)
s6.place(x=150,y=340)
s7.place(x=150,y=400)
s8.place(x=150,y=460)
#Entry
loadFile_en = tk.Entry(width=40)
loadFile_en.place(x=100 ,y=0)
#button
button_pop = tk.Button(win, text="說明書",height=3,command= pop_up)
button_pop.place(x=485,y=0)
loadFile_btn = tk.Button(text="...",height=1,command=loadFile)
loadFile_btn.place(x=380 ,y=0)
b2= tk.Button(text="黑白照片",height=3,command=lambda:thread_it(button_b2))
b3= tk.Button(text="邊緣化照片",height=3,command=lambda:thread_it(button_b3))
out=tk.Button(text="輸出照片",height=3,command= lambda:thread_it(save_button))
b0_0=tk.Button(text="開始/重新",height=3,command=lambda:thread_it(button_0_0))
b1_1=tk.Button(text="確定",height=1,command=lambda:thread_it(button_1_1))
b2_2=tk.Button(text="確定",height=1,command=lambda:thread_it(button_2_2))
b3_3=tk.Button(text="確定",height=1,command=lambda:thread_it(button_3_3))
b4_4=tk.Button(text="確定",height=1,command=lambda:thread_it(button_4_4))
b5_5=tk.Button(text="確定",height=1,command=lambda:thread_it(button_5_5))
b6_6=tk.Button(text="確定",height=1,command=lambda:thread_it(button_6_6))
b7_7=tk.Button(text="確定",height=1,command=lambda:thread_it(button_7_7))
b8_8=tk.Button(text="確定",height=1,command=lambda:thread_it(button_8_8))
b2.place(x=100,y=520)
b3.place(x=200,y=520)
out.place(x=300,y=520)
b0_0.place(x=400,y=0)
b1_1.place(x=505,y=60)
b2_2.place(x=505,y=120)
b3_3.place(x=505,y=180)
b4_4.place(x=505,y=240)
b5_5.place(x=505,y=300)
b6_6.place(x=505,y=360)
b7_7.place(x=505,y=420)
b8_8.place(x=505,y=480)
win.mainloop()


# In[ ]:




