from PIL import Image
import numpy as np
import random
import math
from datetime import datetime
import time

def image(file):

    img = Image.open(file)
    #print(img)
    #img.show()

    k = np.array(img)
    k1=np.array(img)
    #print(k)
    return k,k1

def inp():
    f=open("test111.txt","r")
    m=f.read()
    print(len(m))
    return m

def ran_key():
    k = random.randint(2,255)
    print(k)
    return k

def header_prep(length_message, key):
    eight="01111110"+format(length_message,"016b") + format(key,'08b')
    return eight

def encryption(m,key):
    enc=[]
    for i in m:
        nu=ord(i)
        #print(i, nu)
        enc.append(nu^key)
    return enc

def binary_con(s):
    num = ""
    for i in s:
        num += format(i, "08b")
    return num

def rev_convert(s):
    return int("".join(str(i) for i in s), 2)

def embedd(array, num):
   
    l = 0
    l1 = len(num)
    i = 0
    while i < len(array) and l < l1:
        j = 0
        while j < len(array[0]) and l < l1:
            k = 0
            while k < 3 and l < l1:
                r = array[i][j][k] % 2
                if r == 0 and num[l] == '1':
                    array[i][j][k] += 1
                elif r == 1 and num[l] == '0':
                    array[i][j][k] -= 1
                k += 1
                l += 1
            j += 1
        i += 1
    

def extract_header(array, header_size):
    l = 0
    i = 0
    msg = ""
    while i < len(array) and l < header_size :
        j = 0
        while j < len(array[0]) and l < header_size:
            k = 0
            while k < 3 and l < header_size:
                r = array[i][j][k] % 2
                msg += str(r)
                k += 1
                l += 1
            j += 1
        i += 1
    return msg

def merge(exract1):
    new =[]
    for i in range(0, len(extract1), 8):
        b = extract1[i:i+8]
        new.append(rev_convert(b))
    return new

def extract(array, l1):
    l = 0
    i = 0
    msg = ""
    while i < len(array) and l < l1:
        j = 0
        while j < len(array[0]) and l < l1:
            k = 0
            while k < 3 and l < l1:
                r = array[i][j][k] % 2
                msg += str(r)
                k += 1
                l += 1
            j += 1
        i += 1
    return msg

def decryption(enc, key):
    re=''
    for i in enc:
        d = i ^ key
        re=re+chr(d)
    return re


def rme(array1, array):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff = int(array1[i][j][k]) - int(array[i][j][k])
                ss = diff*diff
                s=s+diff
    return  s / (height*width*3)

def rmse(mse):
    root_mean=MSE**0.5
    return root_mean

def psnr(sizemax , mse):
    dupsnr = sizemax ** 2 / mse
    PSNR = 10*math.log(dupsnr,10)
    return PSNR

def ssim(array, array1):
    mux=np.array(array)
    xavg=np.mean(mux)

    muy=np.array(array1)
    yavg=np.mean(muy)
 
    '''npsxy2 = []
    nps = []
    n=len(array)
    m= len(array[0])
    print(n,m)
    for i in range(n):
        sub=[]
        nn=[]
        for j in range(m):
            h=[]
            jj=[]
            for k in range(3):
                h.append(int(array[i][j][k]) *int(array1[i][j][k]))
                jj.append(  (int(array[i][j][k]) *int(array1[i][j][k]))**2)
            sub.append(h)
            nn.append(jj)
        nps.append(sub)
        npsxy2.append(nn)
    xyavg = np.mean(nps)            #E(x^2)
    xyavg2 = np.mean(npsxy2)            #E(x^2)
    varxy = xyavg2 - (xyavg**2)
    stdxy = math.sqrt(varxy)'''
    
    stdx=np.std(array)
    stdy=np.std(array1)
   
    varx=np.var(array)
    vary=np.var(array1)
    lumm = (2*xavg*yavg) / (xavg**2 + yavg**2)
    contrast = (2*stdx*stdy ) / (stdx**2 + stdy**2)
    #stru = (stdxy) / (stdx*stdy)
    #print(lumm, contrast)
    result_value = lumm * contrast;
   
    '''print("sd",stdx,stdy)

    upval = (10.885*xavg*yavg+6.5025 ) * (10.885*xyavg+58.5225)   #c1=10, c2=10 Asummption
    #print("numerator",upval)
    downval = (xavg + yavg+6.5025) * (varx**2 + vary**2+58.5225)
    #print("denominator",downval)
    result_value = upval/downval '''
    return result_value

def ae(array, array1):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff = abs( int(array1[i][j][k]) - int(array[i][j][k]))
                s += diff
            
    return  s / (height*width*3)
 
def ad(array, array1):
    s=0
    width = len(array)
    height = len (array[0])
    for i in range(width):
        for j in range(height):
            for k in range(3):
                diff =  int(array1[i][j][k]) - int(array[i][j][k])
                s += diff
    s= s / (height*width*3)
    return  s
                        
header_size= 32
print("OPERATIONS\n1.EMBEDDING\n2.EXTRACTION")
ch=int(input("ENTER OPERATION:"))

if(ch==1):
    s=inp()
    key = ran_key()
    enc = encryption(s,key)
    
    head=header_prep(len(s), key)
    num=head+binary_con(enc)
    embsize = len(num)

    array1, array = image("obito.jpg")#image input
    width = len(array) ; height = len(array[0]); total = width * height*3
    sizemax=np.max(array1)
    print("IMAGE WIDTH = ",len(array))
    print("IMAGE HEIGHT = ",len(array[0]))
    print("TOTAL EMBEDDING CAPACITY",total)
    if(total>=embsize):
        start = datetime.now()
        emit = embedd(array, num)
        end = datetime.now() 
        immm = Image.fromarray(array)
        immm.save('vasanthganesh.tif')
        emit = (end-start)
        print("TIME ",emit.total_seconds())
        MSE = rme(array1, array)
        print("MSE = ",MSE)
        RMSE = rmse(MSE)
        print("RMSE = ",RMSE)
        PSNR = psnr(sizemax , MSE)
        print("PSNR = ",PSNR)
        SSIM = ssim(array, array1)
        print("SSIM = ",SSIM)
        AE = ae(array, array1)
        print("AE = ",AE)
        AD = ad(array, array1)
        print("AD = ", AD)

    else:
        print("NOPE CANNOT BE EMBEDEDD, PLEASE SELECT NEW IMAGE")
    
    
elif(ch==2):
    inpp= input("ENTER IMAGE NAME:")
    array1, array = image(inpp)
    msg = extract_header(array, header_size)
    #print (msg)
    
    form = rev_convert (msg[:8])
    leng = rev_convert (msg [ 8:24 ])
    key1 = rev_convert (msg[24:])
    #print(form, leng, key1)

    if(form ==126):
        start = datetime.now()
        extract1 = extract(array, leng*8+header_size)
        #print(len(extract1), len(s)*8 + 32)
      
        extract1 = extract1[header_size:]
        new=merge(extract1)
        re = decryption(new,key1)
        end = datetime.now()
        f=open("yasin.txt","w")
        f.write(re)
        f.close()
        #print(len(re), len(s))
        #print(num)
        #print(s)
        print("TIME", end - start)
    else:
        print("NOT EMBEDDED, MESSAGE CANNOT BE EXTRACTED")
