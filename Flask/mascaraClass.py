import os
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.misc
import cv2 as cv

class MaskClass:
    
    def generarMasks(self):
        for x in range(2000):
            mascara=generadorM()
            scipy.misc.imsave('./maskDataset/mask'+str(x)+'.png', mascara)
            
    def detectarMask(self,images):
        sizes=np.array(images).shape
        mascara=np.zeros((sizes[0],sizes[1]))#tamaño mascara
        im=images
        for x in range(sizes[0]):
            for y in range(sizes[1]):
                if im[x,y,0]==0 and im[x,y,1]==0 and im[x,y,2]==0:
                    mascara[x,y]=1

        mascara=np.dstack((mascara, mascara, mascara))
        mascara = cv.morphologyEx(mascara, cv.MORPH_CLOSE, np.ones((5,5),np.uint8))
        mascara = cv.morphologyEx(mascara, cv.MORPH_OPEN, np.ones((5,5),np.uint8))
        mascara = cv.morphologyEx(mascara, cv.MORPH_DILATE, np.ones((5,5),np.uint8))
        return np.array(abs(mascara - 1),dtype=np.uint8),np.array(mascara,dtype=np.uint8)
            
            

def generadorM():
    mascara=np.ones((256,256))#tamaño mascara
    cambioP=70
    for i in range(random.randint(1,5)):#numero de pinceles
        cambioP=70
        x=random.randint(100,230)#inicio
        y=random.randint(100,230)
        grosor=random.randint(3,20)
        direccionX=random.randint(-1,1)
        direccionY=random.randint(-1,1)
        for tam in range(random.randint(50,500)):#numero de pasos
            cambio=random.randint(0,100)
            if cambio>cambioP:
                direccionX=random.randint(-1,1)
                direccionY=random.randint(-1,1)
                cambioP+=1
            x=x+direccionX
            y=y+direccionY
            if x<1 or x>255-grosor :
                x=-direccionX
            if y>255-grosor or y<1:
                 y=-direccionY
            for i in range(x-grosor,x+grosor+1):
                for j in range(y-grosor,y+grosor+1):
                    if (i-x)**2 + (j - y)**2 < (grosor**2):
                        mascara[i,j]=0
            
    return np.dstack((mascara, mascara, mascara))


import unittest
class MaskClassTest(unittest.TestCase):
    
    def test0(self):
        clase =MaskClass()
        #clase.generarMasks()
        path=('./maskDataset/')
        self.assertEqual(len(os.listdir(path)),2000)
        print("----------test0----------")
    
        
    def test1(self):
        m=generadorM()
        self.assertEqual(m.shape,(256,256,3))
        print("----------test3----------")

        
    

    


# In[112]:


if __name__=="__main__":
    unittest.main()
    




