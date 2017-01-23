# -*- coding: cp936 -*-  
import os  
path = os.getcwd()  
pic_ext = ['.py','.JPG']  
i = 0  
for file in os.listdir(path):  
    if os.path.isfile(file) == True:
        name,ext = os.path.splitext(file)    
        if ext in pic_ext:  
             i = i+1
             s = "%06d" % i  
             print s
             newname1 = str(s) + '.jpg'  
             os.rename(file,newname1) 