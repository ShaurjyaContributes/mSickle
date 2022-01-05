
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

#capture the image
path_to_img = 'Sickle.jpg'

img = cv2.imread(path_to_img)
#convert to grayscale

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img,cmap = 'gray') 
plt.show()  # display it
#cv2.imshow('Original', img)
'''
ret1, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY) 
ret2, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV) 
ret3, thresh3 = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC) 
ret4, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO) 
ret5, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV) 


thresh6 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
thresh7 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
'''

# Otsu's thresholding
ret1,th1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [
          img, 0, th1,
          blur, 0, th2]
titles = [
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(2):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()

thresh_otsu = threshold_otsu(img)
#filling contours in image holes

des = cv2.bitwise_not(th1)
_,contour,_ = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(des,[cnt],0,255,-1)

img = cv2.bitwise_not(des)

plt.figure()
plt.imshow(img,cmap='gray') 
plt.show()


#cleaning border

#print(type(thresh_otsu))
#print(thresh_otsu)
bw = closing(img > ret1, square(3))

# clearing image border
cleared = bw.copy()
clear_border(cleared)
print(type(ret1))
print(type(cleared))
print(type(img))
print(cleared.shape)
print(img.shape)

plt.figure()
plt.imshow(cleared,cmap='gray') 
plt.show()
clear = np.float32(cleared)
#imgUMat = cv2.cvtColor(clear, cv2.COLOR_BGR2GRAY)
#small object removal

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
#min_size = 150  
min_size = 150
#print(type(output))
#print(output)
print(nb_components)
#print(output.shape)
#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
max_label = 1
max_size = sizes[1]
for i in range(0, nb_components):
    print(sizes[i])
    if sizes[i]>max_size:
        max_label = i
        max_size = sizes[i]
        #img2[output == i + 1] = 255

img2[output == max_label] = 255



plt.figure()
plt.imshow(img2,cmap='gray') 
plt.show()

#cv2.imshow('Image', img) 
    
# De-allocate any associated memory usage   
#if cv2.waitKey(0) & 0xff == 27:  
 #   cv2.destroyAllWindows()  
#labelling
#sorted_contours= sorted(contour, key=cv2.contourArea, reverse= True)



#for (i,c) in enumerate(sorted_contours):
for c in contour: 
    M= cv2.moments(c)
    if M["m00"] != 0:
       cX = int(M["m10"] / M["m00"])
       cY = int(M["m01"] / M["m00"])
    else:
    # set values as what you need in the situation
       cX, cY = 0, 0
    #cx= int(M['m10']/M['m00'])
    #cy= int(M['m01']/M['m00'])
    cv2.putText(img, text= str(i+1), org=(cX,cY),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,255,255),
            thickness=2, lineType=cv2.LINE_AA)

    

plt.imshow(img)
plt.show()


