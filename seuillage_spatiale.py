import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def seuillage_global(image, threshold):
    # histogram
    hist = cv.calcHist([image], [0], None, [256], [0,256])
    c = 1
    idx = np.arange(0,256,1)
    while c !=0:
        # Segment the image
        # G1: pixels with intensities >= T
        G1 = hist[threshold:-1]
        # G2: pixels with intensities <= T
        G2 = hist[0:threshold]
        #Compute averages m1 and m2 for the pixels in G1 and G2
        m1 = np.dot(idx[threshold:-1], G1)/np.sum(G1)
        m2 = np.dot(idx[0:threshold], G2)/np.sum(G2)
        # repeat until T is no longer changing
        c = threshold - int((m1+m2)/2)
        threshold = int((m1+m2)/2)
    # binarization
    seg_image = np.where( image > threshold , np.uint8(255), np.uint8(0))
    return threshold, seg_image

def seuillage_Otsu(vidObj):

    # checks whether frames were extracted
    success = 1
  
    while success:
        # extract frames
        success, image = vidObj.read()
        sec = vidObj.get(cv.CAP_PROP_POS_FRAMES)
        if success:
            # Read as grayscale 
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            ret, thresh = cv.threshold(image, 150, 255, cv.THRESH_OTSU)
            if (sec % 20 == 0):
                cv.imshow('original', image)
                cv.imshow('Otsu Threshold', thresh)
            if cv.waitKey(1) == ord('q'):
                break

    # Closes all the frames
    vidObj.release()
    # De-allocate any associated memory usage         
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows() 


if __name__ == '__main__':
    path = "frames/frame0.jpg"
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    plt.hist(image.ravel(),256,[0,256])
    plt.show()
    threshold = 65
    threshold, seg_image = seuillage_global(image, threshold)
    print('Threshold (seuillage global):', threshold)
    cv.imshow('Seuillage Global frame0', seg_image)
    cv.waitKey(0)
    vidObj = cv.VideoCapture("my_video.mp4")
    seuillage_Otsu(vidObj)