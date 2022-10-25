import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


def seuillage_basic(vidObj):
    # checks whether frames were extracted
    success = 1
    count = 0
    # step of I(t-N) 
    N = 1
    directory = "segmented frames"
    if not os.path.isdir(directory):
        os.mkdir(directory)
  
    while success:
        # extract frames
        success, image = vidObj.read()
        if success:
            # Read as grayscale 
            image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            # Frame 0
            if count == 0:
                B = image
                _, d = cv.threshold(B, 127, 255, cv.THRESH_TRIANGLE)
                cv.imwrite(directory+"\\frame%d.jpg" % count, d)
                count += 1
                continue
            # Step
            if count % N != 0:
                count += 1
                continue
            # we got the segmented frames flipped upside down
            image = cv.rotate(image, cv.ROTATE_180)

            # Basic background substraction
            M = cv.absdiff(B, image)
            B = image
            _, diff = cv.threshold(M, 150, 255, cv.THRESH_TRIANGLE)
            #diff = cv.adaptiveThreshold(M, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 2)

            
            cv.imshow('original', image)
            cv.imshow('seg_temp', diff)
            cv.imwrite(directory+"\\frame%d.jpg" % count, diff)
            count+=1

            if cv.waitKey(1) == ord('q'):
                break

def seuillage_3_frame(vidObj):
    # checks whether frames were extracted
    success = 1
    count = 1
    # step of I(t-N) 
    N = 20
    directory = "segmented-3-frames"
    if not os.path.isdir(directory):
        os.mkdir(directory)
  
    while success:
        # extract frames
        success, image = vidObj.read()
        if success:
            if count%N == 0 and count+N < vidObj.get(cv.CAP_PROP_FRAME_COUNT):
                # where object was
                B1 = cv.imread("segmented frames"+"\\frame%d.jpg" % (count-N))
                # where object will be
                B2 = cv.imread("segmented frames"+"\\frame%d.jpg" % (count+N))
                # where object is now
                image = cv.imread("segmented frames"+"\\frame%d.jpg" % count)
                # where object was and where it is now
                D1 = cv.absdiff(image, B1) 
                # where object is now and where it will be
                D2 = cv.absdiff(image, B2)
                # AND: where object is now
                D = cv.bitwise_and(D1, D2)

                # show frame
                cv.imshow('seg_temp', D)
                cv.imwrite(directory+"\\frame%d.jpg" % count, D)

            count+=1
            if cv.waitKey(1) == ord('q'):
                break

def adaptive_bg_substruction(video):
    N = 2
    T = 96
    c = 0
    alpha = 0.3
    B = []
    directory = "segmented frames"
    if not os.path.isdir(directory):
        os.mkdir(directory)

    while video.isOpened():
        success, frame = video.read()
        if (c % N) == 0:
            if success == True:
                frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
                if c == 0:
                    B = frame
                    c += 1
                    continue
                M = cv.absdiff(frame, B)
                #addition d'image avec poids
                B = cv.addWeighted(frame, 1-alpha, B, alpha, 0.0)
                T, D = cv.threshold(M,100,255,cv.THRESH_TRIANGLE)
                
                # show frame
                cv.imshow('original', frame)
                cv.imshow('seg_temp', D)
                cv.imwrite(directory+"\\frame%d.jpg" % c, D)

                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        c += 1


def backgroundSubstractor(fgbg, vidObj):
    directory = "segmented frames"
    if not os.path.isdir(directory):
        os.mkdir(directory)

    c=0
    success = 1
        
    while(success):
        success, frame = vidObj.read()
        # we got the segmented frames flipped upside down
        frame = cv.rotate(frame, cv.ROTATE_180)
        fgmask = fgbg.apply(frame)
        cv.imshow('seg_temp',fgmask)
        cv.imwrite(directory+"\\frame%d.jpg" % c, fgmask)
        c+=1
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break


if __name__ == '__main__':

    vidObj = cv.VideoCapture('my_video.mp4')

    #seuillage_basic(vidObj)
    seuillage_3_frame(vidObj)
    #adaptive_bg_substruction(vidObj)

    #fgbg = cv.createBackgroundSubtractorMOG2()
    #backgroundSubstractor(fgbg, vidObj)
    #fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
    #backgroundSubstractor(fgbg, vidObj)

    # Closes all the frames
    vidObj.release()
    # De-allocate any associated memory usage         
    if cv.waitKey(0) & 0xff == 27:
        cv.destroyAllWindows() 
