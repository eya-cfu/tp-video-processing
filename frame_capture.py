import cv2
import os

# Function to extract frames
def frameCapture(path):
    
    # Make a folder for the video frames
    directory = "frames"
    if not os.path.isdir(directory):
        os.mkdir(directory)
    # Read video file
    vidObj = cv2.VideoCapture(path)
    width = vidObj.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidObj.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vidObj.get(cv2.CAP_PROP_FPS)
    frame_count = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
    props = {"width": width, "height": height, "fps": fps, "frame_count": frame_count}
    print(props)
    
    count = 0
    # checks whether frames were extracted
    success = 1
  
    while success:
        # extract frames
        success, image = vidObj.read()
        sec = vidObj.get(cv2.CAP_PROP_POS_FRAMES)
        if success:
            # Save the frames with frame-count in the video folder
            cv2.imwrite(directory+"\\frame%d.jpg" % count, image)
            # Show images every 50ms
            if (sec % 50 == 0):
                cv2.imshow('frame', image)
            count += 1
            if cv2.waitKey(1) == ord('q'):
                break

    # Closes all the frames
    cv2.destroyAllWindows()
    vidObj.release()


# Driver Code
if __name__ == '__main__':
  
    # Calling the function
    frameCapture("D:\\E480\\Documents\\insat\\RT5\\sem1\\analyse video\\TP\\tp1\\my_video.mp4")