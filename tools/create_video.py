import cv2
import numpy as np
import glob



if __name__ == '__main__':
    img_array = []
    
    for filename in sorted(glob.glob("tools/output/pcl_*.png")):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('pcl_output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    
    out.release()