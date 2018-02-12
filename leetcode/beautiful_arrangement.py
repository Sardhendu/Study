

def beautiful_arrangement(n, k):
    output_arr = []
    i = 0
    j = i+k
    set_cmpr = set()
    for i in range(0,n):
        
        # print ('running i, j :', i, j, i+1, j+1)
        # print (output_arr)
        if i+1 not in set_cmpr:
            output_arr.append(i+1)
            set_cmpr.add(i+1)
            
        if len(output_arr) == n:
            break
            
        # print ('Value i+j ', i+j)
        if i + j <= k and j>0 and j+1 not in set_cmpr:
            output_arr.append(j+1)
            set_cmpr.add(j+1)
            # i += 1
        j -= 1
        # print (output_arr)
        # print('')
        if len(output_arr) == n:
            break
            
    # print (output_arr)
    return output_arr
            
        # i+=1

# print(beautiful_arrangement(n=9999, k=9998))





import cv2
import numpy as np
def resize(image, width= None, height = None, inter=cv2.INTER_AREA):
    dim=None
    (h,w)= image.shape[:2]  # numpy array stores images in (height, width) array, but cv2 uses images in order (width, height) order
    print (h, w)
    if width is None and height is None:  # when no resizing occur
        return image

    if width is None:       # when resized height is passed and width is not then we calculate the aspect ratio of the weidth
        r= height / float(h)
        dim=(int(w * r), height)      # height is the resized hieght
    elif height is None:    # When resized width is passed and hieght is not then we calculate the aspect ratio for the height
        r= width / float(w)
        dim=(width , int(h * r))
    else:                   # when both width and height ratio are provided
        dim=(width, height)

    resized= cv2.resize(image, dim, interpolation=inter)
    # the third argument hold an algorithm in cv2 defined to resize the image
    # we can also use other algorithm like cv2.INTER_LINEAR, cv2.INTER_CUBIC, and cv2.INTER_NEAREST.

    return resized
    
print (resize(image=np.ndarray((5,6))))
# resize(image=np.ndarray((5,6)))