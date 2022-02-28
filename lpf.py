import cv2
import numpy as np

n=2000
for number in range(0,n+1):
    filename_o = '/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_kitti/{1:06d}.png'.format(0,number)
    img = cv2.imread(filename_o,0)

    avg = cv2.blur(img,(5,5))
    gaussian = cv2.GaussianBlur(img,(5,5),0)
    median = cv2.medianBlur(img,5)
    bilateral = cv2.bilateralFilter(img,9,75,75)

    #save as ...

    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_avg/{1:06d}.png'.format(0,number), avg)
    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_gau/{1:06d}.png'.format(0,number), gaussian)
    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_med/{1:06d}.png'.format(0,number), median)
    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_bil/{1:06d}.png'.format(0,number), bilateral)
    
    print(number)


print("finish")
cv2.waitKey(0)
cv2.destroyAllWindows()
