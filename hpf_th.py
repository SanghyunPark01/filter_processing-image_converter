import cv2
import numpy as np

n=2000
for number in range(0,n+1):
    filename_o = '/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_kitti/{1:06d}.png'.format(0,number)
    original_img = cv2.imread(filename_o,0)

    #convert img->HPF
    f = np.fft.fft2(original_img)
    fshift = np.fft.fftshift(f)
    rows,cols = original_img.shape
    crow,ccol = (int)(rows/2),(int)(cols/2)
    fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
    f_ishift = np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    dst = np.zeros(original_img.shape[:2], np.uint8)
    dst[img_back > 90] = 255

    #save as ...
    cv2.cvtColor(np.uint8(dst), cv2.COLOR_GRAY2RGB)

    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_hpf_th/{1:06d}.png'.format(0,number), dst)
    print(number)



print("finish")
cv2.waitKey(0)
cv2.destroyAllWindows()
