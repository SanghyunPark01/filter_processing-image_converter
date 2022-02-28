import numpy as np
import cv2
n=2000
for number in range(0,n+1):
    filename_o = '/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_kitti/{1:06d}.png'.format(0,number)
    img = cv2.imread(filename_o)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    phase_spectrum = np.angle(dft_shift)

    normalized=phase_spectrum/phase_spectrum.max()*255.0
    cv2.imwrite('/home/psh0302/img_preprocessing/dataset/sequence/{0:02d}/image_phaseonly/{1:06d}.png'.format(0,number), normalized)
    print(number)

print("finish")
cv2.waitKey(0)
cv2.destroyAllWindows()
