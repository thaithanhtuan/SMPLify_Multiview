import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d




def check_camera_conversion():
    """ 
        To show 4 images for checking the FOV change and alignments 
        original  COLOR               color2Depth
        depth2color                   original Depth
    """
    c_file = "./input/results-0908/color2depths/color_to_depth0.png"
    c_img = cv2.imread(c_file)[:,:,::-1]

    c_file2 = "./input/results-0908/background/color2depths/color_to_depth0.png"
    c_img2 = cv2.imread(c_file2)[:,:,::-1]
    sub = np.absolute(c_img2 - c_img)

    thres = 20
    mask = np.expand_dims(np.min(sub, axis=-1), axis=-1) > thres
    sub = c_img * mask

    #d2c_file = "color2depth0.png"
    #c2d_img = cv2.imread(c2d_file)
    
    #c2d_file = "color0.png"
    #d_file = "depth0.png"
    #d_img = cv2.imread(d_file, cv2.IMREAD_UNCHANGED) 
    d2c_file = "./input/results-0908/color2depths/depth0.png"
    d2c_img = cv2.imread(d2c_file, cv2.IMREAD_UNCHANGED)

    cv2.imwrite("./input/results-0908/color2depths/newdepth0.png", d2c_img.astype(np.uint16))
    newdepth0 = cv2.imread("./input/results-0908/color2depths/newdepth0.png", cv2.IMREAD_UNCHANGED)
    print(type(newdepth0[0,0]))
    d2c_file2 = "./input/results-0908/background/color2depths/depth0.png"
    d2c_img2 = cv2.imread(d2c_file2, cv2.IMREAD_UNCHANGED)

    image = np.absolute(d2c_img2 - d2c_img)
    thres = 100
    mask = image < thres
    image = d2c_img2 * mask


     
    plt.subplot(3,2,1), plt.imshow(c_img), plt.title('color1')
    plt.subplot(3,2,3), plt.imshow(c_img2), plt.title('color2')
    plt.subplot(3,2,2), plt.imshow(d2c_img), plt.title('d2c_img')
    plt.subplot(3,2,4), plt.imshow(d2c_img2), plt.title('d2c_img2')
    plt.subplot(3,2,5), plt.imshow(sub),  plt.title('colormask')
    plt.subplot(3,2,6), plt.imshow(mask * 255),  plt.title('depthmask')
    plt.show()

if __name__ == "__main__":

    check_camera_conversion()
    
    
