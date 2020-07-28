#coding: utf8
import numpy as np
import os
from helpers import load_image, save_image
import matplotlib.pyplot as plt
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale
from helpers import image_padding
from helpers import vis_hybrid_image


def imfilter_onecolor(image, kernel):
    filtered_image = np.zeros(image.shape)
    [height_kernel, width_kernel] = kernel.shape
    if height_kernel % 2 == 0:
        raise ValueError("kernel has even height")
    if width_kernel % 2 == 0:
        raise ValueError("kernel has even width")


    image = image_padding(image, (height_kernel - 1) / 2, (width_kernel - 1) / 2)
    [height, width] = filtered_image.shape
    for j in range(height):
        for k in range(width):
            filtered_image[j, k] = np.sum(np.multiply(image[j:j + height_kernel].T[k:k + width_kernel].T, kernel))
    return filtered_image

def convolution_onecolor(image, kernel):
    filtered_image = np.zeros(image.shape)
    [height_kernel, width_kernel] = kernel.shape
    if height_kernel % 2 == 0:
        raise ValueError("kernel has even height")
    if width_kernel % 2 == 0:
        raise ValueError("kernel has even width")

    kernel_temp = kernel.reshape(kernel.size)
    kernel_temp = kernel_temp[::-1]
    kernel = kernel_temp.reshape(kernel.shape)
    image = image_padding(image, (height_kernel - 1) / 2, (width_kernel - 1) / 2)
    [height, width] = filtered_image.shape
    for j in range(height):
        for k in range(width):
            filtered_image[j, k] = np.sum(np.multiply(image[j:j + height_kernel].T[k:k + width_kernel].T, kernel))
    return filtered_image

def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #t
    if filtered_image.shape.__len__()==2:
        filtered_image = imfilter_onecolor(image, kernel)
    else:
        test_image_red = image[:, :, 0]
        test_image_green = image[:, :, 1]
        test_image_blue = image[:, :, 2]
        filtered_image_red = imfilter_onecolor(test_image_red, kernel)
        filtered_image_green = imfilter_onecolor(test_image_green, kernel)
        filtered_image_blue = imfilter_onecolor(test_image_blue, kernel)
        filtered_image[:, :, 0] = filtered_image_red
        filtered_image[:, :, 1] = filtered_image_green
        filtered_image[:, :, 2] = filtered_image_blue
    #print('my_imfilter function in student.py needs to be implemented')
    ##################

    return filtered_image

"""
EXTRA CREDIT placeholder function
"""

def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    if filtered_image.shape.__len__()==2:
        filtered_image = convolution_onecolor(image, kernel)
    else:
        test_image_red = image[:, :, 0]
        test_image_green = image[:, :, 1]
        test_image_blue = image[:, :, 2]
        filtered_image_red = convolution_onecolor(test_image_red, kernel)
        filtered_image_green = convolution_onecolor(test_image_green, kernel)
        filtered_image_blue = convolution_onecolor(test_image_blue, kernel)
        filtered_image[:, :, 0] = filtered_image_red
        filtered_image[:, :, 1] = filtered_image_green
        filtered_image[:, :, 2] = filtered_image_blue
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    kernel = kernel / np.sum(kernel)
    low_frequencies = my_imfilter_fft(image1,kernel)
    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #

    high_frequencies = image2 - my_imfilter_fft(image2,kernel)
    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = high_frequencies+low_frequencies
    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.

    return low_frequencies, high_frequencies, hybrid_image




if __name__=="__main__":
    resultsDir = '..' + os.sep + 'results'
    if not os.path.exists(resultsDir):
        os.mkdir(resultsDir)

    test_image = load_image('../data/fish.bmp')
    test_image2 = io.imread('../data/submarine.bmp')
    test_image = rescale(test_image, 0.7, mode='reflect', multichannel=True)
    test_image2 = rescale(test_image2, 0.7, mode='reflect', multichannel=True)
    low_frequencies, high_frequencies, hybrid_image = gen_hybrid_image(test_image,test_image2,3)
    #plt.imshow(low_frequencies)
    plt.imshow((low_frequencies * 255).astype(np.uint8))
    plt.show()
    plt.imshow(((high_frequencies+0.5)*255).astype(np.uint8))
    plt.show()
    plt.imshow((hybrid_image*255).astype(np.uint8))
    plt.show()
    # high_frequencies = test_image2 - gaussian_blur(test_image2,3)
    # plt.imshow(high_frequencies+0.5)
    # plt.show()


