import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def vis_hybrid_image(hybrid_image):
    """
    Visualize a hybrid image by progressively downsampling the image and
    concatenating all of the images together.
    """
    scales = 5
    scale_factor = 0.5
    padding = 5
    original_height = hybrid_image.shape[0]#vertical size
    num_colors = 1 if hybrid_image.ndim == 2 else 3#dimention

    output = np.copy(hybrid_image)
    cur_image = np.copy(hybrid_image)
    for scale in range(2, scales+1):
        # add padding
        output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                            dtype=np.float32)))
        # downsample image
        cur_image = rescale(cur_image, scale_factor, mode='reflect', multichannel=True)
        # pad the top to append to the output
        pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                       num_colors), dtype=np.float32)
        tmp = np.vstack((pad, cur_image))
        output = np.hstack((output, tmp))
    return output

def load_image(path):
    return img_as_float32(io.imread(path))


def save_image(path, im):
    return io.imsave(path, img_as_ubyte(im.copy()))

def image_padding(image,padding_height=1,padding_width=1):
    image_shape = image.shape
    dimension = image_shape.__len__()
    if dimension==3:
        [num, height, width] = image.shape
        padding_array = np.zeros([num, height + 2 * padding_height, width + 2 * padding_width])
        for k in range(num):
            for i in range(height):
                for j in range(width):
                    padding_array[k,i+padding_height,j+padding_width] = image[k,i,j]
    else:
        [height, width] = image.shape
        padding_array = np.zeros([int(height + 2 * padding_height), int(width + 2 * padding_width)])
        for i in range(height):
            for j in range(width):
                padding_array[int(i + padding_height), int(j + padding_width)] = image[i, j]
    return padding_array
if __name__ == "__main__":
    #test_image = np.array([[1,2,3],[1,5,9]],dtype=np.float32)
    test_image = load_image('../data/cat.bmp')
    #a=np.array([test_image[0]])
    #print(a)
    #print(a.shape)
    kernel = np.ones([3,3])
    [height_kernel,width_kernel]=kernel.shape
    a = [0,1,height_kernel-1]
    b = [0,1,width_kernel-1]
    #print test_image[0][a]
    #print test_image[0][a][:,b]
    print (test_image[0][0:3].T[0:3].T)
    print ("image\n",test_image[0])
    #print(image_padding(test_image))


