import skimage
import skimage.io

def read_img(path):
    img_data=skimage.io.imread(str(path), as_gray = True)
    img_data=skimage.img_as_ubyte(img_data)
    return img_data
def write_img(path,img_data):
    img_data=skimage.img_as_ubyte(img_data)
    skimage.io.imsave(str(path), img_data, plugin='tifffile', compress = 6, check_contrast=False)