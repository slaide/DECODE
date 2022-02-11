import skimage
import skimage.io
import numpy

def read_img(path,dtype='u16'):
    img_data=skimage.io.imread(str(path), as_gray = True)
    assert dtype in ('u16',"float","float32")
    if dtype in ("float","float32"):
        img_data=img_data.astype(dtype=numpy.float32)/4095.0
    return img_data
def write_img(path,img_data:numpy.ndarray):
    assert isinstance(img_data,numpy.ndarray)
    assert img_data.dtype in (numpy.uint16,numpy.float32)
    
    if img_data.dtype==numpy.float32:
        assert img_data.max()<1.0 and img_data.min()>0.0
        img_data*=4095.0
        img_data=img_data.astype(dtype=numpy.uint16)

    assert img_data.max()<4096
    skimage.io.imsave(str(path), img_data, plugin='tifffile', compress = 6, check_contrast=False)