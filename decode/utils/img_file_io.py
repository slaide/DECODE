import skimage
import skimage.io
import numpy

def read_img(path,from_dtype='u12',to_dtype="float32"):
    img_data=skimage.io.imread(str(path), as_gray = True)
    assert from_dtype in ("u8","u12",'u16',"float32")
    assert to_dtype in ("u8","u12",'u16',"float32")

    if from_dtype==to_dtype:
        return img_data

    if to_dtype=="float32":
        if from_dtype=="u12":
            max_scale=float(2**12-1)
        elif from_dtype=="u16":
            max_scale=float(2**16-1)
        else:
            raise Error()

        img_data=img_data.astype(dtype=numpy.float32)/max_scale
        assert img_data.max()<=1.0
    else:
        raise Error()

    return img_data

def write_img(path,img_data:numpy.ndarray,from_dtype:str="float32",as_dtype:str="u16"):
    assert isinstance(img_data,numpy.ndarray)
    assert img_data.dtype in (numpy.uint8,numpy.uint16,numpy.float32)
    assert from_dtype in ("u8","u12","u16","float32")
    assert as_dtype in ("u8","u12","u16","float32")
    
    if from_dtype=="float32":
        assert img_data.dtype==numpy.float32
        if as_dtype!="float32":
            if as_dtype=="u8":
                max_scale_i=2**8-1
                max_scale_f=float(max_scale_i)
                out_dtype=numpy.uint8
            elif as_dtype=="u12":
                max_scale_i=2**12-1
                max_scale_f=float(max_scale_i)
                out_dtype=numpy.uint16
            elif as_dtype=="u16":
                max_scale_i=2**16-1
                max_scale_f=float(max_scale_i)
                out_dtype=numpy.uint16
            else:
                raise Error()

            assert img_data.min()>=0.0 and img_data.max()<=1.0, f"{img_data.min(),img_data.max(),str(path)}"

            img_data*=max_scale_f
            img_data=img_data.astype(dtype=out_dtype)

            assert img_data.max()<=max_scale_i

    elif from_dtype=="u12":
        assert img_data.dtype==numpy.uint16
        if as_dtype=="u8":
            img_data=img_data/numpy.array([2**(12-8)],dtype=numpy.uint16)
            img_data=img_data.astype(dtype=numpy.uint8)
        elif as_dtype=="u12":
            pass
        else:
            raise Error()
    elif from_dtype=="u16":
        assert img_data.dtype==numpy.uint16
        if as_dtype=="u16":
            pass
        elif as_dtype=="u8":
            img_data=img_data/numpy.array([2**(16-8)],dtype=numpy.uint16)
            img_data=img_data.astype(dtype=numpy.uint8)
        elif as_dtype=="u12":
            img_data=img_data/numpy.array([2**(16-12)],dtype=numpy.uint16)
            img_data=img_data.astype(dtype=numpy.uint16)
        else:
            raise Error()
    elif img_data.dtype==numpy.uint8:
        if as_dtype=="u8":
            pass
        else:
            raise Error()
    else:
        raise Error()

    skimage.io.imsave(str(path), img_data, plugin='tifffile', compression = "zlib", check_contrast=False)