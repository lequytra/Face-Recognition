import numpy

def standardize(img):
    """
    Standardize the given image
    Args:
        img:

    Returns:

    """
    img = img.astype('float32')
    mean, std = img.mean(), img.std()
    return (img - mean)/std
