from PIL import Image
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import signal


def load_image(path):
    """
    Loads image and converts it into grayscale.

    Parameters
    ----------
    path : str
        Path to the image.

    Returns
    -------
    img : array
        Grayscale image.
    """
    img = Image.open(path).convert('L')
    return np.array(img)


def crop2square(img):
    '''
    Crop image to square shape.
    '''
    m, n = img.shape
    crop_size = abs(m-n)

    if m > n:
        img = img[crop_size//2:-crop_size//2, :]
    elif m < n:
        img = img[:, crop_size//2:-crop_size//2]
    return img


def crop2shape(img, shape=(400, 400), crop_from=('both', 'both')):
    '''
    Crop image to shape.

    Parameters
    ----------
    img : array
        Image to be cropped.
    shape : tuple, optional
        Shape of the cropped image.
    crop_from : tuple, optional
        Crop from top/bottom/both and left/right/both.

    Returns
    -------
    img : array
        Cropped image.
    '''
    if crop_from[0] not in ['both', 'top', 'bottom']:
        raise ValueError('crop_from[0] must be one of "both", "top", "bottom"')
    if crop_from[1] not in ['both', 'left', 'right']:
        raise ValueError('crop_from[1] must be one of "both", "left", "right"')

    m, n = img.shape
    m_crop, n_crop = abs(m-shape[0]), abs(n-shape[1])

    if crop_from[0] == 'both':
        img = img[m_crop//2:-m_crop//2, :]
    elif crop_from[0] == 'top':
        img = img[:shape[0], :]
    elif crop_from[0] == 'bottom':
        img = img[-shape[0]:, :]

    if crop_from[1] == 'both':
        img = img[:, n_crop//2:-n_crop//2]
    elif crop_from[1] == 'left':
        img = img[:, :shape[1]]
    elif crop_from[1] == 'right':
        img = img[:, -shape[1]:]

    return img


def gaussian_filter_masks(image_shape, std=10.):
    """
    Creates Gaussian low-pass and high-pass filter masks.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image.
    std : float, optional
        Standard deviation of the Gaussian distribution.

    Returns
    -------
    low_pass_filter : array
        Gaussian low-pass filter mask.
    high_pass_filter : array
        Gaussian high-pass filter mask.
    """
    m, n = image_shape
    gaussian_window = np.outer(  # 2D Gaussian window
        signal.windows.gaussian(m, std),  # 1D Gaussian window
        signal.windows.gaussian(n, std),  # 1D Gaussian window
    )
    low_pass_filter = gaussian_window/gaussian_window.max()  # normalize
    return low_pass_filter, 1-low_pass_filter  # == high_pass_filter


def apply_filter(img, filter):
    """
    Aplies given filter mask in the frequency domain (via FFT).

    Parameters
    ----------
    img : array
        Image to be filtered.
    filter : array
        Filter mask.

    Returns
    -------
    img_filtered : array
        Filtered image.
    """
    img_fft = fftshift(fft2(img))
    img_filtered = ifft2(ifftshift(img_fft*filter))
    return np.real(img_filtered)


def main():
    pass


if __name__ == "__main__":
    main()
