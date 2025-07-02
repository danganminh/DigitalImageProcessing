import numpy as np
from skimage.transform import rescale

"""
    Nguyễn Minh Đăng

    Almost function using with image gray scale
    If you want using with RGB image, please process one by one for ndim
    Example:
    img_rgb = plt.imread("aaa.png")
    for i in range(3):
        img_rgb[:,:,i] = function(img_rgb[:,:,i])
        
    Variables:
    img: is image gray scale
    img_rgb: is image RGB with ndim = 3
"""


def rescale_sk(img: np.array, scale: float, anti_aliasing: bool = True) -> np.array:
    """_summary_

    Args:
        img (np.array): _description_
        scale (float): _description_
        anti_aliasing (bool, optional): _description_. Defaults to True.

        scale = (img.shape[0] / factor) / img.shape[0]
    Returns:
        np.array: _description_
    """
    return rescale(img, scale, anti_aliasing=anti_aliasing)


def gray_lightness(img_rbg: np.array) -> np.array:
    RGB = np.array([img_rbg[:, :, i] for i in range(3)])
    return np.add(np.min(RGB, axis=0), np.max(RGB, axis=0)) / 2


def gray_average(img_rbg: np.array) -> np.array:
    RGB = np.array([img_rbg[:, :, i] for i in range(3)])
    return np.mean(RGB, axis=0)


def gray_luminosity(img_rbg: np.array) -> np.array:
    RGB = np.array([img_rbg[:, :, i] for i in range(3)])
    return 0.3 * RGB[0] + 0.59 * RGB[1] + 0.11 * RGB[2]


def zero_padding(img: np.array, nx: int, ny: int) -> np.array:
    return np.pad(img, ((nx, nx), (ny, ny)), "constant", constant_values=0)


def mirror_padding(img: np.array, nx: int, ny: int) -> np.array:
    return np.pad(img, ((nx, nx), (ny, ny)), "reflect")


def back_white(img: np.array) -> np.array:
    """_summary_

    Args:
        img (np.array): image gray scale

    Returns:
        np.array: return a black and white image with dtype = uint8
    """
    return np.round(img / np.max(img)).astype(np.uint8)


def image_translate(img: np.array, tx: int, ty: int) -> np.array:
    if tx == 0 and ty == 0:
        return img.copy()

    height, width = img.shape
    new_height = height + abs(tx)
    new_width = width + abs(ty)
    result = np.zeros([new_height, new_width])

    start_x = 0 if tx < 0 else tx
    end_x = height if tx < 0 else new_height
    start_y = 0 if ty < 0 else ty
    end_y = width if ty < 0 else new_width

    result[start_x:end_x, start_y:end_y] = img.copy()
    return result


def image_rotate(img: np.array, theta: float) -> np.array:
    """
    If you want run faster, use skimage
    Example:
        from skimage.transform import rotate
        image_rotate = rotate(img, theta)

    Args:
        img (np.array): image
        theta (float): degree

    Returns:
        np.array: output
    """
    angle = np.radians(theta)
    height, width = img.shape

    new_height = (
        np.abs(height * np.cos(angle)) + np.abs(width * np.sin(angle))
    ).astype(np.int32)

    new_width = (np.abs(height * np.sin(angle)) + np.abs(width * np.cos(angle))).astype(
        np.int32
    )

    output = np.zeros([new_height, new_width])
    cx = height // 2
    cy = width // 2
    center_x = new_height // 2
    center_y = new_width // 2

    for i in range(new_height):
        for j in range(new_width):
            x = (i - center_x) * np.cos(angle) - (j - center_y) * np.sin(angle)
            y = (i - center_x) * np.sin(angle) + (j - center_y) * np.cos(angle)
            x = round(x) + cx
            y = round(y) + cy
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]

    return output


def image_shear(img: np.array, sv: float, sh: float) -> np.array:
    """
    If you want run faster, use skimage
    Example:
        from skimage import transform as tf

        afine_tf = tf.AffineTransform(shear=(sv,sh))
        img_shear_tf = tf.warp(img, inverse_map=afine_tf)

    Args:
        img (np.array): image
        sv (float): _description_
        sh (float): _description_

    Returns:
        np.array: output
    """
    height, width = img.shape
    add_height = np.abs(height * sv).astype(np.int32)
    add_weight = np.abs(width * sh).astype(np.int32)
    new_height = height + add_height
    new_width = width + add_weight
    output = np.zeros([new_height, new_width])
    for i in range(new_height):
        for j in range(new_width):
            x = round(i - sv * j)
            y = round(j - sh * i)
            if 0 <= x < height and 0 <= y < width:
                output[i, j] = img[x, y]
    return output[
        add_height // 2 : height + add_height // 2,
        add_weight // 2 : width + add_weight // 2,
    ]
