import cv2
import numpy as np
from PIL import Image
import jassor.utils as J
from jassor.components import Masking


def main():
    image = Image.open('./cat.jpg')
    image = np.asarray(image)
    edge = process(image, blur_size=21, thresh_min=0., thresh_max=1.)
    human = Masking.get_human(image, onnx_path=rf'D:\jassor\workspace_my_util\jassor_util\resources\modnet_photographic_portrait_matting.onnx')
    # canvas = np.where(human[..., None] < 0.5, image, edge)
    human = human[..., None] / 255
    # canvas = image * human + edge * (1 - human)
    canvas = image * (1 - human) + edge * human
    canvas = canvas.clip(0, 255).astype(np.uint8)
    # mask[mask < 55] = 0
    J.plots([image, edge, human, canvas])


def process(image: np.ndarray, blur_size: int = 9, thresh_min: float = 0, thresh_max: float = 1.):
    # 探测边缘（我也不知道为什么，但总之，它确实可以做到）
    # unify image channels to 3
    if len(image.shape) == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, 0: 3]
    image = image / 255
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), sigmaX=0, sigmaY=0)
    divided = (image + 0.01) / (blurred + 0.01)
    mask = divided
    # mask = np.where(divided < 1, divided, 1 / divided)
    mask = mask.clip(thresh_min, thresh_max)
    mask -= thresh_min
    mask /= thresh_max - thresh_min
    mask = (mask * 255).round().astype(np.uint8)

    return mask


main()
