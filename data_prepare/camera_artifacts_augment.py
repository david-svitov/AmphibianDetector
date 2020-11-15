import argparse
from enum import Enum
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from tqdm import tqdm


class DegradationType(Enum):
    NONE = "NONE"
    JPEG = "JPEG"
    GAUSSIAN_NOISE = "GAUSSIAN_NOISE"
    JPEG_AND_GAUSSIAN = "JPEG_AND_GAUSSIAN"


def get_images(folder_name: Path,
               image_type: str = "jpg") -> Generator[Path, None, None]:
    """
    Gets a list of images in folder with selected type
    :param folder_name: Path to folder
    :param image_type: Type of images
    :return: List of images
    """
    images = folder_name.glob(f"*.{image_type}")
    return images


def jpeg_degradation(image: np.ndarray,
                     quality_factor: int = 50) -> np.ndarray:
    """
    Makes JPEG image degradation
    :param image: Input image
    :param quality_factor: Quality factor for JPEG compression algorithm from 1 to 100
    :return: Image with JPEG artifacts after compression
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor]
    _, processed_image = cv2.imencode(".jpg", image, encode_param)
    processed_image = cv2.imdecode(processed_image, 1)
    return processed_image


def noise_degradation(image: np.ndarray,
                      sigma: float = 8) -> np.ndarray:
    """
    Makes Gaussian noise degradation
    :param image: Input image
    :param sigma: Standard deviation.
    :return: Image with additive Gaussian noise
    """
    row, col, ch = image.shape
    mean = 0.8
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    image = image.astype(np.float32)
    #gaussian = cv2.normalize(gaussian, None, alpha=0, beta=1,
    #                         norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    noisy = image * gaussian
    noisy = cv2.normalize(noisy, None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return noisy


def process_image(image: np.ndarray,
                  degradation_type: DegradationType,
                  jpeg_quality_factor: int = 50,
                  gaussian_sigma: float = 8) -> np.ndarray:
    """
    Resize image and add degradations on image.
    :param image: Input image
    :param degradation_type: Type of degradation: JPEG or Gauss or both
    :param jpeg_quality_factor: Quality factor for JPEG compression algorithm from 1 to 100
    :param gaussian_sigma: Standard deviation.
    :return: Image after resize and degradation
    """
    if degradation_type in [DegradationType.GAUSSIAN_NOISE, DegradationType.JPEG_AND_GAUSSIAN]:
        processed_image = noise_degradation(image,
                                            gaussian_sigma)
    else:
        processed_image = image

    if degradation_type in [DegradationType.JPEG, DegradationType.JPEG_AND_GAUSSIAN]:
        processed_image = jpeg_degradation(processed_image,
                                           jpeg_quality_factor)

    return processed_image


def save_image(image: np.ndarray,
               full_image_name: Path,
               dst_folder: Path):
    """
    Save image to selected folder
    :param image: Image for save
    :param full_image_name: Full path to source image
    :param dst_folder: Folder for writing image
    :return: Nothing
    """
    file_name = full_image_name.name
    dst_full_name = dst_folder.joinpath(file_name)

    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(dst_full_name), converted_image)


def main():
    parser = argparse.ArgumentParser(description="Synthetic data degradation with JPEG and Gaussian noise")
    parser.add_argument("input_folder", type=str,
                        help="Folder with input images")
    parser.add_argument("output_folder", type=str,
                        help="Folder with output processed images")
    parser.add_argument("--degradation_type", type=DegradationType, default=DegradationType.JPEG_AND_GAUSSIAN,
                        help="Type of image degradation: [JPEG, GAUSSIAN_NOISE, JPEG_AND_GAUSSIAN]")
    parser.add_argument("--jpeg_quality_factor", type=int, default=100,
                        help="JPEG compression ratio")
    parser.add_argument("--gaussian_sigma", type=float, default=0.3,
                        help="Standard deviation of Gaussian noise")
    args = parser.parse_args()

    np.random.seed(128)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_images = get_images(input_folder)
    for image_name in tqdm(input_images):
        input_image = cv2.imread(str(image_name))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        output_image = process_image(input_image,
                                     degradation_type=args.degradation_type,
                                     jpeg_quality_factor=args.jpeg_quality_factor,
                                     gaussian_sigma=args.gaussian_sigma)
        save_image(output_image, image_name, output_folder)


if __name__ == "__main__":
    main()
