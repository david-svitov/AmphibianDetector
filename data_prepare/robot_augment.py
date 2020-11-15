import argparse
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from tqdm import tqdm


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


def overlay_transparent(background_img: np.ndarray,
                        img_to_overlay_t: np.ndarray,
                        x: int = 0,
                        y: int = 0) -> np.ndarray:
    """
    Overlays a transparant PNG onto another image using CV2
    :param background_img: The background image
    :param img_to_overlay_t: The transparent image to overlay (has alpha channel)
    :param x: x location to place the top-left corner of our overlay
    :param y: y location to place the top-left corner of our overlay
    :return: Background image with overlay on top
    """
    bg_img = background_img.copy()

    # Extract the alpha mask of the RGBA image, convert to RGB
    r, g, b, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((r, g, b))

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y + h, x:x + w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(a))
    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=a)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

    return bg_img


def process_image(image: np.ndarray,
                  robot_image: np.ndarray,
                  x: int = 0,
                  y: int = 0,
                  scale: float=0) -> np.ndarray:
    """
    Resize image and add degradations on image.
    :param image: Input image
    :param robot_image: image to inpaint into frame
    :param x: x coordinate of robot
    :param y: y coordinate of robot
    :param scale: scale of robot
    :return: Image after resize and degradation
    """
    rescaled_robot = cv2.resize(robot_image, None, fx=scale, fy=scale)
    processed_image = overlay_transparent(image, rescaled_robot, x, y)

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
    parser = argparse.ArgumentParser(description="Insert robot image to frames")
    parser.add_argument("input_folder", type=str,
                        help="Folder with input images")
    parser.add_argument("output_folder", type=str,
                        help="Folder with output processed images")
    parser.add_argument("--robot_image", type=str, default="alesha-robotronic-228x228.png",
                        help="Path to image with robot to insert into frames")
    parser.add_argument("--x", type=int, default=0,
                        help="X coordinate of robot in image")
    parser.add_argument("--y", type=int, default=0,
                        help="Y coordinate of robot in image")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Scale of image with robot")
    args = parser.parse_args()

    np.random.seed(128)

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_images = get_images(input_folder)
    for image_name in tqdm(input_images):
        input_image = cv2.imread(str(image_name))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        robot_image = cv2.imread(args.robot_image, -1)
        robot_image = cv2.cvtColor(robot_image, cv2.COLOR_BGRA2RGBA)

        output_image = process_image(input_image, robot_image,
                                     x = args.x, y = args.y,
                                     scale = args.scale)
        save_image(output_image, image_name, output_folder)


if __name__ == "__main__":
    main()
