from math import floor
from typing import Tuple
import os

import click
from faker import Faker
from PIL import Image, ImageDraw, ImageFont


@click.command()
@click.argument("source_dir")
@click.argument("output_dir")
def main(source_dir: str, output_dir: str):
    _process_directory(source_dir, output_dir)


def add_title(source_image: str, title: str, background_color: str = "black"):
    """Adds a title to an image

    Args:
        source_image (str): Path to source image
        title (str): Title to add to image
        background_color (str): Background color for title

    Returns:
        PIL.Image: Image extended to include a title
    """
    orig_img = Image.open(source_image)
    orig_width, orig_height = orig_img.size

    title = _draw_text(title, (orig_width, floor(orig_height * 0.1)), background_color)
    _, title_height = title.size

    image = Image.new("RGBA", (orig_width, orig_height + title_height))
    image.paste(title, (0, 0))
    image.paste(orig_img, (0, title_height))
    return image


def _draw_text(text: str, image_size: Tuple[int, int], background_color: str = "black"):
    """Draws a text image

    Args:
        text (str): Text to draw
        image_size (Tuple[int, int]): Width, height of image to draw text on
        background_color (str): Background color for text

    Returns:
        PIL.Image: An image of text
    """
    image = Image.new("RGBA", image_size, color=background_color)
    point_size = _pixels_to_points(image_size[1])
    # Load font to use
    font = ImageFont.truetype("DejaVuSans.ttf", floor(point_size))
    draw_obj = ImageDraw.Draw(image)
    draw_obj.text((0, 0), text, font=font)
    return image


def _pixels_to_points(pixels):
    """Convert pixels to points assuming 96 dots per inch

    Args:
        pixels (int): Number of pixels to convert to point

    Returns:
        float: Number of points equivalent
    """
    return pixels * 72 / 96


def _gen_title():
    """Generates a title

    Returns:
        generator: A title generator
    """
    locale_list = [
        "ar_EG",
        "ar_PS",
        "ar_SA",
        "bg_BG",
        "bs_BA",
        "cs_CZ",
        "de_DE",
        "dk_DK",
        "el_GR",
        "en_AU",
        "en_CA",
        "en_GB",
        "en_IN",
        "en_NZ",
        "en_US",
        "es_ES",
        "es_MX",
        "et_EE",
        "fa_IR",
        "fi_FI",
        "fr_FR",
        "hi_IN",
        "hr_HR",
        "hu_HU",
        "hy_AM",
        "it_IT",
        "ja_JP",
        "ka_GE",
        "ko_KR",
        "lt_LT",
        "lv_LV",
        "ne_NP",
        "nl_NL",
        "no_NO",
        "pl_PL",
        "pt_BR",
        "pt_PT",
        "ro_RO",
        "ru_RU",
        "sl_SI",
        "sv_SE",
        "tr_TR",
        "uk_UA",
        "zh_CN",
        "zh_TW",
    ]
    fake = Faker(locale_list)
    while True:
        yield fake.text(max_nb_chars=30)


def _process_directory(in_dir, out_dir):
    """Adds random titles to all images in a directory

    Args:
        in_dir (str): Directory of images to add titles to
        out_dir (str): Directory of images with titles added

    Returns:
        None
    """
    def title_img(path):
        fname = os.path.splitext(os.path.basename(path))[0]
        ext = '.png'
        title = next(_gen_title())
        add_title(path, title).save(os.path.join(out_dir, fname + ext))

    os.makedirs(out_dir, exist_ok=True)
    with os.scandir(in_dir) as dir:
        for entry in dir:
            if not entry.name.startswith(".") and entry.is_file():
                title_img(os.path.join(in_dir, entry.name))


if __name__ == "__main__":
    main()
