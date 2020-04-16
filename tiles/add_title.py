from math import floor
from typing import Tuple

import click
from PIL import Image, ImageDraw, ImageFont


@click.command()
@click.argument('source_image')
def add_title(source_image: str):
    """Adds a title to an image

    Args:
        source_image (str): Path to source image

    Returns:
        None: Does not return a value
    """
    orig_img = Image.open(source_image)
    orig_width, orig_height = orig_img.size

    title = draw_text("text", (orig_width, floor(orig_height*0.1)))
    _, title_height = title.size

    image = Image.new("RGBA", (orig_width, orig_height + title_height))
    image.paste(title, (0, 0))
    image.paste(orig_img, (0, title_height))
    image.show()

def draw_text(text: str, image_size: Tuple[int, int]):
    """Draws a text image

    Args:
        text (str): Text to draw
        image_size (Tuple[int, int]): Width, height of image to draw text on

    Returns:
        PIL.Image: An image of text
    """
    image = Image.new("RGBA", image_size, color="blue")
    point_size = pixels_to_points(image_size[1])
    # Load font to use
    font = ImageFont.truetype('Arial.ttf', floor(point_size))
    draw_obj = ImageDraw.Draw(image)
    draw_obj.text((0,0), text, font=font)
    return image

def pixels_to_points(pixels):
    """Convert pixels to points assuming 96 dots per inch

    Args:
        pixels (int): Number of pixels to convert to point

    Returns:
        float: Number of points equivalent
    """
    return pixels * 72/96

if __name__ == '__main__':
    add_title()
