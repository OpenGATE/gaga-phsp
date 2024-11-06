#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager
import os


def find_font_path(font_name):
    """Search common font directories for the specified font."""
    common_font_dirs = [
        "/Library/Fonts/",
        "/System/Library/Fonts/",
        os.path.expanduser("~/Library/Fonts/"),  # macOS
        "C:/Windows/Fonts/",  # Windows
        "/usr/share/fonts/",
        os.path.expanduser("~/.fonts/"),  # Linux
    ]

    for font_dir in common_font_dirs:
        if os.path.isdir(font_dir):
            for font_file in os.listdir(font_dir):
                if font_name.lower() in font_file.lower():
                    return os.path.join(font_dir, font_file)

    raise FileNotFoundError(f"Font '{font_name}' not found in common directories.")


def get_character_image(character="S", font_size=200):
    # Create a blank image with enough space for the character
    image_size = (font_size, font_size)
    image = Image.new("L", image_size, 0)  # 'L' mode for greyscale

    # Draw the character onto the image
    draw = ImageDraw.Draw(image)
    font_path = find_font_path("Arial")
    print(font_path)
    font = ImageFont.truetype(font_path, 100)

    # Calculate the text bounding box and center the character
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_position = (
        (image_size[0] - text_width) // 2,
        (image_size[1] - text_height) // 2,
    )

    # Draw the character onto the image
    draw.text(text_position, character, fill=255, font=font)

    image.save("toto.png", format="PNG")

    # Convert image to numpy array for pixel value access
    img_array = np.array(image)
    return img_array


def evaluate(points, character="S", font_size=200):
    img_array = get_character_image(character, font_size)
    points[:, 1] = font_size - points[:, 1]  # Flip y-axis
    x_rand = points[:, 0]
    y_rand = points[:, 1]
    x_rand[x_rand < 0] = 0
    x_rand[x_rand >= font_size] = font_size - 1
    y_rand[y_rand < 0] = 0
    y_rand[y_rand >= font_size] = font_size - 1
    mask = img_array[y_rand.astype(int), x_rand.astype(int)] > 0
    points[:, 1] = font_size - points[:, 1]  # Flip y-axis
    return np.sum(mask)


def generate_continuous_character_points_rejection_sampling(
    character="S", num_points=500, font_size=200
):
    # Convert image to numpy array for pixel value access
    img_array = get_character_image(character, font_size)

    # Initialize list to hold sampled points
    sampled_points = []

    # Generate random points in batches for faster rejection sampling
    batch_size = num_points * 5  # Generating more points per batch increases speed
    while len(sampled_points) < num_points:
        # Generate random continuous coordinates
        x_rand = np.random.uniform(0, font_size, batch_size)
        y_rand = np.random.uniform(0, font_size, batch_size)

        # Check if each point is within the character region (non-zero pixel)
        mask = img_array[y_rand.astype(int), x_rand.astype(int)] > 0

        # Select points that fall within the character region
        valid_points = np.column_stack((x_rand[mask], y_rand[mask]))

        # Add valid points to the sample list until we reach the target number
        sampled_points.extend(valid_points[: num_points - len(sampled_points)])

    # Convert to numpy array and adjust y-coordinates to "flip" the image
    sampled_points = np.array(sampled_points)
    sampled_points[:, 1] = font_size - sampled_points[:, 1]  # Flip y-axis

    # Center and normalize points to [-1, 1] range
    # sampled_points = sampled_points - sampled_points.mean(axis=0)
    # max_abs = np.abs(sampled_points).max()
    # sampled_points = sampled_points / max_abs

    return sampled_points
