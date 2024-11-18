#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


def generate_continuous_character_points_rejection_sampling(
    character="S", num_points=500, font_size=100
):
    # Create a blank image with enough space for the character
    image_size = (font_size * 2, font_size * 2)
    image = Image.new("L", image_size, 0)  # 'L' mode for greyscale

    # Draw the character onto the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Calculate the text bounding box and center the character
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_position = (
        (image_size[0] - text_width) // 2,
        (image_size[1] - text_height) // 2,
    )

    # Draw the character onto the image
    draw.text(text_position, character, fill=255, font=font)

    # Convert image to numpy array for pixel value access
    img_array = np.array(image)

    # Initialize list to hold sampled points
    sampled_points = []

    # Generate random points in batches for faster rejection sampling
    batch_size = num_points * 5  # Generating more points per batch increases speed
    while len(sampled_points) < num_points:
        # Generate random continuous coordinates
        x_rand = np.random.uniform(0, image_size[0], batch_size)
        y_rand = np.random.uniform(0, image_size[1], batch_size)

        # Check if each point is within the character region (non-zero pixel)
        mask = img_array[y_rand.astype(int), x_rand.astype(int)] > 0

        # Select points that fall within the character region
        valid_points = np.column_stack((x_rand[mask], y_rand[mask]))

        # Add valid points to the sample list until we reach the target number
        sampled_points.extend(valid_points[: num_points - len(sampled_points)])

    # Convert to numpy array and adjust y-coordinates to "flip" the image
    sampled_points = np.array(sampled_points)
    sampled_points[:, 1] = image_size[1] - sampled_points[:, 1]  # Flip y-axis

    # Center and normalize points to [-1, 1] range
    sampled_points = sampled_points - sampled_points.mean(axis=0)
    max_abs = np.abs(sampled_points).max()
    sampled_points = sampled_points / max_abs

    return sampled_points


if __name__ == "__main__":
    # Example usage
    points = generate_continuous_character_points_rejection_sampling(
        character="K", num_points=1500, font_size=200
    )
    plt.scatter(points[:, 0], points[:, 1], s=5)
    plt.title("Continuous Random Points Sampling the Character 'S'")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
