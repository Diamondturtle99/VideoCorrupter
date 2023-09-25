import cv2
import numpy as np
from tqdm import tqdm

# Function to pixelate a frame with a random pixel size in the given range
def pixelate(frame, min_pixel_size, max_pixel_size):
    pixel_size = np.random.uniform(min_pixel_size, max_pixel_size)
    height, width = frame.shape[:2]
    temp = cv2.resize(frame, (int(width // pixel_size), int(height // pixel_size)), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

# Function to add noise with a random factor in the given range
def add_noise(frame, min_noise_factor, max_noise_factor):
    noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)
    # Ensure the noise range is within [0, 255] for uint8
    noise = np.random.randint(-noise_factor * 32 + 32, noise_factor * 32 + 32, frame.shape, dtype=np.uint8)
    noisy_frame = cv2.add(frame, noise)
    return noisy_frame

# Function to randomly smudge a part of the frame
def smudge(frame, smudge_probability, min_smudge_size, max_smudge_size):
    if np.random.rand() < smudge_probability:
        height, width, _ = frame.shape
        smudge_size = np.random.randint(min_smudge_size, max_smudge_size)
        for _ in range(smudge_size):
            y = np.random.randint(0, height)
            x = np.random.randint(0, width)
            frame[y:y+3, x:x+3] = np.mean(frame[y:y+3, x:x+3], axis=(0, 1), dtype=np.uint8)
    return frame

# Function to add dead and stuck pixels
def add_dead_and_stuck_pixels(frame, dead_pixel_probability, stuck_pixel_probability):
    height, width, _ = frame.shape
    for _ in range(height * width):
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        if np.random.rand() < dead_pixel_probability:
            frame[y, x] = [0, 0, 0]  # Set the pixel to black (dead pixel)
        elif np.random.rand() < stuck_pixel_probability:
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            frame[y, x] = color  # Set the pixel to a random color (stuck pixel)
    return frame

# Function to add text overlay to a frame
def add_text_overlay(frame, text):
    height, width, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (0, 0, 0)  # Black text
    background_color = (0, 0, 255)  # Red background
    text_size = cv2.getTextSize(text, font, font_scale, 4)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, 4, cv2.LINE_AA)
    return frame

# Open the video file
video_path = 'File.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Calculate the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the initial pixelation size and noise factor range
initial_pixel_size = 10
min_pixel_size = 1
max_pixel_size = 30
initial_noise_factor = 0.1
min_noise_factor = 0.01
max_noise_factor = 0.5

# Define smudging parameters
smudge_probability = 0.1  # Probability of applying smudging on a frame
min_smudge_size = 20      # Minimum size of smudged area
max_smudge_size = 40      # Maximum size of smudged area

# Define dead and stuck pixel parameters
dead_pixel_probability = 0.01  # Probability of adding a dead pixel to a frame
stuck_pixel_probability = 0.02  # Probability of adding a stuck pixel to a frame

# Create VideoWriter object with default codec for AVI format
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('corrupted_video.avi', fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

# Create a tqdm progress bar to track the processing progress
with tqdm(total=total_frames) as pbar:
    frame_count = 0
    text_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Randomly change pixel size and noise factor every 3 frames
        if frame_count % 3 == 0:
            pixel_size = np.random.uniform(min_pixel_size, max_pixel_size)
            noise_factor = np.random.uniform(min_noise_factor, max_noise_factor)

        # Apply pixelation effect to the frame with the current pixel size
        pixelated_frame = pixelate(frame, pixel_size, pixel_size)

        # Add noise to the frame with the current noise factor
        noisy_frame = add_noise(pixelated_frame, noise_factor, noise_factor)

        # Apply smudging effect every 50 frames
        if frame_count % 50 == 0:
            noisy_frame = smudge(noisy_frame, smudge_probability, min_smudge_size, max_smudge_size)

        # Add dead and stuck pixels to the frame
        noisy_frame = add_dead_and_stuck_pixels(noisy_frame, dead_pixel_probability, stuck_pixel_probability)

        output.write(noisy_frame)

        # If it's time to add the text frames (at the end of the video)
        if frame_count >= total_frames - 100:
            text_frame = np.zeros_like(frame)
            text_frame = add_text_overlay(text_frame, 'This Video Was Distorted By CACTUSMAXIMUS')
            text_frames.append(text_frame)

        frame_count += 1
        pbar.update(1)  # Update the progress bar

# Write the text frames to the output
for text_frame in text_frames:
    output.write(text_frame)

# Release video capture and writer
cap.release()
output.release()
cv2.destroyAllWindows()

