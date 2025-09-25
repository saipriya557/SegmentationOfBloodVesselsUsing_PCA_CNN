import cv2
import numpy as np
import os
from skimage import exposure

# Function to process and extract the PCA image from an image
def process_image(image_path, output_folder):
    # Load the retinal image (RGB)
    img = cv2.imread(image_path)

    # Extract the green channel
    green_channel = img[:, :, 1]

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green_channel = clahe.apply(green_channel)

    # Normalize the pixel values to [0, 1]
    normalized_green_channel = enhanced_green_channel / 255.0

    # Step 1: Apply PCA for dimensionality reduction
    rows, cols = normalized_green_channel.shape
    data = normalized_green_channel.reshape(rows * cols, 1)  # Reshape to a single feature

    # Mean center the data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    # Ensure the data is 2D for PCA
    if centered_data.ndim == 1:
        centered_data = centered_data[:, np.newaxis]  # Convert to 2D

    # Compute the covariance matrix and eigen decomposition
    covariance_matrix = np.cov(centered_data, rowvar=False)
    if covariance_matrix.ndim < 2:
        covariance_matrix = np.array([[covariance_matrix]])  # Ensure 2D covariance matrix

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Project the data onto the top eigenvector
    principal_component = eigenvectors[:, 0]
    reduced_data = centered_data.dot(principal_component)

    # Reshape the reduced data back to image format
    pca_image = reduced_data.reshape(rows, cols)

    # Normalize PCA output for visualization
    pca_image_normalized = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))

    # Step 2: Apply Gamma Correction on the PCA image
    gamma = 1.5  # Adjust gamma value
    gamma_corrected = exposure.adjust_gamma(pca_image_normalized, gamma)

    # Save PCA image to the output folder
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    pca_image_path = os.path.join(output_folder, f"{base_filename}.tif")
    cv2.imwrite(pca_image_path, (gamma_corrected * 255).astype(np.uint8))

    # Return the path of the saved PCA image
    return pca_image_path

# Function to process multiple images in a folder
def process_multiple_images(image_folder, output_folder):
    # List all image files in the directory
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        
        print(f"Processing {image_file}...")
        
        # Process the image and save the PCA result
        process_image(image_path, output_folder)

# Specify the folder containing the images
image_folder = "C:\\Users\\linga\\OneDrive\\Desktop\\project\\data\\training\\images"   # Your input folder
output_folder = "C:\\Users\\linga\\OneDrive\\Desktop\\project\\data\\training\\preprocessed_images"   # Your output folder

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all images in the folder
process_multiple_images(image_folder, output_folder)
