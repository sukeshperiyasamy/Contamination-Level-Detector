import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Define paths
gradient_image_path = 'static/gradient_image.jpeg'  # Static gradient image
results_folder = 'static/results/'  # Folder to store result plots
os.makedirs(results_folder, exist_ok=True)  # Create results folder if it doesn't exist

# Load the static gradient image
gradient_image = Image.open(gradient_image_path).convert("RGB")

# Process the gradient image
gradient_array = np.array(gradient_image)
gradient_height, gradient_width, _ = gradient_array.shape
bin_width = gradient_width // 100  # We have 100 bins

# Calculate average color for each bin in the gradient image
bin_colors = []
for i in range(100):
    start_x = i * bin_width
    end_x = start_x + bin_width if i < 99 else gradient_width
    bin_pixels = gradient_array[:, start_x:end_x, :].reshape(-1, 3)
    bin_avg_color = bin_pixels.mean(axis=0)
    bin_colors.append(bin_avg_color)

# Function to process the uploaded image and detect contamination level
def detect_contamination_level(uploaded_image):
    # Load the uploaded image
    input_image = Image.open(uploaded_image).convert("RGB")
    input_array = np.array(input_image)
    input_height, input_width, _ = input_array.shape
    input_center_color = input_array[input_height // 2, input_width // 2]

    # Normalize the input center color to [0, 1] range
    input_center_color = input_center_color / 255.0
    bin_colors_normalized = [bin_color / 255.0 for bin_color in bin_colors]

    # Find the closest bin to the input image's center color
    min_distance = float('inf')
    closest_bin = -1
    for i, bin_color in enumerate(bin_colors_normalized):
        # Euclidean distance (L2 norm) between the input center color and the bin color
        distance = np.linalg.norm(bin_color - input_center_color)
        if distance < min_distance:
            min_distance = distance
            closest_bin = i + 1  # Bin number is 1-based

    # Determine contamination level based on the closest bin
    if closest_bin <= 15:
        contamination_level = "Safe for drinking"
    elif closest_bin <= 60:
        contamination_level = "Moderately contaminated"
    else:
        contamination_level = "Severely contaminated"

    # Define the bins and the contamination levels
    bins = list(range(1, 101))  # 1 to 100 bins
    scale = np.linspace(1, 10, 100)  # Map bins to scale of 1 to 10

    # Find the scale corresponding to the closest bin
    scale_value = scale[closest_bin - 1]

    # Plot the scale diagram
    plt.figure(figsize=(10, 2))
    plt.plot(bins, scale, label="Scale (1 to 10)", color="blue")
    plt.scatter([closest_bin], [scale_value], color="red", label=f"Input Bin {closest_bin}\nScale {scale_value:.1f}")
    plt.title(f"Contamination Level Scale\nDetected: {contamination_level}")
    plt.xlabel("Bin Number (1 to 100)")
    plt.ylabel("Scale (1 to 10)")
    plt.axvline(x=closest_bin, color="red", linestyle="--", alpha=0.7)
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Save the plot
    result_image_path = os.path.join(results_folder, "contamination_level_plot.png")
    plt.savefig(result_image_path)
    plt.close()

    return closest_bin, contamination_level, result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'imageUpload' not in request.files:
        return "No file part", 400

    file = request.files['imageUpload']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # Read the image into memory using BytesIO
        uploaded_image = BytesIO(file.read())

        # Process the image and get the contamination level
        closest_bin, contamination_level, plot_image_path = detect_contamination_level(uploaded_image)

        # Return the results as JSON
        return {
            'contaminationLevel': contamination_level,
            'closestBin': closest_bin,
            'plotImage': plot_image_path
        }

if __name__ == '__main__':
    app.run(debug=True)
