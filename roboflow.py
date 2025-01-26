import os  # Import os module for interacting with the operating system
import sys  # Import sys module to access system-specific parameters and functions
import requests  # Import requests module to handle HTTP requests
from PyQt5.QtWidgets import QApplication, QMessageBox  # Import QApplication and QMessageBox from PyQt5 for GUI interactions

def upload_images_from_folder(folder, api_key, dataset_id, base_label):
    """
    Uploads images from a specified folder to a Roboflow dataset.

    Parameters:
    - folder (str): Path to the folder containing images.
    - api_key (str):  Roboflow API key.
    - dataset_id (str): The ID of the Roboflow dataset where images will be uploaded.
    - base_label (str): Base label name for the images being uploaded.
    """
    # Verify the folder exists
    if not os.path.isdir(folder):
        print(f"Folder '{folder}' does not exist.")  # Inform the user that the folder doesn't exist
        return  # Exit the function early since the folder is invalid

    # Define acceptable image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png')  # Tuple of valid image file extensions

    # List all files in the folder that have the specified image extensions
    files = [f for f in os.listdir(folder) if f.lower().endswith(image_extensions)]

    # Check if there are any image files to upload
    if not files:
        print(f"No image files found in '{folder}'.")  # Inform the user that no images were found
        return  # Exit the function early since there's nothing to upload

    counter = 1  # Initialize a counter to create unique labels for each image

    # Iterate over each image file in the folder
    for file in files:
        image_path = os.path.join(folder, file)  # Construct the full path to the image file
        unique_label = f"{base_label}{counter}"  # Create a unique label by appending the counter to the base label

        # Construct the Roboflow API upload URL with the dataset ID and API key
        upload_url = (
            f"https://api.roboflow.com/dataset/{dataset_id}/upload"
            f"?api_key={api_key}&name={unique_label}"
        )

        print(f"Uploading {image_path} as {unique_label}...")  # Inform the user about the upload progress

        # Attempt to upload the image file
        try:
            with open(image_path, "rb") as image_file:  # Open the image file in binary read mode
                response = requests.post(upload_url, files={"file": image_file})  # Send a POST request to upload the image
            print(f"Upload response for {unique_label}: {response.json()}")  # Print the JSON response from Roboflow
        except Exception as e:
            print(f"Error uploading {unique_label}: {e}")  # Print any errors that occur during the upload

        counter += 1  # Increment the counter for the next image

if __name__ == "__main__":
    # Configuration parameters
    folder_path = "/home/pi/badge/gesture_images"  # Folder containing images to upload
    api_key = "Ue5D8ad8Ta3lvh4i5Kmt"             # Roboflow API key
    dataset_id = "ai-smart-badge"                # Roboflow dataset ID
    base_label = "Gesture"                       # Base label for images

    # Upload images from the specified folder to Roboflow
    upload_images_from_folder(folder_path, api_key, dataset_id, base_label)

    # Initialize a QApplication to display a GUI message box
    app = QApplication(sys.argv)  # Create the QApplication instance with command-line arguments

    # Display an information dialog to inform the user that uploads are complete
    QMessageBox.information(
        None,  # No parent widget
        "Upload Complete",  # Title of the message box
        "Images of gestures uploaded successfully to Roboflow Dataset!"  # Message text
    )

    sys.exit(0)  # Exit the application successfully
