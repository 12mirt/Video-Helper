# Project README

# Real-Time Image and Video Processing with Mediapipe and OpenCV

This project is a real-time computer vision application leveraging OpenCV and Mediapipe to perform various tasks, such as face detection, background manipulation, and gesture-based control. It supports webcam input and interactive features for enhanced video processing.

## Features

- **Face Detection**: Identifies facial landmarks and visualizes them in the video feed.
- **Background Blur**: Applies a Gaussian blur to the background while keeping the subject in focus.
- **Background Replacement**: Replaces the background with a custom image.
- **Person Tracking**: Crops and tracks individuals dynamically in the frame.
- **Hand Gesture Control**: Uses finger gestures to toggle features on or off.
- **Face Blur**: Obscures the face in the video feed using blur or blackout masks.
- **Appearance Enhancement**: Applies bilateral filtering to improve visual quality.
- **Pause and Resume**: Allows pausing and resuming the video processing.

## Setup Instructions

### Prerequisites

Ensure the following are installed on your system:

- Python 3.7 or newer
- OpenCV (`opencv-python`)
- Mediapipe
- NumPy

You can install the dependencies using pip and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Folder Structure

The project includes a folder named `requirements`, which contains all necessary configuration and dependency files:

myapp/
│
├── static/              # Static files and resources
│   ├── config.py        # Configuration file for global settings
│   ├── example.png      # Example background image for replacement
│
├── main.py              # Main script for running the application
├── .gitignore           # Git ignore file for excluding unnecessary files
├── README.md            # Project documentation
├── requirements.txt     # List of dependencies for the project

### Config File

The project relies on a configuration file (`config.py`) to define certain global settings. Customize this file to adjust behavior such as default feature states or paths to resources (e.g., background images).

### Installation and Setup

Clone the repository and navigate to the project folder:

git clone <repository_url>
cd myapp

Install the required dependencies using requirements.txt:

pip install -r requirements.txt

Run the application:

python main.py


### Key Controls

| Key   | Function                           |
|-------|------------------------------------|
| `1`   | Toggle face blur                  |
| `2`   | Toggle background blur            |
| `3`   | Toggle person tracking            |
| `4`   | Toggle face detection             |
| `5`   | Toggle appearance enhancement     |
| `6`   | Toggle hand gesture control       |
| `9`   | Toggle face detection             |
| `p`   | Pause or resume video processing  |
| `Enter` or `q` | Exit the application     |

### Gesture-Based Controls

When hand gesture control is enabled (`6`), you can use finger gestures to toggle features:

- **1 Finger**: Toggle face blur
- **2 Fingers**: Toggle background blur
- **3 Fingers**: Toggle person tracking
- **4 Fingers**: Toggle background replacement
- **5 Fingers**: Toggle appearance enhancement

## Code Overview

The project is organized as follows:

1. **Initialization**:
   - Sets up the webcam and Mediapipe solutions for face mesh, hands, pose, and segmentation.
2. **Real-Time Processing**:
   - Processes frames from the webcam to apply selected features dynamically.
3. **Functions**:
   - `display_text`: Displays information on the frame.
   - `detect_faces`: Highlights face landmarks.
   - `blur_background`: Blurs the background.
   - `replace_background`: Replaces the background with a custom image.
   - `track_people`: Tracks individuals dynamically.
   - `count_fingers`: Counts fingers for gesture detection.
   - `face_blur`: Applies a blur to the face area.
   - `enhance_appearance`: Improves image quality with bilateral filtering.

## Future Improvements

- Add multi-language support for on-screen instructions.
- Extend hand gesture recognition for more actions.
- Optimize performance for high-resolution input.
- Add logging for feature usage and gesture recognition.

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

Enjoy working with this interactive computer vision tool!