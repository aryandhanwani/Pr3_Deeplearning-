# 📸 PR 3 — Image Processing and Face Detection

## 🧾 Project Overview

This project explores **image processing and face detection** using Python, OpenCV, NumPy, and Matplotlib.

The practical work is organized into three main tasks:

1. 🛠️ Environment setup, image loading, and morphological operations.
2. 🎭 Bitwise operations, masked image overlays, histograms, and brightness/contrast adjustment.
3. 🙂 Face detection using the YuNet model on static images.

The completed practical work is available in the Jupyter Notebook: `project3_deeplearning.ipynb`.

---

## 🎯 Project Objectives

- Load and inspect images using OpenCV.
- Convert images between colour and grayscale representations.
- Create binary images using thresholding.
- Apply and compare morphological operations with different kernel shapes and sizes.
- Use bitwise operations and masks to combine or isolate image regions.
- Calculate grayscale and per-channel colour histograms.
- Adjust image brightness and contrast.
- Detect faces using the YuNet face detector.
- Visualize face bounding boxes, facial landmarks, and confidence scores.

---

## 💻 Technologies and Libraries

| Technology | Purpose |
|:---|:---|
| 🐍 Python | Main programming language |
| 👁️ OpenCV (`cv2`) | Image processing and face detection |
| 🔢 NumPy | Array and kernel operations |
| 📊 Matplotlib | Image and histogram visualization |
| 🧠 YuNet ONNX Model | Face detection |

---

## 📁 Project Structure

The suggested folder structure for running the notebook is:

```text
PR3/
│
├── project3_deeplearning.ipynb
│
└── Assets/
    ├── Grayscale_Apple.jpg
    ├── Color_Owl.jpg
    ├── Banana_noisy_binary.jpg
    ├── Headphone_color.jpg
    ├── Boat-Logo.png
    ├── Person.png
    ├── Person_2.png
    ├── Person_3.jpg
    └── face_detection_yunet_2023mar.onnx
```

📌 **Note:** The notebook reads the images and YuNet model from a local `Assets` folder. Make sure these files are available and update the paths in the notebook if your folder location is different.

---

## ⚙️ Installation and Setup

### 1️⃣ Install Python

Install Python and use a Jupyter-compatible environment, such as:

- Jupyter Notebook
- Visual Studio Code with the Jupyter extension

### 2️⃣ Install the Required Libraries

Run the following command in a terminal or notebook cell:

```bash
pip install opencv-python numpy matplotlib
```

### 3️⃣ Prepare the Assets

Place all the required input images and the `face_detection_yunet_2023mar.onnx` model file inside the `Assets` folder.

### 4️⃣ Update File Paths

The notebook currently uses local Windows paths, for example:

```python
r'D:\Deep learning\Pr.3\Assets\Color_Owl.jpg'
```

Update these paths according to the location of your own `Assets` folder.

### 5️⃣ Run the Notebook

1. Open `project3_deeplearning.ipynb`.
2. Select the appropriate Python environment.
3. Make sure all required libraries and assets are available.
4. Execute the notebook cells in order, from top to bottom.

---

## 🧪 Tasks and Implementation

### 🖼️ Task 1: Environment Setup, Image I/O & Morphological Operations

#### 1.1 🛠️ Environment Setup and Image Loading

- Imported OpenCV, NumPy, and Matplotlib.
- Checked the installed OpenCV version.
- Loaded image files using `cv2.imread()`.
- Inspected image dimensions and data type using `.shape` and `.dtype`.
- Created a `show()` helper function to display images with Matplotlib.
- The helper function converts colour images from OpenCV's BGR channel order to RGB before displaying them.

#### 1.2 ⚫ Binarize a Test Image

- Converted a colour image to grayscale using `cv2.cvtColor()`.
- Applied binary thresholding using `cv2.threshold()` with:
  - Threshold value: `127`
  - Maximum value: `255`
- Displayed the grayscale and thresholded images for comparison.

#### 1.3 🧩 Apply the Four Morphological Operations

Created a `5 × 5` kernel using NumPy and applied the following morphological operations:

| Operation | OpenCV Function | Purpose |
|:---|:---|:---|
| Erosion | `cv2.erode()` | Shrinks foreground regions |
| Dilation | `cv2.dilate()` | Expands foreground regions |
| Opening | `cv2.morphologyEx()` with `cv2.MORPH_OPEN` | Removes small foreground noise |
| Closing | `cv2.morphologyEx()` with `cv2.MORPH_CLOSE` | Fills small gaps and holes |

Displayed the four resulting images in a comparison layout.

#### 1.4 🔍 Kernel Shape and Size Comparison

Created rectangular, elliptical, and cross-shaped structuring elements using `cv2.getStructuringElement()`.

Compared the following kernel sizes:

- `3 × 3`
- `5 × 5`
- `9 × 9`

Applied opening and closing operations to the noisy binary banana image and displayed the opening results for comparison.

📌 **Observation:** The `3 × 3` cross-shaped kernel gave better performance for the tested image.

---

### 🎨 Task 2: Bitwise Operations & Image Histograms

#### 2.1 🔗 Bitwise AND / OR / XOR / NOT

- Created a circular mask and a rectangular mask using NumPy and OpenCV drawing functions.
- Applied the following bitwise operations:

| Operation | OpenCV Function |
|:---|:---|
| AND | `cv2.bitwise_and()` |
| OR | `cv2.bitwise_or()` |
| XOR | `cv2.bitwise_xor()` |
| NOT | `cv2.bitwise_not()` |

- Displayed the original masks and their corresponding bitwise results.

#### 2.2 🪄 Applied Use Case: Masked Overlay

- Resized the circular mask to match the dimensions of the test image.
- Used `cv2.bitwise_and()` with the mask to extract the selected circular region from the image.
- Loaded a headphone background image and a logo.
- Resized the logo and selected a Region of Interest (ROI) from the background.
- Combined the ROI with the logo using `cv2.bitwise_or()`.
- Displayed the original background and the result with the logo.

#### 2.3 📊 Grayscale and Per-Channel Colour Histograms

- Calculated the grayscale histogram using `cv2.calcHist()`.
- Calculated separate Blue, Green, and Red channel histograms from the colour image.
- Plotted the grayscale histogram.
- Overlaid the three colour-channel histograms using Matplotlib.

#### 2.4 ☀️ Brightness and Contrast Adjustment

Used `cv2.convertScaleAbs()` to adjust image brightness and contrast.

| Image | Alpha (Contrast) | Beta (Brightness) |
|:---|:---:|:---:|
| Brightened Image | `1.5` | `30` |
| Darkened Image | `0.7` | `-30` |

- Converted both adjusted images to grayscale.
- Displayed the original grayscale image alongside the brightened and darkened versions.

---

### 🙂 Task 3: Face Detection with YuNet (OpenCV 5.0)

#### 3.1 🧠 Load the YuNet Face Detector

- Loaded a static face image.
- Read its height and width.
- Loaded the `face_detection_yunet_2023mar.onnx` model.
- Created a YuNet detector using `cv2.FaceDetectorYN_create()` with a score threshold of `0.9`.
- Set the detector input size.
- Ran face detection using `detector.detect()`.

#### 3.2 📍 Run Detection on Multiple Static Images

- Loaded three static images.
- Set the detector input size to match each image.
- Ran the detector on each image and stored the results separately.

For each detected face:

1. Extracted the bounding box coordinates `(x, y, w, h)`.
2. Drew a rectangle around the detected face using `cv2.rectangle()`.
3. Extracted the five facial landmark points.
4. Drew the facial landmarks using `cv2.circle()`.
5. Extracted the confidence score.
6. Displayed the confidence score using `cv2.putText()`.

Finally, displayed the annotated results for all three images.

📌 **Understanding YuNet Detection Output:**

Each detected face returned by YuNet contains:

- **Bounding Box:** Coordinates defining the detected face region.
- **Facial Landmarks:** Five facial landmark coordinate pairs.
- **Confidence Score:** A value indicating the detector's confidence in the face detection.

---

## 📈 Results

The notebook contains visual outputs demonstrating the following implementations:

- ✅ Grayscale conversion and binary thresholding.
- ✅ Erosion, dilation, opening, and closing.
- ✅ Morphological comparisons using different kernel shapes and sizes.
- ✅ Bitwise mask operations (AND, OR, XOR, NOT).
- ✅ Masked image region extraction.
- ✅ Logo overlay on a background image.
- ✅ Grayscale and colour-channel histograms.
- ✅ Brightness and contrast adjustments.
- ✅ YuNet face detection on three static images.
- ✅ Face bounding boxes, facial landmarks, and confidence scores.

---

## 📝 Notes and Scope

- The notebook uses local image and model files. These assets must be available at the paths specified in the code.
- The face detector requires the YuNet ONNX model file.
- The uploaded notebook includes Tasks 1, 2, and 3.1–3.2.
- 🚧 A live IP Webcam/video-stream implementation and FPS measurement for Task 3.3 are not included in the uploaded notebook, so they are not described as completed work here.

---

## 🏁 Conclusion

This practical demonstrates fundamental image-processing techniques using OpenCV, including thresholding, morphological transformations, bitwise operations, masking, histogram analysis, and brightness/contrast adjustment.

It also demonstrates how to load and implement the **YuNet face detection model** to detect faces in multiple static images and visualize their bounding boxes, facial landmarks, and confidence scores.

Through this practical, the project provides hands-on experience with image manipulation, computer vision fundamentals, and face detection using Python.

---

## ✨ Project Information

| Detail | Information |
|:---|:---|
| **Project** | PR 3 — Image Processing and Face Detection |
| **Notebook** | `project3_deeplearning.ipynb` |
| **Language** | Python |
| **Main Library** | OpenCV |
| **Face Detection Model** | YuNet ONNX |

📓 **Main Notebook:** `project3_deeplearning.ipynb`

⭐ *If you find this project useful, consider giving the repository a star!*
