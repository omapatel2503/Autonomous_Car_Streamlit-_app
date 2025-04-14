import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image

# 📁 Directory containing video files
VIDEO_DIR = "demo"  # Change this to your folder

# Get list of video files
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))]

# Function to detect lanes in the image
def detect_lanes(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred  = cv2.GaussianBlur(gray, (5, 5), 0)

    
    # Apply Canny Edge Detector
    edges = cv2.Canny(blurred, 50, 150)

    # Define region of interest (ROI) mask to focus on lane area
    mask = np.zeros_like(edges)
    height, width = edges.shape
    polygon = np.array([[
        (0, height), 
        (width / 2, height / 2), 
        (width, height)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)

    # Bitwise 'and' operation to apply the mask to the edge-detected image
    masked_edges = cv2.bitwise_and(edges, mask)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)

    # Create a blank image to draw the lines
    line_image = np.copy(image)

    # Draw the detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    return line_image

# Pages in the sidebar
pages = ["Lane Detection"] + video_files
selected_page = st.sidebar.selectbox("📄 Select a Page", pages)

# Lane Detection Page
if selected_page == "Lane Detection":
    st.title("🛣️ Enhanced Lane Detection")
    
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load image and convert to RGB format
        image = Image.open(uploaded_file).convert('RGB')
        image_array = np.array(image)

        # Display original image
        st.subheader("🔍 Original Image")
        st.image(image_array, use_column_width=True)

        # Detect lanes in the image
        lane_image = detect_lanes(image_array)

        # Display the result
        st.subheader("🚧 Detected Lanes")
        st.image(lane_image, use_column_width=True)

# Video Page
else:
    # 🎬 Video page
    video_path = os.path.join(VIDEO_DIR, selected_page)
    page_title = selected_page.rsplit('.', 1)[0].replace('_', ' ').title()

    st.title(f"🎬 {page_title}")
    st.video(video_path)
