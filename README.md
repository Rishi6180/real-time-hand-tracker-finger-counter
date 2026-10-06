# Real-Time Hand Gesture & Finger Counter

A real-time hand-tracking computer vision application built with OpenCV (`cv2`) and Google MediaPipe's Hand Landmarker Tasks API.

## Features
- Dual-Hand Tracking Tracks left and right hands simultaneously with mirror correction.
- Accurate Finger Counting Euclidean knuckle-distance checks prevent closed fists from registering false thumb counts.
- Gesture Classification Detects Closed Fist, Peace Sign, Thumbs Up, and Thumbs Down.
- Visual Skeleton Overlays bone connection lines and joint landmarks.
- Automatic Asset Management Downloads the required `hand_landmarker.task` model file on first run.

## Setup & Running

1. Clone this repository
   ```bash
   git clone [httpsgithub.comYOUR_USERNAMEhand-gesture-counter.git](httpsgithub.comYOUR_USERNAMEhand-gesture-counter.git)
   cd hand-gesture-counter