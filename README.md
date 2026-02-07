# 🎯 AI Precision Aim Tracker

An advanced computer vision project that simulates realistic AI-powered aiming mechanics using facial recognition and real-time tracking algorithms.

## 📋 Description

This application demonstrates how AI-assisted aiming systems work by implementing sophisticated target acquisition, tracking, and engagement mechanics. Using your webcam, it detects faces in real-time and simulates an auto-aim system with realistic features like confidence calculation, target locking, recoil simulation, and hit detection.

**Perfect for:**
- Understanding computer vision and tracking algorithms
- Learning about AI-assisted targeting systems
- Educational demonstrations of Kalman filtering and motion prediction
- Game development research and prototyping

## ✨ Features

### 🎮 Core Mechanics
- **Real-time Face Detection** - Uses OpenCV's Haar Cascade classifier for fast facial recognition
- **Precision Targeting** - Automatically calculates optimal hit zones (headshot positioning)
- **Confidence System** - Measures targeting accuracy before engagement (0-100%)
- **Target Lock Mechanism** - Requires sustained aim before firing (configurable threshold)
- **Automatic Firing** - Engages target when confidence and lock conditions are met

### 🔧 Advanced Features
- **Kalman Filtering** - Smooths tracking and predicts target movement
- **Multi-threaded Detection** - Separate thread for face detection to maintain high FPS
- **Recoil Simulation** - Realistic weapon kickback after each shot
- **Hit/Miss Detection** - Accurate collision detection at the moment of firing
- **Position History Smoothing** - Moving average for stable crosshair movement
- **Muzzle Flash Effect** - Visual feedback when firing
- **Hit Markers** - Yellow X markers appear on successful hits with fade-out animation

### 📊 Visual Feedback
- Dynamic crosshair that changes color based on lock status
- Confidence arc showing target acquisition progress
- Lock indicators (corner brackets when target is locked)
- Real-time FPS counter
- Live statistics: shots fired, hits, accuracy percentage
- Target ellipse overlay on detected faces
- Status messages (SEARCHING / TARGET ACQUIRED / LOCKED - FIRE!)

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- Webcam
- Internet connection (for initial Haar Cascade download)

### Required Libraries

```bash
pip install opencv-python numpy
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

### Download

```bash
git clone https://github.com/yourusername/ai-precision-aim-tracker.git
cd ai-precision-aim-tracker
```

## 🚀 Usage

### Basic Usage

Simply run the script:

```bash
python ai_aim_enhanced.py
```

### Controls

- **Q** - Quit the application
- The system operates automatically once a target is detected

### How It Works

1. **Target Detection** - Position yourself in front of the webcam
2. **Acquisition Phase** - The green crosshair will track your face
3. **Lock Building** - A confidence arc appears, filling as aim stabilizes
4. **Target Lock** - Crosshair turns red with corner brackets when locked
5. **Automatic Engagement** - System fires automatically when conditions are met
6. **Hit Detection** - Yellow X markers show successful hits

### First Run

On first execution, the script will automatically download the Haar Cascade classifier file (~900KB). This only happens once.

## ⚙️ Configuration

You can customize the behavior by modifying constants at the top of the script:

```python
# Targeting precision
CONFIDENCE_THRESHOLD = 0.85     # How accurate aim must be (0.0-1.0)
LOCK_TIME_THRESHOLD = 0.3       # Seconds target must be locked
MAX_DISTANCE_FOR_LOCK = 15      # Pixel distance for successful lock

# Firing behavior
SHOT_COOLDOWN = 0.5             # Seconds between shots

# Visual settings
CROSSHAIR_SIZE = 25             # Size of the crosshair
AIM_SMOOTHING = 0.25            # Crosshair movement speed (0.0-1.0)

# Recoil settings
RECOIL_STRENGTH = 8             # Intensity of recoil effect
RECOIL_DECAY = 0.7              # How fast recoil recovers
```

## 📊 Statistics & Output

The application displays real-time statistics:
- **FPS** - Current frames per second
- **Status** - Current system state
- **Shots Fired** - Total number of engagements
- **Hits** - Successful hits on target
- **Accuracy** - Hit percentage

### Console Output

```
=== AI AIMING SIMULATOR ===
Press 'q' to quit...
The system fires automatically when target is locked!

SHOT #1 - HIT! - Accuracy: 100.0%
SHOT #2 - MISS (target moved/too far) - Accuracy: 50.0%
SHOT #3 - HIT! - Accuracy: 66.7%

=== FINAL STATISTICS ===
Total Shots: 3
Hits: 2
Accuracy: 66.7%
```

## 🧪 Technical Details

### Algorithms Used

- **Haar Cascade Classifier** - Face detection
- **Kalman Filter** - Motion prediction and smoothing
- **CLAHE** - Contrast Limited Adaptive Histogram Equalization for better detection
- **Moving Average** - Position history smoothing
- **Euclidean Distance** - Hit detection calculations

### Performance Optimization

- Multi-threaded face detection (separate detection thread)
- Frame buffering minimization for reduced latency
- Optimized camera settings (auto-focus disabled, minimal buffer)
- Efficient queue management
- Target position caching for smoother tracking

### Accuracy System

The system uses multiple checks to ensure accurate hit detection:

1. **Distance Check** - Crosshair must be within threshold pixels of target
2. **Current Frame Check** - Target must be detected in the current frame (not cached)
3. **Lock Verification** - Both confidence and lock-time requirements must be met
4. **Recoil Compensation** - Applied after shot for realistic behavior

## 🔒 Safety & Ethics

**Important Notice:** This is an educational project for learning computer vision and tracking algorithms. It should only be used for:

- Educational purposes
- Understanding CV algorithms
- Game development research
- Technical demonstrations

**Do NOT use this for:**
- Any harmful purposes
- Surveillance without consent
- Invasion of privacy
- Weapon systems

## 🐛 Troubleshooting

### Camera not detected
```python
# Check available cameras
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Low FPS
- Close other camera applications
- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT`
- Disable `USE_KALMAN` for faster processing

### Poor detection
- Ensure good lighting
- Face the camera directly
- Adjust `minNeighbors` in detection parameters
- Clean your webcam lens

### Download issues
- Check internet connection
- Manually download from: https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml
- Place in the same directory as the script

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🎓 Learning Resources

Want to learn more about the concepts used?

- [OpenCV Documentation](https://docs.opencv.org/)
- [Kalman Filter Explained](https://www.kalmanfilter.net/)
- [Computer Vision Basics](https://opencv.org/university/)
- [Face Detection with Haar Cascades](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/ai-precision-aim-tracker](https://github.com/yourusername/ai-precision-aim-tracker)

## 🌟 Acknowledgments

- OpenCV team for the excellent computer vision library
- Haar Cascade classifiers from OpenCV repository
- Python community for NumPy and other tools

---

**⭐ If you found this project interesting or useful, please consider giving it a star!**
