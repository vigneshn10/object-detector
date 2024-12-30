# Object Detector for Blind Assistance

This project implements an object detection system designed to assist visually impaired individuals by detecting objects in real-time and providing audio feedback about the detected objects. The system uses a pre-trained SSD MobileNet v3 model and OpenCV's DNN module for object detection, while a text-to-speech engine provides spoken feedback.

## Features
- Real-time object detection using a webcam.
- Audio feedback for detected objects with high confidence.
- Supports COCO dataset object classes.
- Easy-to-setup Python-based implementation.

## How It Works
1. The webcam captures video frames in real-time.
2. Detected objects are identified using the SSD MobileNet v3 pre-trained model.
3. Object labels with confidence levels above the threshold are spoken aloud using a text-to-speech engine.
4. The system ensures that repetitive detections do not result in redundant audio alerts by maintaining a time interval for each object class.


### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8 or higher
- Git


## Credits
This project is inspired by [PriyanshChhabra0316's Object-Detection-For-A-blind-Person project](https://github.com/PriyanshChhabra0316/Object-Detection-For-A-blind-Person). Special thanks to the original author for the idea and implementation that guided the development of this project.

## License
This project is licensed under the MIT License. Feel free to use, modify, and distribute it as per the license terms.

---

If you have any questions or suggestions, feel free to raise an issue or contact me: vigneshxnallamothu@gmail.com
