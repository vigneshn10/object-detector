import cv2  # OpenCV library for computer vision tasks
import pyttsx3  # Library for text-to-speech functionality
import numpy as np  # Numerical operations library
from pathlib import Path  # For handling file paths
import queue  # Thread-safe queue for managing speech tasks
import threading  # For handling threads
import time  # For time-related operations

class ObjectDetector:
    def __init__(self):
        # Initialize the ObjectDetector class
        
        # Initialize text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Set speaking rate
            self.tts_engine.setProperty('volume', 1.0)  # Set max volume
            
            # Test audio output
            print("Testing audio...")
            self.tts_engine.say("Audio system initialized")
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"Error initializing text-to-speech: {e}")
        
        # Initialize a thread-safe queue to manage speech tasks
        self.speech_queue = queue.Queue()
        
        # Create a thread for handling speech tasks
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()  # Start the speech thread
        
        # Load class names from a file
        with open('Object/coco.names', 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # Paths to the model's configuration and weights
        config_path = 'Object/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights_path = 'Object/frozen_inference_graph.pb'
        
        # Initialize the object detection model
        self.net = cv2.dnn.DetectionModel(weights_path, config_path)
        
        # Set model parameters
        self.net.setInputSize(320, 320)  # Input image size
        self.net.setInputScale(1.0 / 127.5)  # Normalize pixel values
        self.net.setInputMean((127.5, 127.5, 127.5))  # Subtract mean for centering
        self.net.setInputSwapRB(True)  # Swap red and blue channels
        
        # Dictionary to track the last time an object was spoken about
        self.last_spoken = {}

    def _speech_worker(self):
        """
        Continuously process speech tasks from the queue in a separate thread.
        """
        while True:
            try:
                text = self.speech_queue.get()  # Get a speech task from the queue
                if text:
                    self.tts_engine.say(text)  # Use TTS to say the text
                    self.tts_engine.runAndWait()
                time.sleep(0.1)  # Avoid busy-waiting
            except Exception as e:
                print(f"Speech error: {e}")
                time.sleep(1)  # Delay before retrying in case of error

    def speak(self, text, min_interval=2):
        """
        Add a speech task to the queue if enough time has passed since the last utterance.
        
        :param text: Text to speak
        :param min_interval: Minimum interval between repeated utterances of the same text
        """
        current_time = time.time()  # Get the current time
        if text not in self.last_spoken or (current_time - self.last_spoken[text]) >= min_interval:
            self.speech_queue.put(text)  # Add the speech task to the queue
            self.last_spoken[text] = current_time  # Update the last spoken time

    def run(self):
        """
        Main method to start the object detection process using the webcam.
        """
        # Start video capture from the webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Set frame width
        cap.set(4, 480)  # Set frame height

        print("Starting detection... Press 'q' to quit")

        while True:
            success, img = cap.read()  # Capture a frame from the webcam
            if not success:
                break

            # Perform object detection on the frame
            classIds, confs, bbox = self.net.detect(img, confThreshold=0.5)
            
            if len(classIds) != 0:
                # Iterate over detected objects
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    # Draw a bounding box around the detected object
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    
                    # Get the class name of the detected object
                    class_name = self.class_names[classId - 1].upper()
                    
                    # Display the class name and confidence score on the frame
                    cv2.putText(img, f'{class_name} {confidence:.2f}',
                                (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    
                    # Speak the detected object if confidence is high
                    if confidence > 0.7:
                        self.speak(class_name)
                        print(f"Detected: {class_name} with confidence {confidence:.2f}")

            # Show the frame with detection results
            cv2.imshow('Output', img)
            
            # Exit loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the webcam and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped")

if __name__ == "__main__":
    # Create an instance of the ObjectDetector class and run it
    detector = ObjectDetector()
    detector.run()
