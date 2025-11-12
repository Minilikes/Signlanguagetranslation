import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

class GestureModule:
    def __init__(self, model_path='models/gesture_recognizer.task'):
        # Note: We have removed all 'pygame' and 'gtts' code.
        # This module will now only do recognition and translation.
        
        self.model_path = model_path

        BaseOptions = mp.tasks.BaseOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE
        )

        self.recognizer = GestureRecognizer.create_from_options(self.options)

        # Translation dictionary
        self.translation_dict = {
            "Closed_Fist": {"english": "Fist", "hindi": "मुट्ठी"},
            "Open_Palm": {"english": "Palm", "hindi": "हथेली"},
            "Pointing_Up": {"english": "Pointing Up", "hindi": "ऊपर इशारा"},
            "Thumb_Down": {"english": "Thumb Down", "hindi": "अंगूठा नीचे"},
            "Thumb_Up": {"english": "Thumb Up", "hindi": "अंगूठा ऊपर"},
            "Victory": {"english": "Victory", "hindi": "विजय"},
            "ILoveYou": {"english": "I Love You", "hindi": "मैं तुमसे प्यार करता हूँ"},
            # Add a "None" entry for when no gesture is detected
            "None": {"english": "---", "hindi": "---"}
        }

    def process_frame(self, frame_data):
        """
        Processes a single image frame and returns translation data.
        'frame_data' is a numpy array (image).
        """
        
        # Convert the frame to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_data)
        
        # Recognize gestures
        gesture_recognition_result = self.recognizer.recognize(mp_image)

        detected_gesture_name = "None"  # Default if no gesture found
        translations = self.translation_dict["None"] # Default translation

        if gesture_recognition_result.gestures:
            # Get the top gesture
            gesture_name = gesture_recognition_result.gestures[0][0].category_name
            if gesture_name in self.translation_dict:
                detected_gesture_name = gesture_name
                translations = self.translation_dict[gesture_name]

        # Return the name and the translation dictionary
        return {
            "gesture_name": detected_gesture_name,
            "translations": translations
        }

    # We no longer need set_language or speak, as the front-end will handle this.