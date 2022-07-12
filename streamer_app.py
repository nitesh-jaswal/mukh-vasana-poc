from streamlit_webrtc import webrtc_streamer
import numpy as np
import cv2
import av

cascade_classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml") # Simple face detection algorithm

class VideoProcessor:

    def recv(
        self, 
        frame: av.VideoFrame
    ):
        frm: np.ndarray  = frame.to_ndarray(format="bgr24")
        faces = cascade_classifier.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3) # Values from tutotial 1

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")


webrtc_streamer(key="key", video_processor_factory=VideoProcessor)