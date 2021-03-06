import face_recognition
import cv2
import numpy as np
import os
import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

class VideoProcessor:
	def recv(self, frame):
		return frame

webrtc_streamer(key="key", video_processor_factory=VideoProcessor, rtc_configuration=RTCConfiguration(
					{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
					))