import cv2
import numpy as np
import pyaudio
import speech_recognition as sr
import threading
import queue

recognizer = sr.Recognizer()
audio_queue = queue.Queue()

def audio_detection_with_transcription(transcript_callback=None):
    mic = sr.Microphone()

    print("Starting transcription...")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Microphone calibrated for ambient noise")

    def listen_loop():
        with mic as source:
            while True:
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    audio_queue.put(audio)
                except sr.WaitTimeoutError:
                    continue
                except KeyboardInterrupt:
                    break

    def process_loop():
        while True:
            try:
                audio = audio_queue.get()
                text = recognizer.recognize_google(audio)

                print("Transcription:", text)

                if transcript_callback:
                    transcript_callback(text)  # Send to frontend/NLP/logging etc.

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"API error: {e}")
            except KeyboardInterrupt:
                break

    threading.Thread(target=listen_loop, daemon=True).start()
    threading.Thread(target=process_loop, daemon=True).start()
