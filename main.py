import os

import cv2
import dlib
import numpy as np
import requests
from imutils import face_utils
from scipy.spatial import distance as dist
from playsound import playsound
import time
import subprocess
import tkinter as tk
from tkinter import messagebox
from mapbox import Geocoder
import winsound
import webbrowser

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load the pre-trained Haar Cascade classifier for detecting mobile phones
hand_cascade = cv2.CascadeClassifier('models/hand.xml')

# Define eye aspect ratio (EAR) threshold for blinking detection
EAR_THRESHOLD = 0.2

# Define mouth aspect ratio (MAR) threshold for yawning detection
MAR_THRESHOLD = 0.5

# Initialize variables for storing last alert timestamps and event counts
last_alert_time = time.time()
last_activity_time = time.time()
left_blink_count = 0
right_blink_count = 0
yawn_count = 0

# Mapbox API credentials
MAPBOX_ACCESS_TOKEN = os.environ['MAPBOX_ACCESS_TOKEN']

# Initialize Mapbox geocoder
geocoder = Geocoder(access_token=MAPBOX_ACCESS_TOKEN)

def get_nearby_coffee_shops(latitude, longitude, radius=5000):
    response = geocoder.forward(f'{longitude},{latitude}', limit=5, types=['poi'])
    nearby_coffee_shops = []
    for feature in response.json()['features']:
        name = feature['text']
        address = feature['place_name']
        nearby_coffee_shops.append({'name': name, 'address': address})
    return nearby_coffee_shops

def open_location_in_browser(latitude, longitude):
    # Construct the Google Maps URL with latitude and longitude
    url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
    # Open the URL in the default web browser
    webbrowser.open(url)

def get_coordinates(address):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{address}.json?access_token={MAPBOX_ACCESS_TOKEN}"
    response = requests.get(url)
    data = response.json()
    if 'features' in data and data['features']:
        coordinates = data['features'][0]['geometry']['coordinates']
        return coordinates[::-1]  # Mapbox returns coordinates in [longitude, latitude] format
    else:
        return None

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    mar = (A + B + C) / 3.0
    return mar

# Function to detect eye blinking
def detect_blink(eye):
    ear = eye_aspect_ratio(eye)
    return ear < EAR_THRESHOLD

# Function to detect yawning
def detect_yawn(mouth):
    mar = mouth_aspect_ratio(mouth)
    return mar > MAR_THRESHOLD

# Function to detect if a person is handling a mobile phone
def detect_mobile_phone(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect hands (assumed to be holding the phone)
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If hands are detected, assume the person is holding a mobile phone
    if len(hands) > 0:
        return True
    else:
        return False

# Function to handle alert actions
def handle_alert():
    global last_alert_time
    current_time = time.time()
    # Check if enough time has passed since last alert
    if current_time - last_alert_time > 20:
        # Play alert sound
        winsound.Beep(1000, 1000)
        # Print alert message to console
        print("Alert: Detected Action!")
        #subprocess.run(['osascript', '-e', 'display notification "Detected Action!" with title "Alert"'])
        last_alert_time = current_time
        alert_title = "Alert: Detected Action!"
        alert_body = "You Seems Drowsy"
        # Display notification
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(alert_title, alert_body)
        root.destroy()

def handle_alert_2():
    global last_alert_time
    current_time = time.time()
    # Check if enough time has passed since last alert
    if current_time - last_alert_time > 20:
        # Play alert sound
        winsound.Beep(1000, 200)
        # Print alert message to console
        print("Alert: Detected Action!")
        #subprocess.run(['osascript', '-e', 'display notification "Detected Action!" with title "Alert"'])
        last_alert_time = current_time
        alert_title = "Alert: Detected Action!"
        alert_body = "Do Not Use Mobile Phone"
        # Display notification
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(alert_title, alert_body)
        root.destroy()

# Function to check for inactivity
def check_for_inactivity():
    global last_activity_time
    current_time = time.time()
    if current_time - last_activity_time > 30:  # Adjust as needed
        handle_alert()
        last_activity_time = current_time


def show_custom_alert(title, message):
    root = tk.Toplevel()
    root.title(title)
    root.geometry("300x150")  # Set the size of the window

    # Create a frame to hold the content
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    # Create a label to display the message
    label = tk.Label(frame, text=message, font=("Helvetica", 12), wraplength=250)
    label.pack(pady=(0, 10))  # Add padding at the bottom

    # Create an OK button to close the window
    ok_button = tk.Button(frame, text="OK", width=10, command=root.destroy)
    ok_button.pack()

    # Center the window on the screen
    root.eval('tk::PlaceWindow . center')

# Main loop for video capture and processing
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Extract coordinates for eyes and mouth
        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[48:68]

        # Detect eye blinking, yawning, and mobile phone usage
        left_blink = detect_blink(left_eye)
        right_blink = detect_blink(right_eye)
        yawn = detect_yawn(mouth)
        mobile_phone = detect_mobile_phone(frame)

        # Increment blink and yawn counts
        if left_blink:
            left_blink_count += 1
        if right_blink:
            right_blink_count += 1
        if yawn:
            yawn_count += 1

        print(left_blink_count, right_blink_count, yawn_count)

        # Check if any event count reaches three occurrences
        if left_blink_count == 300 or right_blink_count == 300 or yawn_count == 150:
            # Reset event counts
            left_blink_count = 0
            right_blink_count = 0
            yawn_count = 0

            user_location = (6.704770, 79.984970)  # Replace with user's actual location
            nearby_coffee_shops = get_nearby_coffee_shops(36.778259,-119.417931)
            if nearby_coffee_shops:
                for shop in nearby_coffee_shops:
                    shop_name = shop['name']
                    shop_address = shop['address']
                    shop_info = f"Name: {shop_name}\nAddress: {shop_address}"
                    print("Nearby Coffee Shop:")
                    print(shop_info)
                    tk.messagebox.showinfo("Nearby Coffee Shop", shop_info)

                    # Retrieve coordinates based on address
                    coordinates = get_coordinates(shop_address)
                    if coordinates:
                        latitude, longitude = coordinates
                        # Ask user if they want to view the location on the map
                        user_input = tk.messagebox.askyesno("View Location","Do you want to view the location on the map?")
                        if user_input:
                            # If user confirms, open the location in a web browser
                            open_location_in_browser(latitude, longitude)
                    else:
                        print("Failed to retrieve coordinates for this shop.")

        if mobile_phone:
            cv2.putText(frame, 'Mobile Phone Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'No Mobile Phone Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Adjust y-coordinate for line breaks
        y_coord = 100

        cv2.putText(frame, f"Left Blinks: {left_blink}", (50, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_coord += 50  # Increase y-coordinate for the next line
        cv2.putText(frame, f"Right Blinks: {right_blink}", (50, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        y_coord += 50  # Increase y-coordinate for the next line
        cv2.putText(frame, f"Yawns: {yawn}", (50, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Perform alert actions if any of the conditions are met
        if left_blink or right_blink or yawn:
            handle_alert()
        elif mobile_phone:
            handle_alert_2()
        else:
            check_for_inactivity()

        # Draw rectangles around eyes and mouth for visualization
        cv2.rectangle(frame, tuple(np.min(left_eye, axis=0)), tuple(np.max(left_eye, axis=0)), (0, 255, 0), 2)
        cv2.rectangle(frame, tuple(np.min(right_eye, axis=0)), tuple(np.max(right_eye, axis=0)), (0, 255, 0), 2)
        cv2.rectangle(frame, tuple(np.min(mouth, axis=0)), tuple(np.max(mouth, axis=0)), (0, 0, 255), 2)

        # user_location = (6.704770, 79.984970)  # Replace with user's actual location
        # nearby_coffee_shops = get_nearby_coffee_shops(-37.813629,144.963058)
        # if nearby_coffee_shops:
        #     coffee_shop_info = '\n'.join([f"{shop['name']}: {shop['address']}" for shop in nearby_coffee_shops])
        #     coffee_shop_info = '\n\n'.join([f"Name: {shop['name']}\nAddress: {shop['address']}" for shop in nearby_coffee_shops])
        #     print("Nearby Coffee Shops:")
        #     print(coffee_shop_info)
        #     tk.messagebox.showinfo("Nearby Coffee Shops", coffee_shop_info)
        #     print("Nearby Coffee Shops", coffee_shop_info)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()