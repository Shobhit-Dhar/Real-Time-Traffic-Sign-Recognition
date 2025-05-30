import numpy as np
import cv2
import pickle




frameWidth = 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX


# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)

# IMPORT THE TRAINED MODEL
try:
    with open("model_trained.p", "rb") as pickle_in:
        model = pickle.load(pickle_in)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: model_trained.p not found. Make sure the model file is in the correct path.")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def grayscale(img_input):
    img_output = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    return img_output

def equalize(img_input):
    img_output = cv2.equalizeHist(img_input)
    return img_output

def preprocessing(img_input):
    img_proc = grayscale(img_input)
    img_proc = equalize(img_proc)
    img_proc = img_proc / 255.0
    return img_proc


CLASS_NAMES = {
    0: 'Speed Limit 20 km/h', 1: 'Speed Limit 30 km/h', 2: 'Speed Limit 50 km/h',
    3: 'Speed Limit 60 km/h', 4: 'Speed Limit 70 km/h', 5: 'Speed Limit 80 km/h',
    6: 'End of Speed Limit 80 km/h', 7: 'Speed Limit 100 km/h', 8: 'Speed Limit 120 km/h',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield',
    14: 'Stop', 15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve', 22: 'Bumpy road',
    23: 'Slippery road', 24: 'Road narrows on the right', 25: 'Road work',
    26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def getCalssName(classNo):
    return CLASS_NAMES.get(classNo, "Unknown Class")

while True:
    # READ IMAGE
    success, imgOriginal = cap.read()
    if not success:
        print("Error: Failed to capture image")
        break

    # PROCESS IMAGE
    img_resized = cv2.resize(imgOriginal, (32, 32))
    img_processed = preprocessing(img_resized)
    cv2.imshow("Processed Image", img_processed)

    img_reshaped = img_processed.reshape(1, 32, 32, 1) # Reshape for model input


    display_img = imgOriginal.copy() # Work on a copy to avoid modifying the original if needed elsewhere
    cv2.putText(display_img, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(display_img, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    # PREDICT IMAGE
    predictions = model.predict(img_reshaped)
    classIndex = np.argmax(predictions, axis=1)[0]
    probabilityValue = np.amax(predictions)

    if probabilityValue > threshold:
        className = getCalssName(classIndex)
        cv2.putText(display_img, str(classIndex) + " " + str(className), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display_img, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Result", display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Application exited.")