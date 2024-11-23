import cv2
import numpy as np
import dlib
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's face detector and create facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to get fixed Region of Interest (ROI) based on facial landmarks
def get_roi(frame, landmarks, points):
    """
    Extracts the Region of Interest (ROI) from the frame based on facial landmarks.

    Args:
        frame (numpy.ndarray): The video frame containing the face.
        landmarks (dlib.full_object_detection): The facial landmarks detected by dlib.
        points (list): List of landmark points defining the ROI.

    Returns:
        roi (numpy.ndarray): The extracted region of interest.
        roi_coords (tuple): The coordinates of the ROI (x_min, y_min, x_max, y_max).

    Importance:
    - Focuses on a specific area of the face (like the forehead or cheeks) which is less prone to noise.
    - Helps in extracting a stable signal for further processing.
    """
    x = [landmarks.part(p).x for p in points]
    y = [landmarks.part(p).y for p in points]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)

# Variables to hold signal data
initial_signals = []
user_signals = []

# Capture initial environment signals for 7 seconds
capture_duration = 30  # Set capture duration to 30 seconds for consistent measurement
delay = 3  # Delay for initial transients
capture_started = False  # Flag to indicate start of capture

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for face detection
    faces = detector(gray)  # Detect faces in the frame

    roi = None
    for face in faces:
        landmarks = predictor(gray, face)  # Detect facial landmarks
        # Define fixed points for ROI, adjust as needed
        points = [36, 37, 38, 39, 40, 41]  # Using eye region as ROI
        roi, roi_coords = get_roi(frame, landmarks, points)  # Extract ROI and its coordinates
        break  # Assume only one face is detected for simplicity

    if capture_started:
        current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()  # Calculate current time in seconds

        if roi is not None:
            green_channel = roi[:, :, 1]  # Extract green channel from the ROI
            avg_green = np.mean(green_channel)  # Calculate average green value in the ROI

            if current_time < delay:
                continue  # Skip initial transient period
            if current_time < capture_duration + delay:
                user_signals.append(avg_green)  # Append the average green value to the user signals list
                
            if current_time >= capture_duration + delay:
                print("Capture completed automatically after 30 seconds.")
                capture_started = False  # Stop the capture
                break

    # Real-time feedback to the user
    if len(faces) > 0:
        cv2.putText(frame, 'Face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if roi_coords:
            cv2.rectangle(frame, (roi_coords[0], roi_coords[1]), (roi_coords[2], roi_coords[3]), (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Face Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    if capture_started:
        remaining_time = capture_duration + delay - current_time  # Calculate the remaining time for the capture
        cv2.putText(frame, f"Remaining time: {remaining_time:.2f} sec", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Face Detection', frame)  # Display the frame with annotations

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        capture_started = True
        start_time = cv2.getTickCount()  # Reset start time when capture starts
        print("Capture started...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Debugging statement
print(f"Initial signals captured: {len(initial_signals)}")
print(f"User signals captured: {len(user_signals)}")

if len(user_signals) == 0:
    print("No user signals captured. Exiting.")
    exit()

user_signals = np.array(user_signals)

# Ensure the signal length is even
if len(user_signals) % 2 != 0:
    user_signals = user_signals[:-1]

# Determine maximum level for wavelet transform
max_level = pywt.swt_max_level(len(user_signals))

# Apply Wavelet Scattering Transform for HRV extraction
def wavehrv(signal, max_level):
    """
    Applies the Wavelet Scattering Transform to extract HRV features from the signal.

    Args:
        signal (numpy.ndarray): The input signal.
        max_level (int): The maximum level for wavelet transform.

    Returns:
        scattering (numpy.ndarray): The extracted HRV features.

    Importance:
    - Extracts features that represent variations in the heart rate signal.
    - Uses wavelets for efficient and robust feature extraction.
    """
    coeffs = pywt.swt(signal, 'db1', level=max_level)
    scattering = np.concatenate([coeff[0] for coeff in coeffs], axis=0)
    return scattering

# Apply the WaveHRV algorithm
scattered_signal = wavehrv(user_signals, max_level)

# Bandpass filter design
def bandpass_filter(data, lowcut, highcut, fs, order=1):
    """
    Designs and applies a bandpass filter to the input data.

    Args:
        data (numpy.ndarray): The input signal to be filtered.
        lowcut (float): The lower cutoff frequency of the filter.
        highcut (float): The upper cutoff frequency of the filter.
        fs (float): The sampling frequency.
        order (int): The order of the filter.

    Returns:
        y (numpy.ndarray): The filtered signal.

    Importance:
    - Removes noise and unwanted frequency components from the signal.
    - Focuses on the frequency range relevant to heart rate signals.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Sampling frequency
fs = 30.0
# Apply the bandpass filter
filtered_signal = bandpass_filter(scattered_signal, 0.5, 3.5, fs)

# Pan-Tompkins algorithm for peak detection
def pan_tompkins(ppg_signal, fs):
    """
    Implements the Pan-Tompkins algorithm for peak detection in the PPG signal.

    Args:
        ppg_signal (numpy.ndarray): The filtered PPG signal.
        fs (float): The sampling frequency.

    Returns:
        peaks (numpy.ndarray): Indices of the detected peaks.

    Importance:
    - Detects peaks in the heart rate signal, which correspond to heartbeats.
    - Accurate peak detection is crucial for reliable heart rate calculation.
    """
    # Derivative of the signal
    derivative_signal = np.diff(ppg_signal)
    # Squaring the signal
    squared_signal = derivative_signal ** 2
    # Moving window integration
    window_size = int(0.15 * fs)
    integrated_signal = np.convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
    # Peak detection
    peaks, _ = find_peaks(integrated_signal, distance=fs/2.0)
    return peaks

# Detect peaks using the Pan-Tompkins algorithm
peaks = pan_tompkins(filtered_signal, fs)

# Calculate heart rate
peak_intervals = np.diff(peaks) / fs  # Calculate the intervals between detected peaks
heart_rate = 60.0 / np.mean(peak_intervals)  # Convert the average interval to heart rate in BPM

print(f"Estimated Heart Rate: {heart_rate:.2f} BPM")

# Plot the results to visualize the filtered signal and detected peaks
plt.plot(filtered_signal, label='Filtered Signal')
plt.plot(peaks, filtered_signal[peaks], 'ro', label='Detected Peaks')
plt.legend()
plt.show()

# Reset variables
initial_signals = []
user_signals = []
capture_started = False


#               Permutation and combination
#
# Band pass filter butterworth order -> 2,3 etc
# use savitzky golay filter
# heart rate frequency -> 0.5 to 5hz
# band pass filter gaussian filter
# change roi
# advanced face detection algorithm
# android app for better camera quality
# ML for better overrtime training
