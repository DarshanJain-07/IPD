# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp  # Note: We'll use the new MediaPipe Tasks API
from mediapipe.tasks.python import vision  # New Tasks API for vision solutions
import pywt                   # For wavelet-based denoising
import matplotlib.pyplot as plt
from scipy.signal import detrend, filtfilt, butter, find_peaks, convolve
import time

# ---------------------------
#           CONFIGURATION
# ---------------------------
# Duration for which the video is captured (in seconds)
CAPTURE_DURATION = 30  
# Initial delay (in seconds) to allow the signal to stabilize before recording begins
DELAY = 3              
# Approximate sampling frequency (FPS) of the camera
FS = 30.0              
# Small constant to avoid division by zero in ratio calculations
EPSILON = 1e-6         
# Size (in pixels) of the square patch around each landmark for intensity extraction
PATCH_SIZE = 5         

# Frequency domain parameters: Only frequencies within [0.65, 3.5] Hz will be considered
LOW_FREQ_BOUND = 0.65   # Lower bound in Hz (~39 BPM)
HIGH_FREQ_BOUND = 3.5   # Upper bound in Hz (~210 BPM)

# ---------------------------
#   SETUP MEDIA PIPELINE FACE LANDMARKER (New Tasks API)
# ---------------------------
# Create options for the Face Landmarker. Replace 'face_landmarker.task' with the actual model file if needed.
landmarker_options = vision.FaceLandmarkerOptions(
    model_asset_path='face_landmarker.task',  # Path to the new face landmarker model file
    running_mode=vision.RunningMode.LIVE_STREAM  # For real-time video processing
)
# Create a FaceLandmarker instance using the provided options.
face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

# ---------------------------
#   DEFINE REGIONS OF INTEREST (ROIs) AND WEIGHTS
# ---------------------------
# These are approximate landmark indices (for MediaPipe Face Landmarker) corresponding to:
# Glabella, Medial Forehead, Left Lateral Forehead, Right Malar, Right Lateral Forehead, Left Malar, and Upper Nasal Dorsum.
region_indices = {
    "glabella": 10,               # Glabella: central area between the eyebrows
    "medial_forehead": 151,        # Medial Forehead: central forehead area
    "left_lateral_forehead": 234,  # Left Lateral Forehead: left edge of the forehead
    "right_malar": 454,            # Right Malar: right cheekbone region
    "right_lateral_forehead": 93,  # Right Lateral Forehead: right edge of the forehead
    "left_malar": 323,             # Left Malar: left cheekbone region
    "upper_nasal_dorsum": 6        # Upper Nasal Dorsum: top portion of the nasal bridge
}

# Weights based on research indicating the relative contribution of each region
region_weights = {
    "glabella": 4.0,
    "medial_forehead": 3.5,
    "left_lateral_forehead": 3.0,
    "right_malar": 3.0,
    "right_lateral_forehead": 3.0,
    "left_malar": 2.5,
    "upper_nasal_dorsum": 2.5
}
# Total weight for normalization
total_weight = sum(region_weights.values())  # Expected total: 21.0

# ---------------------------
#   UTILITY FUNCTION: PATCH INTENSITY EXTRACTION
# ---------------------------
def get_patch_intensity(frame, x, y, patch_size=PATCH_SIZE):
    """
    Extract the mean intensity of the green channel from a square patch around (x, y).
    
    Parameters:
        frame (np.array): RGB image.
        x (float): x-coordinate (pixel) for the center of the patch.
        y (float): y-coordinate (pixel) for the center of the patch.
        patch_size (int): Size of the patch.
        
    Returns:
        float: Mean green channel intensity within the patch.
        
    The patch is clamped to the image boundaries.
    """
    h, w, _ = frame.shape
    half = patch_size // 2
    x_min = max(0, int(x) - half)
    x_max = min(w, int(x) + half + 1)
    y_min = max(0, int(y) - half)
    y_max = min(h, int(y) + half + 1)
    patch = frame[y_min:y_max, x_min:x_max, :]
    return np.mean(patch[:, :, 1])  # Green channel is index 1 in RGB

# ---------------------------
#   FUNCTION TO EXTRACT WEIGHTED rPPG SIGNAL USING RATIO-BASED METHOD
# ---------------------------
def extract_weighted_rppg(frame, face_landmarks):
    """
    Extracts a weighted rPPG value from the frame using the new MediaPipe Face Landmarker landmarks.
    
    For each predefined region, the function:
      - Converts the normalized landmark coordinates to pixel values.
      - Extracts a small patch from the region.
      - Computes the mean intensities for the R, G, and B channels.
      - Calculates ratios: GR = G/(R+EPSILON) and GB = G/(B+EPSILON).
      - Sums these ratios (GR+GB) as the region's contribution.
      - Multiplies by the region's weight.
    
    The final rPPG value is the weighted average (normalized by the total weight) of the contributions.
    
    Parameters:
        frame (np.array): The RGB image.
        face_landmarks: A face landmarks object from the new API (with attribute 'landmarks').
    
    Returns:
        float: The final weighted rPPG value for the frame.
    """
    h, w, _ = frame.shape
    # Get the individual channels from the RGB image
    R_channel = frame[:, :, 0]
    G_channel = frame[:, :, 1]
    B_channel = frame[:, :, 2]
    
    def region_value(region, idx):
        """
        Calculate the ratio-based value (GR + GB) for a given region using its landmark index.
        
        Parameters:
            region (str): Name of the region (used to retrieve weight).
            idx (int): Landmark index for that region.
            
        Returns:
            float: The computed ratio-based value for the region.
        """
        lm = face_landmarks.landmarks[idx]
        # Convert normalized coordinates (0-1) to pixel coordinates
        x, y = lm.x * w, lm.y * h
        # Define the patch boundaries around the landmark
        half = PATCH_SIZE // 2
        x_min = max(0, int(x) - half)
        x_max = min(w, int(x) + half + 1)
        y_min = max(0, int(y) - half)
        y_max = min(h, int(y) + half + 1)
        patch_R = R_channel[y_min:y_max, x_min:x_max]
        patch_G = G_channel[y_min:y_max, x_min:x_max]
        patch_B = B_channel[y_min:y_max, x_min:x_max]
        avg_R = np.mean(patch_R) if patch_R.size > 0 else 0
        avg_G = np.mean(patch_G) if patch_G.size > 0 else 0
        avg_B = np.mean(patch_B) if patch_B.size > 0 else 0
        # Compute the two ratios, ensuring we don't divide by zero.
        GR = avg_G / (avg_R + EPSILON)
        GB = avg_G / (avg_B + EPSILON)
        return GR + GB

    weighted_sum = 0.0
    # Loop over each region, compute its contribution, and add it with its weight.
    for region, idx in region_indices.items():
        value = region_value(region, idx)
        weighted_sum += region_weights[region] * value

    # Normalize by the total weight to yield the final rPPG value.
    return weighted_sum / total_weight

# ---------------------------
#   SIGNAL PREPROCESSING FUNCTIONS
# ---------------------------
def detrend_signal(sig):
    """
    Remove the linear trend from the rPPG signal to eliminate low-frequency drift.
    
    Parameters:
        sig (np.array): Input time-domain signal.
    
    Returns:
        np.array: The detrended signal.
    """
    return detrend(sig)

def wavelet_denoise(sig, wavelet='db4', level=4):
    """
    Apply wavelet-based denoising to reduce high-frequency noise.
    
    Parameters:
        sig (np.array): Input signal.
        wavelet (str): Type of wavelet (e.g., 'db4').
        level (int): Decomposition level.
    
    Returns:
        np.array: The denoised signal, reconstructed from thresholded wavelet coefficients.
    """
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
    denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(denoised_coeffs, wavelet)

def bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a Butterworth bandpass filter to isolate frequencies corresponding to typical heart rates.
    
    Parameters:
        data (np.array): Input signal.
        lowcut (float): Lower cutoff frequency (Hz).
        highcut (float): Upper cutoff frequency (Hz).
        fs (float): Sampling frequency.
        order (int): Filter order.
    
    Returns:
        np.array: The filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def pan_tompkins_peak_detection(ppg_signal, fs):
    """
    Detect peaks in the preprocessed rPPG signal using a Pan–Tompkins-inspired algorithm.
    
    Steps:
      1. Compute the derivative to emphasize rapid changes.
      2. Square the derivative to amplify larger differences.
      3. Integrate the squared signal over a short window to smooth it.
      4. Detect peaks based on an adaptive threshold (35% of the maximum integrated value).
    
    Parameters:
        ppg_signal (np.array): Preprocessed rPPG signal.
        fs (float): Sampling frequency.
    
    Returns:
        peaks (np.array): Indices of detected peaks.
        derivative_signal (np.array): The computed derivative of the signal.
        squared_signal (np.array): Squared derivative.
        integrated_signal (np.array): The integrated (smoothed) signal.
    """
    derivative_signal = np.diff(ppg_signal, prepend=ppg_signal[0])
    squared_signal = derivative_signal ** 2
    window_size = int(0.15 * fs) if fs > 0 else 1
    integrated_signal = convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
    peak_threshold = 0.35 * np.max(integrated_signal)
    min_distance = int(0.5 * fs)
    peaks, _ = find_peaks(integrated_signal, height=peak_threshold, distance=min_distance)
    return peaks, derivative_signal, squared_signal, integrated_signal

# ---------------------------
#   FREQUENCY DOMAIN HEART RATE ESTIMATION FUNCTION
# ---------------------------
def frequency_domain_hr_estimation(signal, fs, low_bound=LOW_FREQ_BOUND, high_bound=HIGH_FREQ_BOUND):
    """
    Estimate heart rate by converting the time-domain rPPG signal to the frequency domain using FFT.
    
    Steps:
      1. Compute the FFT of the signal.
      2. Determine frequency bins corresponding to the FFT output.
      3. Restrict analysis to the frequency band [low_bound, high_bound] (in Hz).
      4. Find the dominant frequency in this band and convert it to BPM.
    
    Parameters:
        signal (np.array): Filtered rPPG signal.
        fs (float): Sampling frequency.
        low_bound (float): Lower frequency bound (Hz).
        high_bound (float): Upper frequency bound (Hz).
    
    Returns:
        hr_bpm (float): Estimated heart rate in beats per minute.
        fft_freq (np.array): Frequency bins from FFT.
        fft_magnitude (np.array): Magnitude of FFT components.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    fft_freq = np.fft.rfftfreq(n, d=1/fs)
    valid_idx = np.where((fft_freq >= low_bound) & (fft_freq <= high_bound))[0]
    if len(valid_idx) == 0:
        return 0, fft_freq, np.abs(fft_vals)
    dominant_idx = valid_idx[np.argmax(np.abs(fft_vals[valid_idx]))]
    dominant_freq = fft_freq[dominant_idx]
    hr_bpm = dominant_freq * 60  # Convert Hz to BPM
    return hr_bpm, fft_freq, np.abs(fft_vals)

# ---------------------------
#       MAIN VIDEO CAPTURE AND SIGNAL EXTRACTION
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

capture_started = False
start_tick = 0
user_signals = []  # List to store the weighted rPPG signal value from each frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert captured frame from BGR to RGB as required by the new MediaPipe API
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use the new Face Landmarker to detect facial landmarks
    detection_result = face_landmarker.detect(rgb_frame)
    
    # Provide visual feedback on the frame for face detection status
    if detection_result.face_landmarks:
        cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If capturing is active, display remaining capture time and check for end of capture
    if capture_started:
        current_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
        remaining = CAPTURE_DURATION + DELAY - current_time
        cv2.putText(frame, f"Remaining: {remaining:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
        if current_time >= (CAPTURE_DURATION + DELAY):
            print("Capture completed automatically.")
            capture_started = False
            break

    # If capturing and a face is detected, extract the weighted rPPG signal from the face
    if capture_started and detection_result.face_landmarks:
        # Use the first detected face
        face_landmarks = detection_result.face_landmarks[0]
        if current_time >= DELAY:
            # Extract weighted rPPG signal using our ratio-based method
            rppg_value = extract_weighted_rppg(frame, face_landmarks)
            user_signals.append(rppg_value)

    cv2.imshow("MediaPipe rPPG Capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Begin capturing when 's' is pressed
        capture_started = True
        start_tick = cv2.getTickCount()
        user_signals = []  # Reset previously collected signals
        print("Capture started...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"User signals captured: {len(user_signals)}")
if len(user_signals) < 2:
    print("Not enough data collected. Exiting.")
    exit()

# Convert the collected signal list into a NumPy array for processing
user_signals = np.array(user_signals, dtype=np.float32)

# ---------------------------
#    POST-PROCESSING PIPELINE
# ---------------------------
# Step 1: Remove low-frequency drift via detrending
detrended = detrend_signal(user_signals)
# Step 2: Denoise the detrended signal using wavelet thresholding
denoised = wavelet_denoise(detrended, wavelet='db4', level=4)
# Step 3: Apply a Butterworth bandpass filter to isolate heart rate frequencies (0.5-3.5 Hz)
filtered = bandpass_filter(denoised, 0.5, 3.5, FS, order=2)

# Option A: Estimate HR using time-domain peak detection (Pan–Tompkins-inspired)
peaks, deriv, squared, integrated = pan_tompkins_peak_detection(filtered, FS)
if len(peaks) > 1:
    intervals = np.diff(peaks) / FS  # Calculate time intervals between peaks in seconds
    hr_bpm_time = 60.0 / np.mean(intervals)  # Convert average interval to BPM
else:
    hr_bpm_time = 0

# Option B: Estimate HR using frequency-domain analysis (FFT)
hr_bpm_freq, fft_freq, fft_magnitude = frequency_domain_hr_estimation(filtered, FS)

# Print the estimated heart rates from both methods for comparison
print(f"Estimated HR (Time-domain, Pan-Tompkins): {hr_bpm_time:.2f} BPM")
print(f"Estimated HR (Frequency-domain, FFT): {hr_bpm_freq:.2f} BPM")

# ---------------------------
#       PLOTTING RESULTS
# ---------------------------
plt.figure(figsize=(12, 12))

# Plot raw weighted signal, detrended and denoised versions
plt.subplot(5, 1, 1)
plt.plot(user_signals, label='Raw Weighted rPPG Signal (GRGB)')
plt.plot(detrended, label='Detrended Signal')
plt.plot(denoised, label='Wavelet Denoised Signal')
plt.title("Raw, Detrended, and Denoised rPPG Signal (Weighted GRGB)")
plt.legend()

# Plot the bandpass filtered signal
plt.subplot(5, 1, 2)
plt.plot(filtered, label='Bandpass Filtered Signal')
plt.title("Bandpass Filtered Signal (0.5-3.5 Hz)")
plt.legend()

# Plot the derivative and squared signals used in the Pan–Tompkins process
plt.subplot(5, 1, 3)
plt.plot(deriv, label='Derivative')
plt.plot(squared, label='Squared', alpha=0.7)
plt.title("Pan–Tompkins: Derivative & Squared Signal")
plt.legend()

# Plot the integrated signal with detected peaks
plt.subplot(5, 1, 4)
plt.plot(integrated, label='Integrated Signal')
if len(peaks) > 0:
    plt.plot(peaks, integrated[peaks], 'ro', label='Detected Peaks')
plt.title("Integrated Signal & Detected Peaks")
plt.legend()

# Plot the FFT magnitude spectrum for frequency-domain HR estimation
plt.subplot(5, 1, 5)
plt.plot(fft_freq, fft_magnitude, label='FFT Magnitude Spectrum')
plt.xlim([LOW_FREQ_BOUND, HIGH_FREQ_BOUND])
plt.title("Frequency Domain Analysis")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()

plt.tight_layout()
plt.show()

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
