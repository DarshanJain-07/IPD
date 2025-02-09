# =============================================================================
#   Import Libraries
# =============================================================================
import cv2
import numpy as np
import mediapipe as mp  # MediaPipe now includes the new Tasks API.
import pywt             # Used for wavelet-based denoising.
import matplotlib.pyplot as plt  # For plotting graphs.
from scipy.signal import detrend, filtfilt, butter, find_peaks, convolve  # For signal processing.
import time             # For handling timestamps and delays.

# =============================================================================
#   Import Classes from MediaPipe Tasks API
# =============================================================================
# Extract the required classes from the new MediaPipe Tasks API.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
# Import VisionImage and VisionImageFormat for constructing input images.
from mediapipe.tasks.vision import VisionImage, VisionImageFormat

# =============================================================================
#   Configuration Parameters
# =============================================================================
# CAPTURE_DURATION: Total duration (seconds) to capture video.
# DELAY: Initial delay (seconds) for signal stabilization.
# FS: Estimated frames per second from the camera.
# EPSILON: A small constant used to prevent division by zero.
# PATCH_SIZE: Size (in pixels) of the patch to extract around each landmark.
CAPTURE_DURATION = 30   
DELAY = 3               
FS = 30.0               
EPSILON = 1e-6          
PATCH_SIZE = 5          

# Frequency bounds (in Hz) for heart rate analysis (e.g., 0.65 Hz ~39 BPM to 3.5 Hz ~210 BPM).
LOW_FREQ_BOUND = 0.65   
HIGH_FREQ_BOUND = 3.5   

# =============================================================================
#   Setup Face Landmarker Using the New Tasks API
# =============================================================================
# Specify the model asset file path. Download the model from the official page and place it in your working directory.
model_path = "face_landmarker_v2.task"  # Update if your file name is different.

# Global variable to hold the latest face landmark detection result.
current_face_landmarks = None

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function for asynchronous face landmark detection.
    
    This function is automatically invoked when a new result is available. It saves the 
    detected face landmarks into a global variable (current_face_landmarks) for use by the
    main processing loop (e.g., for rPPG signal extraction).
    
    Parameters:
      - result: FaceLandmarkerResult object containing the detected face landmarks.
      - output_image: The processed output image (unused here).
      - timestamp_ms: Timestamp of the frame in milliseconds.
    """
    global current_face_landmarks
    current_face_landmarks = result.face_landmarks
    # Uncomment below for debugging:
    # print(f"Result received at {timestamp_ms} ms, {len(result.face_landmarks)} face(s) detected.")

# Create FaceLandmarkerOptions with required BaseOptions, running mode, and result callback.
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Create the Face Landmarker instance using a context manager.
with FaceLandmarker.create_from_options(options) as face_landmarker:

    # =============================================================================
    #   Define Facial Regions (ROIs) and Weights
    # =============================================================================
    # Define the regions of interest using lists of landmark indices. These indices are chosen
    # based on the official facial mesh and our reference image, representing:
    # Glabella, Medial Forehead, Left Lateral Forehead, Right Lateral Forehead,
    # Left Malar, Right Malar, and Upper Nasal Dorsum.
    region_indices = {
        "glabella": [6, 7, 8],
        "medial_forehead": [14, 15, 16],
        "left_lateral_forehead": [22, 23, 24],
        "right_lateral_forehead": [30, 31, 32],
        "left_malar": [38, 39, 40],
        "right_malar": [46, 47, 48],
        "upper_nasal_dorsum": [54, 55, 56]
    }
    # Assign weights to each region based on its contribution to a robust rPPG signal.
    region_weights = {
        "glabella": 4.0,
        "medial_forehead": 3.5,
        "left_lateral_forehead": 3.0,
        "right_lateral_forehead": 3.0,
        "left_malar": 2.5,
        "right_malar": 3.0,
        "upper_nasal_dorsum": 2.5
    }
    total_weight = sum(region_weights.values())  # Total weight for normalization (expected: 21.0)

    # =============================================================================
    #   Utility Function: Get Patch Intensity
    # =============================================================================
    def get_patch_intensity(frame, x, y, patch_size=PATCH_SIZE):
        """
        Extracts the mean intensity of the green channel from a square patch centered at (x, y).
        
        Parameters:
            - frame (np.array): The input RGB image.
            - x (float): The x-coordinate (in pixels) for the center of the patch.
            - y (float): The y-coordinate (in pixels) for the center of the patch.
            - patch_size (int): The size of the square patch.
        
        Returns:
            - float: The average green channel intensity within the patch.
        
        This function calculates patch boundaries (ensuring they remain within the image),
        extracts the patch, and computes the mean intensity of the green channel.
        """
        h, w, _ = frame.shape
        half = patch_size // 2
        x_min = max(0, int(x) - half)
        x_max = min(w, int(x) + half + 1)
        y_min = max(0, int(y) - half)
        y_max = min(h, int(y) + half + 1)
        patch = frame[y_min:y_max, x_min:x_max, :]
        return np.mean(patch[:, :, 1])

    # =============================================================================
    #   Function: Extract Weighted rPPG Signal (Ratio-Based)
    # =============================================================================
    def extract_weighted_rppg(frame, face_landmarks):
        """
        Extracts a weighted, ratio-based rPPG signal from the input frame using detected face landmarks.
        
        For each facial region (as defined in region_indices), this function:
          - Iterates over its list of landmark indices.
          - Converts normalized coordinates to actual pixel coordinates.
          - Extracts a small patch around each landmark and computes the average intensities
            for the R, G, and B channels.
          - Computes ratios: GR = G/(R+EPSILON) and GB = G/(B+EPSILON) for each landmark.
          - Averages the ratio values across the landmarks for that region.
          - Multiplies the region’s average ratio by its predefined weight.
        
        The final rPPG signal is the normalized weighted sum across all regions.
        
        Parameters:
            - frame (np.array): The input RGB image.
            - face_landmarks: The face landmarks object from the Face Landmarker.
        
        Returns:
            - float: The computed weighted rPPG signal for the frame.
        
        This approach leverages multiple points per region to improve robustness and accuracy.
        """
        h, w, _ = frame.shape
        # Extract individual color channels.
        R_channel = frame[:, :, 0]
        G_channel = frame[:, :, 1]
        B_channel = frame[:, :, 2]
        
        def region_value(region, indices):
            """
            Computes the average ratio-based value for a specific facial region using multiple landmarks.
            
            For each landmark index in the list, the function:
              - Converts normalized coordinates to pixel coordinates.
              - Extracts a patch and computes the average intensities for R, G, and B channels.
              - Calculates the ratios: GR = G/(R+EPSILON) and GB = G/(B+EPSILON).
            The function returns the average of these ratio values for the region.
            
            Parameters:
                - region (str): Name of the facial region (for reference).
                - indices (list): List of landmark indices for the region.
            
            Returns:
                - float: The average (GR + GB) ratio value for the region.
            
            Averaging over multiple landmarks mitigates noise from individual measurements.
            """
            values = []
            for idx in indices:
                lm = face_landmarks.landmarks[idx]
                x, y = lm.x * w, lm.y * h
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
                GR = avg_G / (avg_R + EPSILON)
                GB = avg_G / (avg_B + EPSILON)
                values.append(GR + GB)
            return np.mean(values)
        
        weighted_sum = 0.0
        # Loop over each region, compute its average ratio, multiply by its weight, and accumulate.
        for region, indices in region_indices.items():
            value = region_value(region, indices)
            weighted_sum += region_weights[region] * value
        return weighted_sum / total_weight

    # =============================================================================
    #   Signal Preprocessing Functions
    # =============================================================================
    def detrend_signal(sig):
        """
        Removes the linear trend from the raw rPPG signal to eliminate baseline drift.
        
        Parameters:
            - sig (np.array): The raw time-domain rPPG signal.
        
        Returns:
            - np.array: The detrended signal.
        
        Detrending is essential to remove slow variations that can obscure the pulsatile signal.
        """
        return detrend(sig)

    def wavelet_denoise(sig, wavelet='db4', level=4):
        """
        Applies wavelet denoising to the detrended rPPG signal to reduce high-frequency noise.
        
        Parameters:
            - sig (np.array): The detrended rPPG signal.
            - wavelet (str): The type of wavelet (e.g., 'db4').
            - level (int): The level of wavelet decomposition.
        
        Returns:
            - np.array: The denoised signal after reconstructing from thresholded wavelet coefficients.
        
        The process decomposes the signal into different frequency components, thresholds the detail coefficients,
        and reconstructs a cleaner signal.
        """
        coeffs = pywt.wavedec(sig, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(denoised_coeffs, wavelet)

    def bandpass_filter(data, lowcut, highcut, fs, order=2):
        """
        Applies a Butterworth bandpass filter to the denoised rPPG signal.
        
        Parameters:
            - data (np.array): The denoised rPPG signal.
            - lowcut (float): The lower cutoff frequency (Hz).
            - highcut (float): The upper cutoff frequency (Hz).
            - fs (float): The sampling frequency.
            - order (int): The order of the filter.
        
        Returns:
            - np.array: The filtered signal containing only frequencies between lowcut and highcut.
        
        This filtering step suppresses unwanted frequencies and retains the band typically associated with heart rate.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def pan_tompkins_peak_detection(ppg_signal, fs):
        """
        Detects peaks in the filtered rPPG signal using a Pan–Tompkins-inspired algorithm.
        
        Steps:
          1. Compute the derivative to emphasize rapid changes in the signal.
          2. Square the derivative to amplify large changes.
          3. Integrate the squared signal over a moving window to smooth it.
          4. Detect peaks using an adaptive threshold (35% of the maximum integrated value).
        
        Parameters:
            - ppg_signal (np.array): The filtered rPPG signal.
            - fs (float): The sampling frequency.
        
        Returns:
            - peaks (np.array): Indices of detected peaks.
            - derivative_signal (np.array): The computed derivative of the signal.
            - squared_signal (np.array): The squared derivative values.
            - integrated_signal (np.array): The integrated (smoothed) signal.
        
        This method identifies pulse peaks corresponding to heartbeats, which are then used to estimate heart rate.
        """
        derivative_signal = np.diff(ppg_signal, prepend=ppg_signal[0])
        squared_signal = derivative_signal ** 2
        window_size = int(0.15 * fs) if fs > 0 else 1
        integrated_signal = convolve(squared_signal, np.ones(window_size) / window_size, mode='same')
        peak_threshold = 0.35 * np.max(integrated_signal)
        min_distance = int(0.5 * fs)
        peaks, _ = find_peaks(integrated_signal, height=peak_threshold, distance=min_distance)
        return peaks, derivative_signal, squared_signal, integrated_signal

    def frequency_domain_hr_estimation(signal, fs, low_bound=LOW_FREQ_BOUND, high_bound=HIGH_FREQ_BOUND):
        """
        Estimates heart rate by converting the filtered rPPG signal to the frequency domain using FFT.
        
        Steps:
          1. Compute the Fast Fourier Transform (FFT) of the input signal.
          2. Determine frequency bins corresponding to the FFT output.
          3. Restrict analysis to frequencies between low_bound and high_bound.
          4. Identify the dominant frequency in this band and convert it to BPM.
        
        Parameters:
            - signal (np.array): The filtered rPPG signal.
            - fs (float): The sampling frequency.
            - low_bound (float): Lower frequency bound (Hz).
            - high_bound (float): Upper frequency bound (Hz).
        
        Returns:
            - hr_bpm (float): Estimated heart rate in beats per minute.
            - fft_freq (np.array): Frequency bins from the FFT.
            - fft_magnitude (np.array): Magnitude spectrum of the FFT.
        
        By focusing on the dominant frequency within the expected heart rate range, this method helps mitigate residual noise.
        """
        n = len(signal)
        fft_vals = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(n, d=1/fs)
        valid_idx = np.where((fft_freq >= low_bound) & (fft_freq <= high_bound))[0]
        if len(valid_idx) == 0:
            return 0, fft_freq, np.abs(fft_vals)
        dominant_idx = valid_idx[np.argmax(np.abs(fft_vals[valid_idx]))]
        dominant_freq = fft_freq[dominant_idx]
        hr_bpm = dominant_freq * 60  # Convert dominant frequency from Hz to BPM.
        return hr_bpm, fft_freq, np.abs(fft_vals)

    # =============================================================================
    #       MAIN VIDEO CAPTURE AND SIGNAL EXTRACTION LOOP
    # =============================================================================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        exit()

    capture_started = False
    start_tick = 0
    user_signals = []  # This list will store the weighted rPPG signal for each captured frame.

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Convert the frame from BGR (OpenCV default) to RGB (required by the Tasks API).
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get the current timestamp in milliseconds.
        timestamp_ms = int(time.time() * 1000)
        # Create an input VisionImage object from the RGB frame, specifying the image format as SRGB.
        input_image = VisionImage.create_from_array(rgb_frame, image_format=VisionImageFormat.SRGB)
        # Process the input image asynchronously using the Face Landmarker.
        face_landmarker.detect_async(input_image, timestamp_ms)

        # Provide visual feedback: display "Face Detected" if face landmarks are available.
        if current_face_landmarks:
            cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # If capture is active, compute and display remaining capture time.
        if capture_started:
            current_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
            remaining = CAPTURE_DURATION + DELAY - current_time
            cv2.putText(frame, f"Remaining: {remaining:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            if current_time >= (CAPTURE_DURATION + DELAY):
                print("Capture completed automatically.")
                capture_started = False
                break

        # If capturing and face landmarks are available, extract the weighted rPPG signal.
        if capture_started and current_face_landmarks:
            # Use the landmarks from the first detected face.
            landmarks = current_face_landmarks[0]
            if current_time >= DELAY:
                rppg_value = extract_weighted_rppg(frame, landmarks)
                user_signals.append(rppg_value)
            # Reset the global landmarks variable to avoid reusing stale results.
            current_face_landmarks = None

        cv2.imshow("MediaPipe rPPG Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # When 's' is pressed, start capturing and reset the signal buffer.
            capture_started = True
            start_tick = cv2.getTickCount()
            user_signals = []
            print("Capture started...")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"User signals captured: {len(user_signals)}")
    if len(user_signals) < 2:
        print("Not enough data collected. Exiting.")
        exit()

    # Convert the collected list of rPPG values to a NumPy array for processing.
    user_signals = np.array(user_signals, dtype=np.float32)

    # =============================================================================
    #    POST-PROCESSING PIPELINE
    # =============================================================================
    # Step 1: Detrend the raw rPPG signal to remove baseline drift.
    detrended = detrend_signal(user_signals)
    # Step 2: Apply wavelet denoising to reduce high-frequency noise.
    denoised = wavelet_denoise(detrended, wavelet='db4', level=4)
    # Step 3: Apply a Butterworth bandpass filter to isolate heart rate frequencies (0.5-3.5 Hz).
    filtered = bandpass_filter(denoised, 0.5, 3.5, FS, order=2)

    # Option A: Estimate heart rate using a time-domain method (Pan–Tompkins-inspired peak detection).
    peaks, deriv, squared, integrated = pan_tompkins_peak_detection(filtered, FS)
    if len(peaks) > 1:
        intervals = np.diff(peaks) / FS  # Calculate time intervals (in seconds) between peaks.
        hr_bpm_time = 60.0 / np.mean(intervals)  # Convert the average interval to BPM.
    else:
        hr_bpm_time = 0

    # Option B: Estimate heart rate using frequency-domain analysis (FFT method).
    hr_bpm_freq, fft_freq, fft_magnitude = frequency_domain_hr_estimation(filtered, FS)

    # Print both heart rate estimates.
    print(f"Estimated HR (Time-domain, Pan-Tompkins): {hr_bpm_time:.2f} BPM")
    print(f"Estimated HR (Frequency-domain, FFT): {hr_bpm_freq:.2f} BPM")

    # =============================================================================
    #       PLOTTING RESULTS
    # =============================================================================
    plt.figure(figsize=(12, 12))

    # Plot 1: Raw weighted rPPG signal, detrended signal, and wavelet-denoised signal.
    plt.subplot(5, 1, 1)
    plt.plot(user_signals, label='Raw Weighted rPPG Signal (GRGB)')
    plt.plot(detrended, label='Detrended Signal')
    plt.plot(denoised, label='Wavelet Denoised Signal')
    plt.title("Raw, Detrended, and Denoised rPPG Signal (Weighted GRGB)")
    plt.legend()

    # Plot 2: Bandpass filtered signal.
    plt.subplot(5, 1, 2)
    plt.plot(filtered, label='Bandpass Filtered Signal')
    plt.title("Bandpass Filtered Signal (0.5-3.5 Hz)")
    plt.legend()

    # Plot 3: Derivative and squared signals used for Pan–Tompkins peak detection.
    plt.subplot(5, 1, 3)
    plt.plot(deriv, label='Derivative')
    plt.plot(squared, label='Squared', alpha=0.7)
    plt.title("Pan–Tompkins: Derivative & Squared Signal")
    plt.legend()

    # Plot 4: Integrated signal with detected peaks.
    plt.subplot(5, 1, 4)
    plt.plot(integrated, label='Integrated Signal')
    if len(peaks) > 0:
        plt.plot(peaks, integrated[peaks], 'ro', label='Detected Peaks')
    plt.title("Integrated Signal & Detected Peaks")
    plt.legend()

    # Plot 5: FFT magnitude spectrum for frequency-domain analysis.
    plt.subplot(5, 1, 5)
    plt.plot(fft_freq, fft_magnitude, label='FFT Magnitude Spectrum')
    plt.xlim([LOW_FREQ_BOUND, HIGH_FREQ_BOUND])
    plt.title("Frequency Domain Analysis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()

    plt.tight_layout()
    plt.show()
