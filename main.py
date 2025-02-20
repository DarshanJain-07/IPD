# =============================================================================
#   Import Libraries and Tasks API Classes
# =============================================================================
import cv2
import numpy as np
import mediapipe as mp         # Main MediaPipe package (Tasks API available here)
from mediapipe import solutions     # Add this import for updated MediaPipe structure
from mediapipe.framework.formats import landmark_pb2  # Add this for landmark processing
from mediapipe.tasks.python import vision  # Recommended import per official instructions
import time                    # For timestamps and delays
import pywt                    # For wavelet-based denoising
import matplotlib.pyplot as plt  # For plotting graphs
from scipy.signal import detrend, filtfilt, butter, find_peaks, convolve  # Signal processing functions

# ==== ADD FOR FASTAPI & THREAD SAFETY (REMOVE LATER if not needed) ====
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import threading
import io

app = FastAPI()
capture_lock = threading.Lock()

# Global variables for capture state
capture_started = False
start_tick = 0
baseline_signals = []   # Accumulates rPPG values during the delay period.
valid_signals = []      # Adjusted rPPG values after delay.
valid_timestamps = []   # Timestamps (in seconds) for valid frames.
last_valid_adjusted = None
baseline_computed = False
# ==== END FASTAPI & THREAD SAFETY IMPORTS ====

# =============================================================================
#   Setup Tasks API Classes from MediaPipe
# =============================================================================
# Extract necessary classes from the Tasks API.
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# =============================================================================
#   Configuration Parameters
# =============================================================================
# These parameters control video capture duration, stabilization delay, FPS, and patch size.
CAPTURE_DURATION = 30   # Total duration (in seconds) for video capture.
DELAY = 3               # Initial delay (in seconds) for signal stabilization.
FS = 30.0               # Estimated frames per second (FPS) from the webcam.
EPSILON = 1e-6          # Small constant to avoid division by zero.
PATCH_SIZE = 5          # Size (in pixels) of the patch extracted around each landmark.

# Frequency bounds (in Hz) for heart rate analysis (e.g., 0.65 Hz ~39 BPM, 3.5 Hz ~210 BPM).
LOW_FREQ_BOUND = 0.65 # Lower frequency bound (Hz).
HIGH_FREQ_BOUND = 3.5   # Upper frequency bound (Hz).

# =============================================================================
#   Setup Face Landmarker Using the Tasks API
# =============================================================================
# Specify the model asset file. Download the model from the official MediaPipe Face Landmarker page
# and place it in your working directory. Update the file name if necessary.
model_path = "face_landmarker.task"  # For example, "face_landmarker_v2.task" if applicable.

# Global variable to store the latest detected face landmarks.
current_face_landmarks = None

def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function for asynchronous face landmark detection.
    
    This function is automatically invoked after each inference. It stores the detected face
    landmarks into the global variable 'current_face_landmarks' for subsequent processing.
    
    Parameters:
      - result: A FaceLandmarkerResult object containing the detected face landmarks.
      - output_image: The processed output image from the model (unused here).
      - timestamp_ms: Timestamp (in milliseconds) of the processed frame.
    """
    global current_face_landmarks
    current_face_landmarks = result.face_landmarks
    # For debugging, you may uncomment the following line:
    # print(f"Result at {timestamp_ms} ms: {len(result.face_landmarks)} face(s) detected.")

def process_frame(face_landmarker, frame, timestamp_ms):
    """
    Process a frame using the MediaPipe Face Landmarker.
    
    Parameters:
        face_landmarker: MediaPipe Face Landmarker instance
        frame: numpy array containing the RGB image
        timestamp_ms: current timestamp in milliseconds
    """
    # Convert frame to MediaPipe Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # Process the frame
    face_landmarker.detect_async(mp_image, timestamp_ms)

# Create FaceLandmarkerOptions with the required BaseOptions, live stream running mode, and callback.
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# Create the Face Landmarker instance using the options.
# The context manager ensures proper resource cleanup.
with FaceLandmarker.create_from_options(options) as face_landmarker:

    # =============================================================================
    #   Define Facial Regions of Interest (ROIs) and Their Weights
    # =============================================================================
    # Here we define seven facial regions as lists of landmark indices.
    # Using lists allows averaging over multiple points per region.
    region_indices = {
        "glabella": [6, 7, 8],           # Center of forehead (most stable)
        "medial_forehead": [14, 15, 16],  # Middle forehead area
        "left_lateral_forehead": [22, 23, 24],    # Left forehead
        "right_lateral_forehead": [30, 31, 32],   # Right forehead
        "left_malar": [38, 39, 40],      # Left cheek area
        "right_malar": [46, 47, 48],     # Right cheek area
        "upper_nasal_dorsum": [54, 55, 56] # Upper nose bridge
    }
    
    # Assign weights to each region based on their contribution to a robust rPPG signal.
    region_weights = {
        "glabella": 4.0,                 # Highest weight (most stable, good perfusion)
        "medial_forehead": 3.5,          # Very stable, good perfusion
        "left_lateral_forehead": 3.0,    # Stable but may have some motion
        "right_lateral_forehead": 3.0,   # Stable but may have some motion
        "left_malar": 2.5,               # Good perfusion but more motion
        "right_malar": 3.0,              # Good perfusion but more motion
        "upper_nasal_dorsum": 2.5        # Stable but less perfusion
    }
    
    total_weight = sum(region_weights.values())  # Expected total: 21.0

    # Function to validate landmark indices
    def validate_landmarks(landmarks, region_name, indices):
        """
        Validates if the specified landmark indices exist in the detected landmarks
        """
        if not landmarks or len(landmarks) == 0:
            return False
        try:
            for idx in indices:
                _ = landmarks[idx]
            return True
        except IndexError:
            print(f"Warning: Invalid landmark index in region {region_name}")
            return False

    # =============================================================================
    #   Utility Function: Get Patch Intensity
    # =============================================================================
    def get_patch_intensity(frame, x, y, patch_size=PATCH_SIZE):
        """
        Extracts the mean intensity of the green channel from a square patch centered at (x, y).
        
        Parameters:
            - frame (np.array): The input RGB image.
            - x (float): The x-coordinate (in pixels) of the patch center.
            - y (float): The y-coordinate (in pixels) of the patch center.
            - patch_size (int): The size of the patch.
        
        Returns:
            - float: The average green channel intensity in the patch.
        
        The function computes the patch boundaries, ensures they are within the image dimensions,
        extracts the patch, and computes the mean intensity from the green channel.
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
    def region_value(region, indices, face_landmarks):
        """
        Computes the average ratio-based value for a specific facial region.
        
        For each landmark index in the list, the function:
          - Converts normalized coordinates (range [0, 1]) to pixel coordinates.
          - Extracts a patch and computes average intensities for the R, G, and B channels.
          - Computes the ratios: GR = G/(R + EPSILON) and GB = G/(B + EPSILON).
        The function then averages these values to yield a single value for the region.
        
        Parameters:
            - region (str): The name of the facial region.
            - indices (list): List of landmark indices corresponding to the region.
            - face_landmarks (list): List of landmarks from MediaPipe.
        
        Returns:
            - float: The average (GR + GB) ratio value for the region.
        
        Averaging across multiple landmarks helps mitigate noise from individual measurements.
        """
        values = []
        if validate_landmarks(face_landmarks, region, indices):
            for idx in indices:
                # MediaPipe face landmarks are directly accessible from the list
                lm = face_landmarks[idx]
                x, y = lm.x * w, lm.y * h  # Convert normalized coordinates to pixels
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
        return np.mean(values) if values else 0

    def extract_weighted_rppg(frame, face_landmarks):
        """
        Extracts a weighted, ratio-based rPPG signal from the input frame using detected face landmarks.
        
        For each facial region defined in 'region_indices', the function:
          - Iterates over the list of landmark indices for that region.
          - Converts each normalized landmark coordinate to pixel coordinates.
          - Extracts a small patch around each landmark and computes average intensities for R, G, and B.
          - Computes the ratios: GR = G/(R + EPSILON) and GB = G/(B + EPSILON) for each landmark.
          - Averages these ratio values over all landmarks in the region.
          - Multiplies the averaged value by the region's predefined weight.
        
        Parameters:
            - frame (np.array): The input RGB image.
            - face_landmarks (list): The face landmarks list provided by MediaPipe.
        
        Returns:
            - float: The computed weighted rPPG signal for the frame.
        
        This multi-point approach increases robustness by combining measurements from several points per region.
        """
        global h, w, R_channel, G_channel, B_channel  # Make these accessible to region_value
        h, w, _ = frame.shape
        # Extract individual color channels
        R_channel = frame[:, :, 0]
        G_channel = frame[:, :, 1]
        B_channel = frame[:, :, 2]
        
        weighted_sum = 0.0
        # Loop over each region, compute its average ratio value, multiply by its weight, and accumulate
        for region, indices in region_indices.items():
            value = region_value(region, indices, face_landmarks)
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
        
        Detrending removes slow variations due to motion or lighting changes, enhancing the pulsatile component.
        """
        return detrend(sig)

    def wavelet_denoise(sig, wavelet='db4', level=4):
        """
        Applies wavelet denoising to reduce high-frequency noise in the detrended rPPG signal.
        
        Parameters:
            - sig (np.array): The detrended rPPG signal.
            - wavelet (str): The type of wavelet to use (e.g., 'db4').
            - level (int): The decomposition level for the wavelet transform.
        
        Returns:
            - np.array: The denoised signal reconstructed from thresholded wavelet coefficients.
        
        The signal is decomposed into frequency components, noisy details are thresholded, and a cleaner signal is reconstructed.
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
            - lowcut (float): Lower cutoff frequency (Hz).
            - highcut (float): Upper cutoff frequency (Hz).
            - fs (float): The sampling frequency.
            - order (int): The order of the Butterworth filter.
        
        Returns:
            - np.array: The filtered signal containing only frequencies between lowcut and highcut.
        
        This filter suppresses frequencies outside the typical heart rate band, reducing noise.
        """
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data)

    def pan_tompkins_peak_detection(ppg_signal, fs):
        """
        Detects peaks in the filtered rPPG signal using a Pan–Tompkins-inspired approach.
        
        Steps:
          1. Compute the derivative to highlight rapid changes.
          2. Square the derivative to amplify significant changes.
          3. Apply moving-window integration to smooth the squared signal.
          4. Detect peaks using an adaptive threshold (35% of the maximum integrated value).
        
        Parameters:
            - ppg_signal (np.array): The filtered rPPG signal.
            - fs (float): The sampling frequency.
        
        Returns:
            - peaks (np.array): Array of indices where peaks are detected.
            - derivative_signal (np.array): The computed derivative of the signal.
            - squared_signal (np.array): The squared derivative values.
            - integrated_signal (np.array): The integrated (smoothed) signal.
        
        This method identifies the pulse peaks corresponding to heartbeats, enabling heart rate estimation.
        """
        derivative_signal = np.diff(ppg_signal, prepend=ppg_signal[0])
        squared_signal = derivative_signal ** 2
        window_size = int(0.15 * fs) if fs > 0 else 1
        integrated_signal = convolve(squared_signal, np.ones(window_size)/window_size, mode='same')
        peak_threshold = 0.35 * np.max(integrated_signal)
        min_distance = int(0.5 * fs)
        peaks, _ = find_peaks(integrated_signal, height=peak_threshold, distance=min_distance)
        return peaks, derivative_signal, squared_signal, integrated_signal

    def frequency_domain_hr_estimation(signal, fs, low_bound=LOW_FREQ_BOUND, high_bound=HIGH_FREQ_BOUND):
        """
        Estimates heart rate from the filtered rPPG signal using FFT-based frequency domain analysis.
        
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
            - hr_bpm (float): Estimated heart rate in BPM.
            - fft_freq (np.array): Frequency bins from the FFT.
            - fft_magnitude (np.array): Magnitude spectrum of the FFT.
        
        By focusing on the dominant frequency within the expected heart rate range, this method provides an HR estimate that is robust to residual noise.
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
    #       Main Video Capture and Signal Extraction Loop
    # =============================================================================
    def capture_loop():
        global capture_started, start_tick, baseline_signals, valid_signals, valid_timestamps, last_valid_adjusted, baseline_computed
        with FaceLandmarker.create_from_options(options) as face_landmarker:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Cannot open camera.")
                return
            
            # Lists for dynamic frame counting and baseline computation:
            with capture_lock:
                capture_started = True
                start_tick = cv2.getTickCount()
                # Reset all buffers.
                baseline_signals = []   # Accumulates rPPG values during the initial delay period.
                valid_signals = []      # Adjusted (and derivative) rPPG values after delay.
                valid_timestamps = []   # Timestamps (in seconds) for valid frames.
                last_valid_adjusted = None  # For temporal derivative computation.
                baseline_computed = False
                print("Capture started automatically via API")
    
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame.")
                    break
    
                # Convert frame from BGR to RGB for Tasks API.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Get the current timestamp in milliseconds.
                timestamp_ms = int(time.time() * 1000)
                
                # Process the frame using the updated MediaPipe integration
                process_frame(face_landmarker, rgb_frame, timestamp_ms)
    
                # Provide visual feedback: display "Face Detected" if face landmarks have been received.
                if current_face_landmarks:
                    cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # If capture is active, display the remaining capture time.
                with capture_lock:
                    if capture_started:
                        current_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
                        remaining = CAPTURE_DURATION + DELAY - current_time
                        cv2.putText(frame, f"Remaining: {remaining:.2f}s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        if current_time >= (CAPTURE_DURATION + DELAY):
                            print("Capture completed automatically.")
                            capture_started = False
                            break
                        
                        # If capturing and face landmarks are available, extract the weighted rPPG signal.
                        if current_face_landmarks:
                            # Use the landmarks from the first detected face.
                            landmarks = current_face_landmarks[0]
                            # During the delay period, accumulate baseline signals.
                            if current_time < DELAY:
                                rppg_value = extract_weighted_rppg(rgb_frame, landmarks)
                                baseline_signals.append(rppg_value)
                            else:
                                # After delay, compute baseline once.
                                if not baseline_computed and len(baseline_signals) > 0:
                                    baseline_value = np.mean(baseline_signals)
                                    baseline_computed = True
                                    print(f"Baseline computed: {baseline_value:.4f}")
                                
                                # Extract the current rPPG value and subtract baseline.
                                rppg_value = extract_weighted_rppg(rgb_frame, landmarks)
                                adjusted_value = rppg_value - baseline_value
                                
                                # Compute temporal derivative to enhance alternating features.
                                if last_valid_adjusted is None:
                                    final_value = adjusted_value  # For the first frame after delay.
                                else:
                                    final_value = adjusted_value - last_valid_adjusted
                                last_valid_adjusted = adjusted_value
                                valid_signals.append(final_value)
                                valid_timestamps.append(current_time)
                                
                            # Reset the global landmarks variable to avoid reusing stale results.
                            current_face_landmarks = None
    
                # ==== REMOVE LATER: Display window for debugging ====
                cv2.imshow("MediaPipe rPPG Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # ==== END REMOVE LATER ====
    
            cap.release()
            cv2.destroyAllWindows()
            print(f"Valid frames captured: {len(valid_signals)}")
            if len(valid_signals) < 2:
                print("Not enough data collected. Exiting.")
                exit()


    # Compute effective frame rate based on valid timestamps.
    duration = valid_timestamps[-1] - valid_timestamps[0]
    if duration > 0:
        effective_fs = len(valid_timestamps) / duration
    else:
        effective_fs = FS
    print(f"Effective frame rate: {effective_fs:.2f} FPS")

@app.post("/start_capture")
def start_capture():
    global capture_started
    with capture_lock:
        if not capture_started:
            capture_thread = threading.Thread(target=capture_loop)
            capture_thread.start()
            return {"status": "Capture started"}
        else:
            return {"status": "Capture already running"}

@app.get("/time_remaining")
def time_remaining():
    with capture_lock:
        if not capture_started:
            return {"time_remaining": 0}
        current_time = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
        remaining = max(0, CAPTURE_DURATION + DELAY - current_time)
        return {"time_remaining": remaining}

@app.get("/results")
def get_results():
    # Use the valid_signals (which now contain temporal derivative values) for further processing.
    user_signals = np.array(valid_signals, dtype=np.float32)

    # =============================================================================
    #    Post-Processing Pipeline
    # =============================================================================
    # Step 1: Detrend the raw rPPG signal to remove baseline drift.
    detrended = detrend_signal(user_signals)
    # Step 2: Apply wavelet denoising to reduce high-frequency noise.
    denoised = wavelet_denoise(detrended, wavelet='db4', level=4)
    # Step 3: Apply a Butterworth bandpass filter to isolate frequencies in the 0.5-3.5 Hz range.
    filtered = bandpass_filter(denoised, 0.5, 3.5, effective_fs, order=2)

    # Option A: Estimate heart rate using a time-domain method (Pan–Tompkins-inspired peak detection).
    peaks, deriv, squared, integrated = pan_tompkins_peak_detection(filtered, effective_fs)
    if len(peaks) > 1:
        intervals = np.diff(peaks) / FS  # Compute time intervals between peaks (in seconds).
        hr_bpm_time = 60.0 / np.mean(intervals)  # Convert average interval to BPM.
    else:
        hr_bpm_time = 0

    # Option B: Estimate heart rate using frequency-domain analysis (FFT).
    hr_bpm_freq, fft_freq, fft_magnitude = frequency_domain_hr_estimation(filtered, FS)

    # Print results (for debugging purposes; REMOVE LATER if needed)
    print(f"Estimated HR (Time-domain, Pan-Tompkins): {hr_bpm_time:.2f} BPM")
    print(f"Estimated HR (Frequency-domain, FFT): {hr_bpm_freq:.2f} BPM")
    
    return {"HR_time_domain": hr_bpm_time, "HR_frequency_domain": hr_bpm_freq}

@app.get("/plot")
def get_plot():
    # =============================================================================
    #       Plotting Results
    # =============================================================================
    plt.figure(figsize=(12, 12))

    # Plot 1: Raw weighted rPPG signal, detrended signal, and wavelet-denoised signal.
    plt.subplot(5, 1, 1)
    plt.plot(user_signals, label='Raw Valid Signal (Derivative Applied)')
    plt.plot(detrended, label='Detrended Signal')
    plt.plot(denoised, label='Wavelet Denoised Signal')
    plt.title("Raw, Detrended, and Denoised rPPG Signal (After Baseline Removal & Derivative)")
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

    # Plot 4: Integrated signal with detected peaks marked.
    plt.subplot(5, 1, 4)
    plt.plot(integrated, label='Integrated Signal')
    if len(peaks) > 0:
        plt.plot(peaks, integrated[peaks], 'ro', label='Detected Peaks')
    plt.title("Integrated Signal & Detected Peaks")
    plt.legend()

    # Plot 5: FFT magnitude spectrum for frequency-domain HR estimation.
    plt.subplot(5, 1, 5)
    plt.plot(fft_freq, fft_magnitude, label='FFT Magnitude Spectrum')
    plt.xlim([LOW_FREQ_BOUND, HIGH_FREQ_BOUND])
    plt.title("Frequency Domain Analysis")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.tight_layout()

    # Save plot to a buffer and return as an image.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")