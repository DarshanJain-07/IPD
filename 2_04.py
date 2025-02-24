import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from scipy import signal
import time
from sklearn.decomposition import FastICA
import scipy.stats

# --- MediaPipe Classes ---
BaseOptions = python.BaseOptions
FaceLandmarker = python.vision.FaceLandmarker
FaceLandmarkerOptions = python.vision.FaceLandmarkerOptions
VisionRunningMode = python.vision.RunningMode

# --- Constants ---
MODEL_PATH = "face_landmarker.task"  # Path to your MediaPipe model
PPG_WINDOW_SIZE = 300  # Number of samples for PPG history
SIGNAL_QUALITY_THRESHOLD = 0.4  # Signal quality threshold
HEART_RATE_SMOOTHING_WINDOW = 5  # Window size for heart rate smoothing
MIN_HR = 40  # Minimum plausible heart rate
MAX_HR = 200  # Maximum plausible heart rate
FRAME_RATE_WINDOW = 10  # Window for averaging the frame rate
ROI_SIZE = 100  # Default ROI size in pixels
DETECTION_METHODS = ["GREEN", "CHROM", "POS", "ICA"]  # Available methods

# --- Signal Processing Methods ---
def green_channel_method(frames_data):
    """
    Extract PPG signal using green channel averaging.
    Args:
        frames_data: List of dictionaries containing frame data
    Returns:
        Green channel PPG signal
    """
    green_values = [frame_data.get('green_mean', 0) for frame_data in frames_data if 'green_mean' in frame_data]
    if not green_values:
        return np.array([])
    
    # Normalize
    green_values = np.array(green_values)
    green_values = green_values - np.mean(green_values)
    
    return green_values

def chrom_method(frames_data):
    """
    CHROM method for rPPG signal extraction.
    Args:
        frames_data: List of dictionaries containing frame data with RGB means
    Returns:
        PPG signal
    """
    # Extract RGB signals
    rgb_signals = np.array([
        [frame_data.get('red_mean', 0) for frame_data in frames_data if 'red_mean' in frame_data],
        [frame_data.get('green_mean', 0) for frame_data in frames_data if 'green_mean' in frame_data],
        [frame_data.get('blue_mean', 0) for frame_data in frames_data if 'blue_mean' in frame_data]
    ])
    
    if rgb_signals.shape[1] == 0:
        return np.array([])
    
    # Normalize RGB signals
    rgb_n = rgb_signals / np.mean(rgb_signals, axis=1, keepdims=True)
    
    # CHROM components
    X_chrom = 3*rgb_n[0] - 2*rgb_n[1]
    Y_chrom = 1.5*rgb_n[0] + rgb_n[1] - 1.5*rgb_n[2]
    
    # Project onto skin-tone invariant plane
    std_X = np.std(X_chrom)
    std_Y = np.std(Y_chrom)
    alpha = std_X / std_Y if std_Y > 0 else 1.0
    
    bvp = X_chrom - alpha * Y_chrom
    
    return bvp

def pos_method(frames_data):
    """
    POS method for rPPG signal extraction.
    Args:
        frames_data: List of dictionaries containing frame data with RGB means
    Returns:
        PPG signal
    """
    # Extract RGB signals
    rgb_signals = np.array([
        [frame_data.get('red_mean', 0) for frame_data in frames_data if 'red_mean' in frame_data],
        [frame_data.get('green_mean', 0) for frame_data in frames_data if 'green_mean' in frame_data],
        [frame_data.get('blue_mean', 0) for frame_data in frames_data if 'blue_mean' in frame_data]
    ])
    
    if rgb_signals.shape[1] == 0:
        return np.array([])
    
    # Normalize RGB signals
    rgb_n = rgb_signals / np.mean(rgb_signals, axis=1, keepdims=True)
    
    # POS components
    S1 = rgb_n[0] - rgb_n[1]
    S2 = rgb_n[0] + rgb_n[1] - 2 * rgb_n[2]
    
    # Projection
    std_S1 = np.std(S1)
    std_S2 = np.std(S2)
    alpha = std_S1 / std_S2 if std_S2 > 0 else 1.0
    
    bvp = S1 - alpha * S2
    
    return bvp

def ica_method(frames_data):
    """
    ICA method for rPPG signal extraction.
    Args:
        frames_data: List of dictionaries containing frame data with RGB means
    Returns:
        PPG signal
    """
    # Extract RGB signals
    rgb_signals = np.array([
        [frame_data.get('red_mean', 0) for frame_data in frames_data if 'red_mean' in frame_data],
        [frame_data.get('green_mean', 0) for frame_data in frames_data if 'green_mean' in frame_data],
        [frame_data.get('blue_mean', 0) for frame_data in frames_data if 'blue_mean' in frame_data]
    ])
    
    if rgb_signals.shape[1] < 3 or rgb_signals.shape[0] != 3:
        return np.array([])
    
    try:
        # Transpose for sklearn's ICA implementation
        X = rgb_signals.T
        
        # Apply ICA - we need at least as many samples as components
        if X.shape[0] >= 3:
            ica = FastICA(n_components=3, random_state=42, whiten='unit-variance', max_iter=1000)
            S = ica.fit_transform(X)
            
            # Find component with highest periodicity in expected HR range
            powers = []
            for i in range(3):
                f, Pxx = signal.welch(S[:, i], fs=30, nperseg=min(256, len(S)))
                idx = np.logical_and(f >= 0.6, f <= 4.0)  # HR range (36-240 BPM)
                powers.append(np.sum(Pxx[idx]))
            
            best_idx = np.argmax(powers)
            return S[:, best_idx]
        else:
            return np.array([])
    except:
        return np.array([])

# --- Pan-Tompkins Algorithm (Enhanced) ---
def pan_tompkins(ppg_signal, fs):
    """
    Enhanced Pan-Tompkins algorithm for PPG signals
    """
    if len(ppg_signal) < fs:  # Require at least 1 second of data
        return [], 0, [], 0
    
    ppg_signal = np.array(ppg_signal)  # Ensure NumPy array
    ppg_signal = ppg_signal - np.mean(ppg_signal)  # Remove DC component
    
    # Handle low sampling rate
    if fs < 8.0:
        return [], 0, [], 0
    
    # Bandpass filter (0.5-4 Hz for heart rate: 30-240 BPM)
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    if high >= 1:
        high = 0.99
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ppg_signal)
    
    # Derivative, squaring, and integration
    derivative = np.diff(filtered_signal)
    derivative = np.append(derivative, derivative[-1])  # Match length
    squared = derivative ** 2
    
    # Moving window integration
    window_size = int(0.1 * fs)
    window_size = max(1, window_size)  # Ensure window_size is at least 1
    if window_size % 2 == 0:
        window_size += 1
    window = np.ones(window_size) / window_size
    integrated = np.convolve(squared, window, mode='same')
    
    # Dynamic thresholding
    threshold = np.mean(integrated) + 0.7 * np.std(integrated)
    
    # Peak detection with minimum distance between peaks
    min_distance = int(60/MAX_HR * fs)  # Minimum distance based on max HR
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
    
    # Calculate heart rate
    heart_rate = 0
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks)
        mean_interval = np.mean(peak_intervals)
        heart_rate = 60 * fs / mean_interval
        
        # Validate heart rate
        if heart_rate < MIN_HR or heart_rate > MAX_HR:
            heart_rate = 0
    
    # Signal quality assessment
    quality = assess_signal_quality(filtered_signal, fs)
    
    if quality < SIGNAL_QUALITY_THRESHOLD:
        heart_rate = 0
    
    return peaks.tolist(), heart_rate, filtered_signal.tolist(), quality

def assess_signal_quality(ppg_signal, fs):
    """
    Advanced signal quality assessment for rPPG.
    Returns a quality score between 0 and 1.
    """
    if len(ppg_signal) < fs:  # Need at least 1 second of data
        return 0
    
    # 1. SNR in frequency domain
    f, Pxx = signal.welch(ppg_signal, fs=fs, nperseg=min(256, len(ppg_signal)))
    hr_range = np.logical_and(f >= 0.6, f <= 4.0)  # 36-240 BPM
    if not any(hr_range):
        return 0
    
    hr_power = np.sum(Pxx[hr_range])
    total_power = np.sum(Pxx)
    snr = hr_power / (total_power - hr_power + 1e-10)  # Avoid division by zero
    snr = min(snr, 1.0)  # Cap at 1.0
    
    # 2. Periodicity assessment using autocorrelation
    autocorr = np.correlate(ppg_signal, ppg_signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0] + 1e-10  # Normalize, avoid division by zero
    
    # Find peaks in autocorrelation
    min_dist = int(fs / 4)  # Minimum distance between peaks
    peaks, _ = signal.find_peaks(autocorr, height=0.1, distance=min_dist)
    
    if len(peaks) >= 2:
        periodicity = np.mean(autocorr[peaks[:2]])
    else:
        periodicity = 0
    
    # 3. Waveform quality - use kurtosis as PPG signals typically have specific shape
    try:
        kurt = scipy.stats.kurtosis(ppg_signal)
        # PPG signals should have positive kurtosis (more peaked than normal)
        kurt_score = min(max((kurt + 3) / 6, 0), 1)  # Normalize kurtosis score
    except:
        kurt_score = 0
    
    # Combine metrics with weights
    quality_score = 0.5 * snr + 0.3 * periodicity + 0.2 * kurt_score
    
    return min(quality_score, 1.0)  # Ensure score is in [0,1]

# --- GUI Class ---
class FaceDetectionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("rPPG Heart Rate Monitor")
        self.window.geometry("800x700")  # Larger window for metrics
        
        # Create menu bar
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # Create method menu
        method_menu = tk.Menu(menubar, tearoff=0)
        self.selected_method = tk.StringVar()
        self.selected_method.set(DETECTION_METHODS[1])  # Default to GREEN
        
        for method in DETECTION_METHODS:
            method_menu.add_radiobutton(label=method, variable=self.selected_method, 
                                        command=self.reset_ppg_data)
        
        menubar.add_cascade(label="Method", menu=method_menu)
        
        # Main frames
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(expand=True, fill='both', pady=10)
        
        # Video frame
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(side=tk.TOP, fill='both', expand=True)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill='both')
        
        # Status frame
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(side=tk.TOP, fill='x', pady=5)
        
        self.method_label = ttk.Label(self.status_frame, text="Method: GREEN", font=("Arial", 12))
        self.method_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = ttk.Label(self.status_frame, text="FPS: --", font=("Arial", 12))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        self.hr_label = ttk.Label(self.status_frame, text="Heart Rate: -- BPM", font=("Arial", 12, "bold"))
        self.hr_label.pack(side=tk.LEFT, padx=10)
        
        self.quality_label = ttk.Label(self.status_frame, text="Signal Quality: --", font=("Arial", 12))
        self.quality_label.pack(side=tk.LEFT, padx=10)
        
        # Results frame for multiple methods comparison
        self.results_frame = ttk.LabelFrame(self.main_frame, text="Method Comparison")
        self.results_frame.pack(side=tk.TOP, fill='x', pady=5, padx=10)
        
        # Create result labels for each method
        self.method_results = {}
        for i, method in enumerate(DETECTION_METHODS):
            frame = ttk.Frame(self.results_frame)
            frame.grid(row=i//2, column=i%2, padx=10, pady=5, sticky="w")
            
            label = ttk.Label(frame, text=f"{method}: -- BPM (Quality: --)")
            label.pack(side=tk.LEFT)
            
            self.method_results[method] = label
        
        # --- Face Region Data ---
        self.forehead_points = [103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 285, 417, 351, 419, 197, 196, 122, 193, 55, 107, 66, 105, 104, 103] # Forehead region
        self.left_cheek_points = [345, 340, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352, 345]  # Left cheek
        self.right_cheek_points = [116, 111, 117, 118, 119, 120, 100, 142, 36, 205, 187, 123, 116]  # Right cheek
        
        # --- Data Storage ---
        self.frames_data = []

        # --- Heart Rate Variables ---
        self.heart_rates = {method: 0 for method in DETECTION_METHODS}
        self.heart_rate_histories = {method: [] for method in DETECTION_METHODS}
        self.signal_qualities = {method: 0 for method in DETECTION_METHODS}

        # --- Frame Rate and Timing ---
        self.frame_times = []
        self.fps_history = []
        self.smoothed_fps = 30.0  # Initial FPS estimate
        self.previous_time = time.perf_counter()
        
        # --- Data Reset ---
        self.reset_ppg_data()
        
        # --- MediaPipe Setup ---
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        
        # Start camera
        self.start_camera()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def reset_ppg_data(self):
        """Reset all PPG data when changing methods or as needed."""
        self.frames_data = []  # Store frame data for analysis
        self.method_label.config(text=f"Method: {self.selected_method.get()}")
        
        # Reset heart rates
        for method in DETECTION_METHODS:
            self.heart_rates[method] = 0
            self.heart_rate_histories[method] = []
            self.signal_qualities[method] = 0
            self.method_results[method].config(text=f"{method}: -- BPM (Quality: --)")
    
    def start_camera(self):
        """Initialize camera capture."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Try to set higher resolution if supported
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.update_frame()
    
    def extract_roi_data(self, frame, face_landmarks, region_points):
        """Extract RGB data from a facial region."""
        h, w = frame.shape[:2]
        points = []
        
        for point_idx in region_points:
            landmark = face_landmarks[point_idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])
        
        points = np.array(points, dtype=np.int32)
        
        # Create mask and extract RGB values
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        
        # Draw region on frame
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)
        
        # Extract RGB values
        roi_red = frame[:, :, 2][mask == 255]
        roi_green = frame[:, :, 1][mask == 255]
        roi_blue = frame[:, :, 0][mask == 255]
        
        if len(roi_red) == 0 or len(roi_green) == 0 or len(roi_blue) == 0:
            return None, frame
        
        # Calculate mean values
        roi_data = {
            'red_mean': np.mean(roi_red),
            'green_mean': np.mean(roi_green),
            'blue_mean': np.mean(roi_blue),
            'timestamp': self.timestamp_ms
        }
        
        return roi_data, frame
    
    def process_ppg_signal(self, method, frames_data):
        """Process PPG signal using selected method and calculate heart rate."""
        if len(frames_data) < 30:  # Need at least 30 frames for processing
            return 0, 0, []
        
        # Extract PPG signal based on the selected method
        ppg_signal = np.array([])
        
        if method == "GREEN":
            ppg_signal = green_channel_method(frames_data)
        elif method == "CHROM":
            ppg_signal = chrom_method(frames_data)
        elif method == "POS":
            ppg_signal = pos_method(frames_data)
        elif method == "ICA":
            ppg_signal = ica_method(frames_data)
        
        if len(ppg_signal) < 30:
            return 0, 0, []
        
        # Apply bandpass filter
        nyquist = max(self.smoothed_fps / 2, 4.0)  # Ensure minimum Nyquist freq of 4 Hz
        
        # Set cutoff frequencies for 30-240 BPM (0.5-4 Hz) heart rate range
        low = max(0.5 / nyquist, 0.01)  # Minimum value of 0.01
        high = min(4.0 / nyquist, 0.99)  # Maximum value of 0.99
        
        # Ensure that low < high with a minimum difference
        if low >= high or (high - low) < 0.1:
            # If frequencies are too close or invalid, use default values
            low = 0.1
            high = 0.9
        
        # Only apply filter if frequencies are valid
        try:
            b, a = signal.butter(2, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, ppg_signal)
        except Exception as e:
            print(f"Filter error: {str(e)}")
            filtered_signal = ppg_signal  # Use unfiltered signal as fallback
        
        # Calculate heart rate
        peaks, heart_rate, _, quality = pan_tompkins(filtered_signal, self.smoothed_fps)
        
        # Heart rate validation and smoothing
        if heart_rate > 0:
            self.heart_rate_histories[method].append(heart_rate)
            if len(self.heart_rate_histories[method]) > HEART_RATE_SMOOTHING_WINDOW:
                self.heart_rate_histories[method].pop(0)
            
            # Median filtering for robustness
            smoothed_hr = np.median(self.heart_rate_histories[method])
            self.heart_rates[method] = smoothed_hr
            self.signal_qualities[method] = quality
        
        return self.heart_rates[method], quality, filtered_signal
    
    def update_frame(self):
        """Process each frame and update the GUI."""
        try:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                return
            
            # Calculate frame rate with sanity checks
            current_time = time.perf_counter()
            elapsed = current_time - self.previous_time
            self.previous_time = current_time

            if elapsed > 0:
                fps = 1.0 / elapsed
                # Constrain FPS to reasonable values
                fps = min(max(fps, 8.0), 60.0)  # Clamp between 8 and 60 FPS
                self.fps_history.append(fps)
                if len(self.fps_history) > FRAME_RATE_WINDOW:
                    self.fps_history.pop(0)
                self.smoothed_fps = np.mean(self.fps_history)
            
            # Update timestamp
            self.timestamp_ms += int(1000 / max(self.smoothed_fps, 1))
            
            # MediaPipe face detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
            
            # Process face if detected
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                face_landmarks = detection_result.face_landmarks[0]
                
                # Extract ROI data from forehead and cheeks
                forehead_data, frame = self.extract_roi_data(frame, face_landmarks, self.forehead_points)
                left_cheek_data, frame = self.extract_roi_data(frame, face_landmarks, self.left_cheek_points)
                right_cheek_data, frame = self.extract_roi_data(frame, face_landmarks, self.right_cheek_points)
                
                # Combine ROI data (average of all regions)
                if forehead_data and left_cheek_data and right_cheek_data:
                    # Average RGB values from all regions
                    frame_data = {
                        'red_mean': (forehead_data['red_mean'] + left_cheek_data['red_mean'] + right_cheek_data['red_mean']) / 3,
                        'green_mean': (forehead_data['green_mean'] + left_cheek_data['green_mean'] + right_cheek_data['green_mean']) / 3,
                        'blue_mean': (forehead_data['blue_mean'] + left_cheek_data['blue_mean'] + right_cheek_data['blue_mean']) / 3,
                        'timestamp': self.timestamp_ms
                    }
                    
                    # Store frame data
                    self.frames_data.append(frame_data)
                    if len(self.frames_data) > PPG_WINDOW_SIZE:
                        self.frames_data.pop(0)
                    
                    # Calculate heart rate for selected method
                    selected_method = self.selected_method.get()
                    hr, quality, _ = self.process_ppg_signal(selected_method, self.frames_data)
                    
                    # For demonstration, calculate HR for all methods
                    for method in DETECTION_METHODS:
                        if method != selected_method and len(self.frames_data) > 60:  # Process others with delay
                            method_hr, method_quality, _ = self.process_ppg_signal(method, self.frames_data)
                            self.method_results[method].config(
                                text=f"{method}: {method_hr:.0f} BPM (Quality: {method_quality:.2f})"
                            )
                    
                    # Update display
                    if hr > 0:
                        self.hr_label.config(text=f"Heart Rate: {hr:.0f} BPM")
                        self.quality_label.config(text=f"Signal Quality: {quality:.2f}")
                        
                        # Display heart rate on the frame
                        cv2.putText(frame, f"HR: {hr:.0f} BPM", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(frame, f"Quality: {quality:.2f}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Update method result
                        self.method_results[selected_method].config(
                            text=f"{selected_method}: {hr:.0f} BPM (Quality: {quality:.2f})"
                        )
                    else:
                        cv2.putText(frame, "Calculating...", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                self.hr_label.config(text="Heart Rate: -- BPM")
                self.quality_label.config(text="Signal Quality: --")
            
            # Update FPS display
            self.fps_label.config(text=f"FPS: {self.smoothed_fps:.1f}")
            
            # Display the frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            # Schedule next frame update
            self.window.after(10, self.update_frame)
            
        except Exception as e:
            print(f"Error in update_frame: {str(e)}")
            self.window.after(100, self.update_frame)  # Try again after a short delay
    
    def on_closing(self):
        """Clean up resources when closing the application."""
        if hasattr(self, 'cap'):
            self.cap.release()
        self.window.destroy()

def main():
    """Main function to start the application."""
    root = tk.Tk()
    app = FaceDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()