import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from scipy import signal
import time

# --- MediaPipe Classes ---
BaseOptions = python.BaseOptions
FaceLandmarker = python.vision.FaceLandmarker
FaceLandmarkerOptions = python.vision.FaceLandmarkerOptions
VisionRunningMode = python.vision.RunningMode

# --- Constants ---
MODEL_PATH = "face_landmarker.task"  # Path to your MediaPipe model
PPG_WINDOW_SIZE = 360  # Number of samples for PPG history
SIGNAL_QUALITY_THRESHOLD = 0.5 # Starting threshold - adjust as needed!
HEART_RATE_SMOOTHING_WINDOW = 5  # Window size for heart rate smoothing
MIN_HR = 30  # Minimum plausible heart rate
MAX_HR = 240  # Maximum plausible heart rate
FRAME_RATE_WINDOW = 10  # Window for averaging the frame rate

# --- Pan-Tompkins Algorithm ---
def pan_tompkins(ppg_signal, fs):
    """
    Pan-Tompkins algorithm for PPG signals, with dynamic thresholding and SNR.
    """
    if len(ppg_signal) < 2:  # Not enough data
        return [], 0, [], 0

    ppg_signal = np.array(ppg_signal)  # Ensure NumPy array
    ppg_signal = ppg_signal - np.mean(ppg_signal)

    # --- Handle Low Sampling Rate ---
    if fs <= 8.0:
        return [], 0, [], 0  # Return 0 heart rate if fs is too low

    # --- Bandpass Filter ---
    nyquist = fs / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, ppg_signal)

    # --- Derivative, Squaring, and Integration ---
    derivative = np.diff(filtered_signal)
    derivative = np.append(derivative, derivative[-1])  # Match length
    squared = derivative ** 2

    window_size = int(0.1 * fs)
    window_size = max(1, window_size)  # Ensure window_size is at least 1
    if window_size % 2 == 0:
        window_size += 1
    window = np.ones(window_size) / window_size
    integrated = np.convolve(squared, window, mode='same')

    # --- Dynamic Thresholding ---
    std_window_size = int(0.5 * fs)  # 0.5-second window for std
    std_window_size = max(1, std_window_size)
    if std_window_size % 2 == 0:
        std_window_size += 1

    padded_integrated = np.pad(integrated, (std_window_size // 2, std_window_size // 2), mode='edge')
    moving_std = np.convolve(np.abs(padded_integrated - np.mean(padded_integrated)), np.ones(std_window_size) / std_window_size, mode='valid')
    threshold = np.mean(integrated) + 0.7 * moving_std  # Adjust multiplier as needed

    # --- Peak Detection ---
    min_distance = int(0.5 * fs)
    min_distance = max(1, min_distance)
    peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)

    # --- Calculate Heart Rate ---
    heart_rate = 0
    if len(peaks) > 1:
        peak_intervals = np.diff(peaks)
        mean_interval = np.mean(peak_intervals)
        heart_rate = 60 * fs / mean_interval
        if heart_rate < MIN_HR or heart_rate > MAX_HR:
            heart_rate = 0

    # --- Signal Quality (SNR) - Corrected Calculation ---
    frequencies, power_spectrum = signal.welch(filtered_signal, fs=fs, nperseg=min(len(filtered_signal), 256))
    heart_rate_band_indices = np.where((frequencies >= 0.5) & (frequencies <= 4.0))[0]

    delta_f = frequencies[1] - frequencies[0]  # Bandwidth of each frequency bin
    signal_power = np.sum(power_spectrum[heart_rate_band_indices]) * delta_f
    total_power = np.sum(power_spectrum) * delta_f
    snr = signal_power / (total_power - signal_power) if (total_power - signal_power) > 0 else 0
    # snr = 10 * np.log10(signal_power / (total_power - signal_power)) if (total_power - signal_power) > 0 else 0 #SNR in dB

    if snr < SIGNAL_QUALITY_THRESHOLD:
        heart_rate = 0

    return peaks.tolist(), heart_rate, filtered_signal.tolist(), snr


# --- GUI Class ---
class FaceDetectionGUI:
    def __init__(self, window):
        self.window = window
        self.window.title("Camera Input")

        # --- GUI Layout ---
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(expand=True, fill='both')

        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(expand=True, fill='both')

        self.status_label = ttk.Label(self.main_frame, text="Starting...", font=("Arial", 12))
        self.status_label.pack()

        # --- Facial Regions (Landmark Indices) ---
        self.region_points = [103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336, 285, 417, 351, 419, 197, 196, 122, 193, 55, 107, 66, 105, 104, 103]
        self.region_points2 = [116, 111, 117, 118, 119, 120, 100, 142, 36, 205, 187, 123]
        self.region_points3 = [345, 340, 346, 347, 348, 349, 329, 371, 266, 425, 411, 352]

        # --- PPG Data Storage ---
        self.region1_green_values = []
        self.region2_green_values = []
        self.region3_green_values = []

        # --- Heart Rate Variables ---
        self.current_hr = 0
        self.heart_rate_history = []

        # --- Frame Rate and Timing ---
        self.frame_times = []          # Raw frame timestamps
        self.fps_history = []          # FPS values for smoothing
        self.smoothed_fps = 0.0        # Smoothed FPS
        self.last_timestamp_ms = 0     # Last valid MediaPipe timestamp
        self.fallback_fps = 30         # Default FPS if calculation fails
        self.previous_perf_counter = time.perf_counter()  # For accurate delta_t

        # --- MediaPipe Setup ---
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,  # Use VIDEO mode
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0  # Initial timestamp

        # --- Start Camera and GUI ---
        self.start_camera()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def start_camera(self):
        """Initializes the camera."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Error: Cannot open camera")
            return
        self.update_frame()

    def highlight_region(self, frame, face_landmarks, region_points):
        """Highlights a facial region and extracts green channel values."""
        frame_copy = frame.copy()  # Work on a copy
        points = []
        for point_idx in region_points:
            landmark = face_landmarks[point_idx]
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            points.append([x, y])
        points = np.array(points, dtype=np.int32)

        # --- Create Mask and Extract Green Values ---
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255)
        green_channel = frame[:, :, 1]  # BGR, so index 1 is green
        region_green_values = green_channel[mask == 255]

        # --- Store Green Values (All Regions) ---
        if np.array_equal(region_points, self.region_points):
            self.region1_green_values.extend(region_green_values)
            if len(self.region1_green_values) > PPG_WINDOW_SIZE:
                self.region1_green_values = self.region1_green_values[-PPG_WINDOW_SIZE:]
        elif np.array_equal(region_points, self.region_points2):
            self.region2_green_values.extend(region_green_values)
            if len(self.region2_green_values) > PPG_WINDOW_SIZE:
                self.region2_green_values = self.region2_green_values[-PPG_WINDOW_SIZE:]
        elif np.array_equal(region_points, self.region_points3):
            self.region3_green_values.extend(region_green_values)
            if len(self.region3_green_values) > PPG_WINDOW_SIZE:
                self.region3_green_values = self.region3_green_values[-PPG_WINDOW_SIZE:]

        # --- Draw Region Outline ---
        cv2.polylines(frame_copy, [points], True, (255, 0, 0), 2)  # Blue outline
        return frame_copy

    def update_frame(self):
        """Processes each frame: detects face, extracts PPG, calculates HR."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="Error: Cannot read frame")
                return

            # --- Measure Frame Rate and Time Delta (Modified) ---
            current_perf_counter = time.perf_counter()
            delta_t_ms = int((current_perf_counter - self.previous_perf_counter) * 1000)

            # Enforce Monotonicity
            if delta_t_ms <= 0:
                delta_t_ms = int(1000 / self.fallback_fps)  # Use fallback increment

            self.timestamp_ms += delta_t_ms
            self.last_timestamp_ms = self.timestamp_ms  # Update last valid timestamp
            self.previous_perf_counter = current_perf_counter # Store for next frame

            # --- FPS Calculation (for display/sanity check) ---
            self.frame_times.append(current_perf_counter)
            if len(self.frame_times) > FRAME_RATE_WINDOW:
                self.frame_times.pop(0)
                time_diffs = np.diff(self.frame_times)
                if np.all(time_diffs > 0):
                    epsilon = 1e-9
                    mean_time_diff = np.mean(time_diffs)
                    if mean_time_diff > epsilon:
                        current_fps = 1.0 / mean_time_diff
                        self.fps_history.append(current_fps)
                        if len(self.fps_history) > FRAME_RATE_WINDOW:
                            self.fps_history.pop(0)
                        self.smoothed_fps = np.mean(self.fps_history)
                    else:
                        self.smoothed_fps = 0.0

            # --- MediaPipe Face Landmark Detection ---
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)

            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]

                # --- Highlight and Process Regions ---
                frame = self.highlight_region(frame, face_landmarks, self.region_points)
                frame = self.highlight_region(frame, face_landmarks, self.region_points2)
                frame = self.highlight_region(frame, face_landmarks, self.region_points3)

                # --- Draw Landmarks (Optional) ---
                for landmark in face_landmarks:
                    cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 1, (0, 255, 0), -1)

                # --- Combine PPG Signals (Averaging) ---
                if len(self.region1_green_values) > 0 and len(self.region2_green_values) > 0 and len(self.region3_green_values) > 0:
                    min_len = min(len(self.region1_green_values), len(self.region2_green_values), len(self.region3_green_values))
                    region1_trimmed = self.region1_green_values[-min_len:]
                    region2_trimmed = self.region2_green_values[-min_len:]
                    region3_trimmed = self.region3_green_values[-min_len:]
                    combined_ppg = np.mean([region1_trimmed, region2_trimmed, region3_trimmed], axis=0)

                    # --- Calculate Heart Rate (Use Smoothed FPS) ---
                    peaks, heart_rate, _, snr = pan_tompkins(combined_ppg, self.smoothed_fps if self.smoothed_fps > 8.0 else self.fallback_fps)

                    # --- Heart Rate Smoothing ---
                    if heart_rate > 0:
                        self.heart_rate_history.append(heart_rate)
                        if len(self.heart_rate_history) > HEART_RATE_SMOOTHING_WINDOW:
                            self.heart_rate_history.pop(0)
                        self.current_hr = np.mean(self.heart_rate_history)

                    # --- Display Results ---
                    if self.current_hr > 0:
                        cv2.putText(frame, f"HR: {self.current_hr:.0f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"SNR: {snr:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Display SNR
                    self.status_label.config(text=f"FPS: {self.smoothed_fps:.2f}")

                else:
                    self.status_label.config(text="Initializing PPG...")

            else:
                # --- Handle Face Loss ---
                self.status_label.config(text="No face detected")
                self.current_hr = 0  # Reset heart rate
                self.heart_rate_history = []  # Clear history

            # --- Update Display ---
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # --- Schedule Next Frame Update ---
            self.window.after(10, self.update_frame)  # 10ms delay

        except Exception as e:
            print(f"Error in update_frame: {str(e)}")
            self.status_label.config(text=f"Error: {str(e)}")
            if hasattr(self, 'cap'):  # Check if 'cap' exists before releasing
                self.cap.release()
            self.start_camera()  # Attempt to restart

    def on_closing(self):
        """Releases resources when the window is closed."""
        if hasattr(self, 'cap'):
            self.cap.release()
        self.window.destroy()

# --- Main Function ---
def main():
    root = tk.Tk()
    app = FaceDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()