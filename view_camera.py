"""
View live camera feed from USB camera.
Press 'q' to quit, 's' to save a snapshot.
"""

import cv2
from datetime import datetime

def main():
    # RealSense cameras often use video0 for RGB
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera on /dev/video0, trying /dev/video2...")
        cap = cv2.VideoCapture(2)

    if not cap.isOpened():
        print("Could not open any camera!")
        return

    print("Camera opened successfully!")
    print("Press 'q' to quit, 's' to save snapshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Camera Feed", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
