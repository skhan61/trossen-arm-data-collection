"""
Test USB camera (Intel RealSense D405).
"""

import cv2

def main():
    # Try different video devices
    for device_id in [0, 2, 4]:
        print(f"Trying /dev/video{device_id}...")
        cap = cv2.VideoCapture(device_id)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera found on /dev/video{device_id}")
                print(f"  Frame size: {frame.shape}")

                # Save a test image
                filename = f"test_camera_{device_id}.jpg"
                cv2.imwrite(filename, frame)
                print(f"  Saved test image: {filename}")
            cap.release()
        else:
            print(f"  Could not open /dev/video{device_id}")

    print("\nTo view live camera feed, run:")
    print("  python view_camera.py")

if __name__ == "__main__":
    main()
