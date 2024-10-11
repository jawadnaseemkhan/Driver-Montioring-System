import os
import cv2
import asyncio
from lane_assist import process_road_videos
from facial_expressions import detect_drowsiness_and_emotions

async def process_videos():
    # Directory containing the road video subfolders
    road_video_directory = '/lhome/jawakha/Desktop/Project/Dataset/DREYEVE_DATA'
    
    # Process road videos for lane detection
    process_road_videos(road_video_directory)

async def monitor_driver():
    # Start the webcam for driver monitoring
    cap = cv2.VideoCapture(0)  # Open the webcam
    frame_counter = 0

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Pass the frame and frame counter to detect_drowsiness_and_emotions
        detect_drowsiness_and_emotions(frame, frame_counter)

        # Display the frame
        cv2.imshow("Driver Monitoring", frame)

        frame_counter += 1

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Yield control back to the event loop for a short while to keep it responsive
        await asyncio.sleep(0.01)

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

async def main():
    # Run both tasks concurrently
    await asyncio.gather(
        process_videos(),
        monitor_driver(),
    )

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
