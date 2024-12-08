from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import cv2
import numpy as np
from collections import Counter
import threading

import os
from supabase import create_client, Client

url: str = "https://zpsxvshzdetlcqdsnhbq.supabase.co"
key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inpwc3h2c2h6ZGV0bGNxZHNuaGJxIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczMzYzMTYyMCwiZXhwIjoyMDQ5MjA3NjIwfQ.4k2VnZqp6F-Y9RgnW8BpFYBdTAXHDn2OOrYYwdjrbZ8"
supabase: Client = create_client(url, key)

def save_alert_to_db(camera_uuid, alert_data):
    response = supabase.table("alerts").insert({
        "stream": camera_uuid,
        "alert": alert_data
    }).execute()

# Function to process a single camera stream
def process_camera(video_path, camera_id, batch_length=60, fps_target=1):
    print(f"Starting camera {camera_id} - {video_path}")

    # Load DETR model and processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model.eval()

    # Open the video stream
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open stream for camera {camera_id}")
        return

    # Get original FPS of the video
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_to_skip = max(original_fps // fps_target, 1)  # Process at approximately target FPS

    frame_count = 0
    buffer = []  # Temporary storage for object counts
    averaged_results = []  # Store averaged results

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Stream ended for camera {camera_id}")
            break

        # Process every nth frame
        if frame_count % frames_to_skip == 0:
            # Convert frame (BGR to RGB for PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Prepare inputs and make predictions
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process the results
            target_sizes = torch.tensor([image.size[::-1]])  # Image size in (width, height)
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

            # Count objects by type
            object_counts = Counter()
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                label_name = model.config.id2label[label.item()]
                object_counts[label_name] += 1

            # Add object counts to the buffer
            buffer.append(object_counts)

            # If the buffer is full, calculate averages
            if len(buffer) == batch_length:
                # Compute average counts
                total_counts = Counter()
                for counts in buffer:
                    total_counts.update(counts)

                average_counts = {key: total / batch_length for key, total in total_counts.items()}
                averaged_results.append(average_counts)

                save_alert_to_db(camera_id, average_counts)
                # Print the results for the current batch
                print(f"Camera {camera_id}: Average for frames {frame_count - batch_length + 1} to {frame_count}:")
                print(average_counts)

                # Clear the buffer for the next batch
                buffer = []

        frame_count += 1

    # Release resources
    cap.release()
    print(f"Camera {camera_id}: Processing completed.")

# Multi-camera support
if __name__ == "__main__":
    # Fetch camera links from the database
    response = supabase.table("streams").select("uuid, url").execute()
    camera_links = [{"uuid": item["uuid"], "url": item["url"]} for item in response.data]

    if not camera_links:
        print("No camera links found in the database.")
    else:
        # Create threads for each camera
        threads = []
        for camera in camera_links:
            t = threading.Thread(target=process_camera, args=(camera["url"], camera["uuid"]))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

    print("All cameras processed.")


