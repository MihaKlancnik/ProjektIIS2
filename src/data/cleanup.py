import os
import time

def cleanup_data_folder(folder_path, max_age_days):
    """Delete files older than max_age_days in the specified folder."""
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 60 * 60

    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_age = current_time - os.path.getmtime(file_path)

            if file_age > max_age_seconds:
                print(f"Deleting old file: {file_path}")
                os.remove(file_path)

if __name__ == "__main__":
    data_folder = os.path.join(os.path.dirname(__file__), "../../data")
    cleanup_data_folder(data_folder, max_age_days=70)