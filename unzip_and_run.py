import os
import zipfile
import shutil
import time

workspace_dir = "/Users/ashwanikumar/code/Goal scoring probability ML/probabiltiy code/offside-football-prediction"
search_dirs = [
    "/Users/ashwanikumar/Downloads",
    "/Users/ashwanikumar/Desktop",
    workspace_dir
]

target_files = ['train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv', 'feature_catalog.csv']

def check_files_present():
    return all(os.path.exists(os.path.join(workspace_dir, f)) for f in target_files)

def discover_and_extract():
    if check_files_present():
        print("All target files are already in the workspace.")
        return True

    # 1. Search for a zip file
    for s_dir in search_dirs:
        for f in os.listdir(s_dir):
            if f.endswith('.zip') and ('offside' in f.lower() or 'datathon' in f.lower() or 'archive' in f.lower()):
                zip_path = os.path.join(s_dir, f)
                print(f"Found dataset zip file: {zip_path}")
                print("Extracting files to workspace...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(workspace_dir)
                    print("Extraction complete!")
                    return True
                except Exception as e:
                    print(f"Error extracting zip: {e}")
                    
    # 2. Search for individual CSV files
    found_csvs = {}
    for target in target_files:
        for s_dir in search_dirs:
            csv_path = os.path.join(s_dir, target)
            if os.path.exists(csv_path):
                found_csvs[target] = csv_path
                break
                
    if len(found_csvs) == len(target_files):
        print("Found all individual CSV files! Copying them to workspace...")
        for name, path in found_csvs.items():
            dest = os.path.join(workspace_dir, name)
            if path != dest:
                shutil.copy(path, dest)
                print(f"  Copied {name}")
        return True
        
    return False

def main():
    print("=== Dataset Discovery and Pipeline Runner ===")
    print("Scanning for dataset files...")
    
    if discover_and_extract():
        print("Dataset ready. Launching machine learning pipeline...")
        os.system(".venv/bin/python main.py")
    else:
        print("\n[INFO] Dataset files not found yet.")
        print("Where to find the download button on Kaggle:")
        print("  1. In the browser window you have open, look at the right side of the screen under 'Data Explorer'.")
        print("  2. You will see the file list: train.csv, test.csv, etc. and the total size 972.89 MB.")
        print("  3. Click the download icon (down arrow in a bracket) next to the size to download all files as a zip,")
        print("     or hover over each file and click the individual download icons.")
        print("\nWaiting for the dataset to appear in your Downloads/Desktop folders...")
        print("We will poll every 10 seconds. You can download the dataset now.")
        
        try:
            while True:
                time.sleep(10)
                if discover_and_extract():
                    print("\nDataset detected! Launching machine learning pipeline...")
                    os.system(".venv/bin/python main.py")
                    break
        except KeyboardInterrupt:
            print("\nPolling stopped.")

if __name__ == "__main__":
    main()
