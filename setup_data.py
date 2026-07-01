import os
import shutil
import zipfile
import pandas as pd

def main():
    src_dir = "/Users/ashwanikumar/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/Message/Media/112696183775404@lid"
    dest_dir = "/Users/ashwanikumar/code/Goal scoring probability ML/probabiltiy code/offside-football-prediction"
    
    files_map = {
        "0/0/0031068a-a9ef-4d53-bc97-3dd556e4aec9.csv": "data_dictionary.csv",
        "5/1/51fdfd4d-10db-49f9-a90d-d262db458dc6.csv": "feature_catalog.csv",
        "d/5/d576df31-ddbe-47e0-afd3-9cabe4e42781.csv": "sample_submission.csv"
    }
    
    print("=== Copying Metadata Files ===")
    for rel_path, dest_name in files_map.items():
        src_path = os.path.join(src_dir, rel_path)
        dest_path = os.path.join(dest_dir, dest_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
            print(f"  Copied {dest_name} to workspace.")
        else:
            print(f"  [Warning] Source file not found: {src_path}")
            
    print("\n=== Unzipping train_cleaned.csv ===")
    zip_path = os.path.join(src_dir, "f/3/f3e8f4c7-127a-429b-bd3f-7b2a7b1d71c3.zip")
    train_dest = os.path.join(dest_dir, "train.csv")
    
    if os.path.exists(zip_path):
        print(f"  Extracting train_cleaned.csv from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract to a temp directory and move/rename to train.csv
            zf.extract("train_cleaned.csv", dest_dir)
            
        extracted_path = os.path.join(dest_dir, "train_cleaned.csv")
        if os.path.exists(extracted_path):
            if os.path.exists(train_dest):
                os.remove(train_dest)
            os.rename(extracted_path, train_dest)
            print("  Successfully set up train.csv.")
        else:
            print("  [Error] Extraction failed to locate train_cleaned.csv.")
    else:
        print(f"  [Error] ZIP file not found: {zip_path}")
        
    print("\n=== Creating Dummy test.csv if missing ===")
    test_path = os.path.join(dest_dir, "test.csv")
    if not os.path.exists(test_path):
        if os.path.exists(train_dest):
            print("  Generating dummy test.csv from a sample of train.csv...")
            train_df = pd.read_csv(train_dest, nrows=100)
            if "scored_flag" in train_df.columns:
                test_df = train_df.drop(columns=["scored_flag"])
            else:
                test_df = train_df
            # Add dummy appearance_id if missing (we want to simulate test data)
            if "appearance_id" not in test_df.columns:
                test_df["appearance_id"] = [f"dummy_{i}" for i in range(len(test_df))]
            test_df.to_csv(test_path, index=False)
            print("  Dummy test.csv created successfully with 100 rows.")
        else:
            print("  [Error] Cannot create dummy test.csv because train.csv is missing.")
            
if __name__ == "__main__":
    main()
