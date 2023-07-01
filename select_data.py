# This script moves data based on cvs metadata file

import pandas as pd
import shutil
import os

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(r"C:\Users\rober\Downloads\archive\HAM10000_metadata.csv")

# Set the source directories where images are currently located
source_dirs = [
    r"C:\Users\rober\Downloads\archive\HAM10000_images_part_1",
    r"C:\Users\rober\Downloads\archive\HAM10000_images_part_2",
]

# Set the target directories
target_dir_benign = r"C:\Users\rober\Documents\GitHub\skinvision\data\train\benign"
target_dir_malignant = r"C:\Users\rober\Documents\GitHub\skinvision\data\train\malignant"

# Create target directories if they don't exist
os.makedirs(target_dir_benign, exist_ok=True)
os.makedirs(target_dir_malignant, exist_ok=True)

# Iterate through the DataFrame rows
for idx, row in df.iterrows():
    # Get the image filename
    filename = row['image_id'] + ".jpg"

    # Determine the source path for this file
    for dir in source_dirs:
        source_path = os.path.join(dir, filename)
        
        # If the file exists in the current source directory, break the loop
        if os.path.isfile(source_path):
            break

    # Determine the target path for this file and move it accordingly
    if row['dx'] == 'nv':
        target_path = os.path.join(target_dir_benign, filename)
    else:
        target_path = os.path.join(target_dir_malignant, filename)
    
    # Move the file
    shutil.move(source_path, target_path)
