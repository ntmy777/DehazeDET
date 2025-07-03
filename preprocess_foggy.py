import os

# Directory containing the images
directory = r'./test_foggy/JPEGImages'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if '_leftImg8bit' in filename:
        # Construct the full old and new file paths
        old_file_path = os.path.join(directory, filename)
        new_filename = filename.replace('_leftImg8bit', '')
        new_file_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f'Renamed: {old_file_path} to {new_file_path}')

print("Finished renaming files.")
