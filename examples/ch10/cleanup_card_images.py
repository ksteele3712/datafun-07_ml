import os
import glob

# Folder to keep
main_folder = os.path.join('examples', 'ch10', 'card_images')
# Folder to clean
secondary_folder = os.path.join('examples', 'ch10', 'snippets_ipynb', 'card_images')

# Get all image files in secondary folder
image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.gif']
files_to_delete = []
for pattern in image_patterns:
    files_to_delete.extend(glob.glob(os.path.join(secondary_folder, pattern)))

print(f"Found {len(files_to_delete)} images to delete in {secondary_folder}.")

for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("Cleanup complete. All images removed from secondary folder.")
