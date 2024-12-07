import os

def rename_images(directory, prefix):
    """
    Renames image files in the specified directory from 'train_real_<number>.jpg'
    to 'train_real_<prefix>_<number>.jpg'.

    Args:
        directory (str): Path to the directory containing the images.
        prefix (str): Prefix to insert before the number in the file names.
    """
    try:
        # Counter for processed files
        renamed_count = 0

        # Iterate through all files in the directory
        for file in os.listdir(directory):
            # Check if the file matches the naming format
            if file.startswith("train_real_") and file.endswith(".jpg"):
                # Split filename to extract the base and number
                parts = file.split("_")
                if parts[-1][:-4].isdigit():  # Ensure the last part is numeric (before .jpg)
                    # Construct the new file name
                    base_name = "_".join(parts[:-1])  # e.g., "train_real"
                    number = parts[-1]  # e.g., "1.jpg"
                    new_name = f"{base_name}_{prefix}_{number}"
                    # Rename the file
                    os.rename(os.path.join(directory, file), os.path.join(directory, new_name))
                    renamed_count += 1

        print(f"Renaming completed! Total files renamed: {renamed_count}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    directory_path = "./real_aditya"  # Replace with the directory containing your images
    prefix = "aditya"  # Replace with the desired prefix
    rename_images(directory_path, prefix)
