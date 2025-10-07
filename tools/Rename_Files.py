import os
import sys

def rename_files(directory, start_number):
    """
    Rename all files in a directory with sequential numbers starting from start_number.
    
    Args:
        directory: Path to the directory containing files
        start_number: Starting number for renaming
    """
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return
    
    # Get all files in the directory (not subdirectories)
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    if not files:
        print("No files found in the directory.")
        return
    
    # Sort files to maintain consistent order
    files.sort()
    
    print(f"Found {len(files)} files. Starting renaming from {start_number}...\n")
    
    # Rename each file
    counter = start_number
    for filename in files:
        # Get file extension
        _, extension = os.path.splitext(filename)
        
        # Create new filename
        new_filename = f"{counter}{extension}"
        
        # Full paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_filename}")
        
        counter += 1
    
    print(f"\nSuccessfully renamed {len(files)} files!")

if __name__ == "__main__":
    # Check if correct number of arguments provided
    if len(sys.argv) != 3:
        print("Usage: python rename_files.py <directory_path> <start_number>")
        print("Example: python rename_files.py /path/to/folder 1")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    
    try:
        start_num = int(sys.argv[2])
    except ValueError:
        print("Error: Start number must be an integer.")
        sys.exit(1)
    
    rename_files(directory_path, start_num)