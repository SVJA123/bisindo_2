import os
import shutil

# To combine the folder in source folder to the destination folder

def combine_folders(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        relative_path = os.path.relpath(root, source_folder)
        
        dest_dir = os.path.join(destination_folder, relative_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            
            if os.path.exists(dest_file):
                base, extension = os.path.splitext(file)
                counter = 1
                new_dest_file = os.path.join(dest_dir, f"{base}_{counter}{extension}")
                while os.path.exists(new_dest_file):
                    counter += 1
                    new_dest_file = os.path.join(dest_dir, f"{base}_{counter}{extension}")
                dest_file = new_dest_file
            
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

if __name__ == "__main__":
    source_folder = "data"
    destination_folder = "frames"
    
    combine_folders(source_folder, destination_folder)
    print("Folders combined successfully.")