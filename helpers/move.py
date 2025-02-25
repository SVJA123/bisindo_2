import os
import shutil

# To move the sequence files from the 'frames' directory to the 'data' directory, as some of the images in frames are corrupted

def move_sequence_files(src_directory, dest_directory):
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    for letter in map(chr, range(65, 91)):  
        src_path = os.path.join(src_directory, letter)
        dest_path = os.path.join(dest_directory, letter)

        if os.path.exists(src_path):
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            
            for filename in os.listdir(src_path):
                if filename.startswith("sequence"):
                    src_file_path = os.path.join(src_path, filename)
                    dest_file_path = os.path.join(dest_path, filename)

                    shutil.move(src_file_path, dest_file_path)
                    print(f"Moved: {src_file_path} to {dest_file_path}")

src_directory = 'frames'
dest_directory = 'data'
move_sequence_files(src_directory, dest_directory)
