import os
import shutil
from pathlib import Path

def organize_dataset(base_dir):
    # Define the classes
    classes = ['A', 'L', 'N', 'R', 'V']
    sets = ['train', 'validate', 'test']
    
    # Process each set (train, validate, test)
    for set_name in sets:
        set_path = os.path.join(base_dir, set_name, 'images')
        if not os.path.exists(set_path):
            print(f"Warning: {set_path} does not exist")
            continue
            
        # Create class directories if they don't exist
        for class_name in classes:
            class_dir = os.path.join(base_dir, set_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
        
        # Move files to their respective class folders
        for filename in os.listdir(set_path):
            if filename.endswith('.png'):
                # Extract class from filename (first character)
                class_name = filename[0]
                if class_name in classes:
                    src_path = os.path.join(set_path, filename)
                    dst_path = os.path.join(base_dir, set_name, class_name, filename)
                    shutil.move(src_path, dst_path)
                    print(f"Moved {filename} to {class_name} folder in {set_name}")

def print_image_stats(base_dir):
    classes = ['A', 'L', 'N', 'R', 'V']
    sets = ['train', 'validate', 'test']
    print("\nImage count per class:")
    for set_name in sets:
        print(f"\nSet: {set_name}")
        for class_name in classes:
            class_dir = os.path.join(base_dir, set_name, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            else:
                count = 0
            print(f"  Class {class_name}: {count} images")

def print_percentages(base_dir):
    classes = ['A', 'L', 'N', 'R', 'V']
    sets = ['train', 'validate', 'test']
    print("\nPercentage of images per class:")
    for set_name in sets:
        print(f"\nSet: {set_name}")
        total = 0
        counts = {}
        for class_name in classes:
            class_dir = os.path.join(base_dir, set_name, class_name)
            if os.path.exists(class_dir):
                counts[class_name] = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
            else:
                counts[class_name] = 0
            total += counts[class_name]
        for class_name in classes:
            percent = (counts[class_name] / total) * 100 if total > 0 else 0
            print(f"  Class {class_name}: {percent:.2f}% ({counts[class_name]} / {total})")

if __name__ == "__main__":
    base_dir = "dataset"  # change this to the path of the dataset
    organize_dataset(base_dir)
    #Comment out the functions you don't want to use
    # print_image_stats(base_dir) # print the number of images in each class
    # print_percentages(base_dir) 