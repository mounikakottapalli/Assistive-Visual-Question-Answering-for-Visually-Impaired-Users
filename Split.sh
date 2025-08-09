#!/bin/bash

# Define source, train, and test folders
source_dir="all_sorte"
train_dir="train_videos"
test_dir="test_videos"

# Set the train/test split ratio (e.g., 0.8 = 80% train, 20% test)
train_ratio=0.8

# Convert paths to absolute
source_dir="$(realpath "$source_dir")"
train_dir="$(realpath "$train_dir")"
test_dir="$(realpath "$test_dir")"

# Remove existing train and test folders if they exist (optional)
rm -rf "$train_dir" "$test_dir"

# Create new train and test folders
mkdir -p "$train_dir"
mkdir -p "$test_dir"

# Loop through each subfolder (label/class) in the source directory
for folder in "$source_dir"/*; do
    if [ -d "$folder" ]; then
        label="$(basename "$folder")"
        echo "Processing label: $label"

        # Create subfolders in train and test
        mkdir -p "$train_dir/$label"
        mkdir -p "$test_dir/$label"

        # Get list of all files in the label folder
        files=("$folder"/*)
        total=${#files[@]}

        # Calculate how many go to train
        train_count=$(echo "$total * $train_ratio" | bc | awk '{print int($1)}')

        # Shuffle files
        shuffled=($(printf "%s\n" "${files[@]}" | shuf))

        # Copy to train folder
        for ((i = 0; i < train_count; i++)); do
            cp "${shuffled[$i]}" "$train_dir/$label/"
        done

        # Copy the rest to test folder
        for ((i = train_count; i < total; i++)); do
            cp "${shuffled[$i]}" "$test_dir/$label/"
        done
    fi
done

echo "✅ Done splitting $source_dir into $train_dir and $test_dir"