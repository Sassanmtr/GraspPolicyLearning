#!/bin/bash

# Get the path of the directory containing the bash script
script_directory=$(dirname "$(readlink -f "$0")")

# Move up one directory
parent_directory=$(dirname "$script_directory")

echo "Parent directory: $parent_directory"

# Construct the directory path relative to the parent directory
directory="$parent_directory/bc_files/ValidGraspObjects"

# Set the number of trajectories
nr_traj=50

# Loop through each object in the directory
for folder in "$directory"/*; do
    if [ -d "$folder" ]; then
        # Get the folder name
        folder_name=$(basename "$folder")

        # Print a message indicating the current folder
        echo "Processing folder: $folder_name"

        # Perform the desired task for each object
        python -c "import sys; sys.path.append('$script_directory'); import new_env_various_collector; new_env_various_collector.main('$folder_name', $nr_traj)"
    fi
done

