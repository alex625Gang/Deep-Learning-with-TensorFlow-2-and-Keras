import os
import subprocess

# Specify the directory containing the .ipynb files,
# switch to your directory, e.x. D:/repo/Deep-Learning-with-TensorFlow-2-and-Keras/Chapter 7
directory = r'D:/repo/Deep-Learning-with-TensorFlow-2-and-Keras/Chapter 6'

# List all files in the directory
files = os.listdir(directory)

# Filter .ipynb files
ipynb_files = [file for file in files if file.endswith('.ipynb')]

# Convert each .ipynb file to .py
for ipynb_file in ipynb_files:
    input_path = os.path.join(directory, ipynb_file)
    output_path = os.path.splitext(input_path)[0]
    command = f'jupyter nbconvert --to script "{input_path}" --output "{output_path}"'
    subprocess.run(command, shell=True)