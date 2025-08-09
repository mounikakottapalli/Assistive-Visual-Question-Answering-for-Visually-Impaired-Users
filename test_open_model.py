import os
import sys

model_file = "retrained_graph.pb"

print(f"Current working directory: {os.getcwd()}")
print(f"Attempting to open file: {os.path.abspath(model_file)}")

try:
    with open(model_file, "rb") as f:
        print(f"File opened successfully! Reading first 10 bytes:")
        data = f.read(10)
        print(data)
except PermissionError as e:
    print(f"PermissionError: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")