import os
import requests

# URL for the manifest file
manifest_url = "https://storage.googleapis.com/tfjs-models/weights/posenet/mobilenet_v1_050/manifest.json"

# Directory to save the downloaded files
save_dir = "posenet_model"
os.makedirs(save_dir, exist_ok=True)

# Download the manifest file
manifest_response = requests.get(manifest_url)
manifest_data = manifest_response.json()

# Base URL for the weight files
base_url = os.path.dirname(manifest_url)

# Download each weight file listed in the manifest
for tensor_name, tensor_info in manifest_data.items():
    weight_file = tensor_info['filename']
    weight_url = f"{base_url}/{weight_file}"
    weight_response = requests.get(weight_url)
    weight_path = os.path.join(save_dir, weight_file)
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)
    with open(weight_path, "wb") as f:
        f.write(weight_response.content)

print("Download complete!")
