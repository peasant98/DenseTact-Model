import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the encoder type (e.g., "vit" for Vision Transformer)
encoder = "vitb"  # Other options include "vitb", "vitl", "vitg"
import pdb; pdb.set_trace()
# 1. Load the pre-trained model
model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{encoder}14')
model.eval()  # Set to evaluation mode

# 2. Prepare the transform pipeline for preprocessing
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load and preprocess a sample image
image_path = "/home/arm-beast/Desktop/DenseTact-Model/king.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# 4. Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)

# 5. Perform inference
with torch.no_grad():  # No need to track gradients
    features = model(input_tensor)

# 6. Process the output
# For feature extraction (you get the CLS token by default)
feature_vector = features.squeeze().cpu().numpy()

print(f"Feature vector shape: {feature_vector.shape}")

# 7. Optional: Visualize the features (first few dimensions)
plt.figure(figsize=(10, 5))
plt.bar(range(20), feature_vector[:20])
plt.title("First 20 dimensions of the feature vector")
plt.xlabel("Dimension")
plt.ylabel("Value")
plt.show()

# 8. Example: Use features for a downstream task (e.g., similarity comparison)
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# If you have another image to compare:
# image2 = Image.open("path/to/second/image.jpg").convert('RGB')
# input_tensor2 = transform(image2).unsqueeze(0).to(device)
# with torch.no_grad():
#     features2 = model(input_tensor2)
# feature_vector2 = features2.squeeze().cpu().numpy()
# similarity = cosine_similarity(feature_vector, feature_vector2)
# print(f"Similarity between images: {similarity}")