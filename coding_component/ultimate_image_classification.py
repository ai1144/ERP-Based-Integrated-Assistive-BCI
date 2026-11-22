# I used this tutorial for the transformers library: https://www.tutorialspoint.com/pytorch-torchvision-transforms-randomerasing 
# and this one as well: https://pytorch.org/vision/0.9/transforms.html 
# This was the tutorial for the PIL library: https://pillow.readthedocs.io/en/stable/handbook/tutorial.html 

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import os
from project_config import DATA_DIR
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# initialize model
model = resnet50(pretrained=True) 
model.eval()  #to eval mode

transform = transforms.Compose([ # image transformations
    transforms.Resize((256, 256)),  #256x256 pixels
    transforms.ToTensor(),  #image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #normalisation of images: I got this line from https://pytorch.org/vision/stable/transforms.html 
])
def extract_features(image_path): # extracting features from images
    image = Image.open(image_path).convert("RGB")  #conversion to rgb: this line was taken from https://stackoverflow.com/a/33507138 
    image = transform(image)
    with torch.no_grad():  #no gradient calculation
        features = model(image.unsqueeze(0))  #batch dimension and pass image through model
    return features # lines 25-26 were adapted from the audio classsification script which uses similar syntax

#calculate cosine similarity between two features
def calculate_similarity(feature1, feature2):
    return torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)
def load_images_from_folder(folder_path): # this was adapted from the ultimate_connect script
    features = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            features.append(extract_features(image_path))
    return torch.mean(torch.stack(features), dim=0) # this last line I got from this repository, suggestig the adidion of stack: https://github.com/pytorch/vision/issues/3657

#classify an image and print the maximum similarity score
def classify_image(image_path):
    image_feature = extract_features(image_path)
    category_features = {
        "ambulance": load_images_from_folder(str(DATA_DIR / 'image_categories/ambulance_category')),
        "unexpected pedestrian": load_images_from_folder(str(DATA_DIR / 'image_categories/unexpected_pedestrian_category')),
        "dashboard warning": load_images_from_folder(str(DATA_DIR / 'image_categories/dashboard_warning_category')),
        "nothing": load_images_from_folder(str(DATA_DIR / 'image_categories/nothing_category'))
    }
    similarities = {category: calculate_similarity(image_feature, feature).item() for category, feature in category_features.items()}
    max_category = max(similarities, key=similarities.get) # this part i adapted from the ultimate_connect script
    max_score = similarities[max_category]
    return max_category, max_score

# classify_image(str(DATA_DIR / 'examples/example_Ambulance/ambulance_input_image.jpeg'))

'''
print all similarities scores:
    for category, score in similarities.items():
        print(f"Image Similarity score for {category}: {score:.3f}")
    return max(similarities, key=similarities.get)  # return the category with the highest similarity

'''