import argparse
import sys
import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

# Set the seed for random module
seed = 42
random.seed(seed)

# Set the seed for Numpy
np.random.seed(seed)

# Set the seed for PyTorch
torch.manual_seed(seed)

# If you're using GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Add a new directory to the system path
def main():
    parser = argparse.ArgumentParser(description='BLIP with RLHF inference script')
    parser.add_argument('--figure_path', help='Path to the figure image')
    parser.add_argument('--model_path', help='Path to the downloaded model')
    args = parser.parse_args()

    file_path = args.figure_path
    
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

    import torch
    from PIL import Image
    
    model_path = args.model_path
    model = blip_decoder(pretrained=model_path, vit='base')
    model.eval()
    
    def load_demo_image(image_path, device):
        raw_image = Image.open(image_path).convert('RGB')
        w,h = raw_image.size
        transform = transforms.Compose([
            transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])
        image = transform(raw_image).unsqueeze(0).to(device)
        return image
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    file_path = [file_path]
    image = torch.cat([load_demo_image(image_path=image_path, device=device) for image_path in file_path])
    caption = model.generate(image, sample=True, top_p=0.7, max_length=512, min_length=10)
    print("The caption for this figure is: " + caption[0] )
    return caption

if __name__ == '__main__':
    main()
