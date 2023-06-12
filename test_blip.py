from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder
import os
import evaluate
import os
import numpy as np
import json
import argparse
import pandas as pd
from tqdm import tqdm

def main():  
    rouge = evaluate.load("rouge")
    meteor = evaluate.load('meteor')
    bleu = evaluate.load("bleu")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='BLIP test script')
    parser.add_argument('--benchmark_path', help='Path to the benchmark dataset')
    parser.add_argument('--model_path', help='Path to the model')
    args = parser.parse_args()

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
    
    model_path = args.model_path
    model = blip_decoder(pretrained=model_path, vit='base')
    model.eval()
    model = model.to(device)
    batch_size = 128
    benchmark_path = args.benchmark_path
    test_json_path = os.path.join(benchmark_path, 'No-Subfig-Img','test/metadata.jsonl')    
    json_df = pd.read_json(test_json_path, lines= True) 
    json_df['file_name'] = os.path.join(benchmark_path, "No-Subfig-Img", "test") + "/" + json_df['file_name']
    captions_list = json_df['text'].tolist()
    image_list = json_df['file_name'].tolist()
    
    # test_json = json.load(open(test_json_path, 'rb'), lines = True)
    for i in tqdm(range(0, len(captions_list)-batch_size+1, batch_size)):
        gt_captions = [captions_list[i+k] for k in range(batch_size)]
        image_paths = [image_list[i+k] for k in range(batch_size)]
        image = torch.cat([load_demo_image(image_path=image_path, device=device) for image_path in image_paths])
        with torch.no_grad():
            caption = model.generate(image, sample=True, top_p=0.7, max_length=512, min_length=10)
            rouge.add_batch(predictions=caption, references=gt_captions)
            meteor.add_batch(predictions=caption, references=gt_captions)
            bleu.add_batch(predictions=caption, references=gt_captions)

    print('ROGUE:', rouge.compute())
    print('METEOR:', meteor.compute())
    print('BLEU:', bleu.compute())


if __name__ == '__main__':
    main()

