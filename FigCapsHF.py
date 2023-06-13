from PIL import Image
import json
import os, sys
import pandas as pd
import requests
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, AutoModel, BlipForImageTextRetrieval
from sklearn.neural_network import MLPRegressor
from datasets import load_dataset 

class FigCapsHF():
    
    """
    Main class for FigCapsHF
    """

    def generate_jsonl(self):

        """
        Used to generate metadata.jsonl's for train/test/val and place it under their respective folders in No-Subfig-Img. Also verifies if dataset is consistent.
        """

        #Prepare the train dataset
        file_idx_path = os.path.join(self.benchmark_path,"List-of-Files-for-Each-Experiments","First-Sentence","No-Subfig","train","file_idx.json")
        captions_list = []
        f = open(file_idx_path)
        file_idx = json.load(f)
        for image_name in tqdm(file_idx):
            file_metadata_path = os.path.join(self.benchmark_path,"Caption-All","train",os.path.splitext(image_name)[0]+".json")
            c = open(file_metadata_path)
            file_metadata = json.load(c)
            captions_list.append({"file_name":file_metadata['figure-ID'],
                                  "text":file_metadata['1-lowercase-and-token-and-remove-figure-index']['caption'],  
                                  "human_feedback": file_metadata['human-feedback']})
            c.close()
        f.close()
        print("There are " + str(len(captions_list)) + " training samples")
        train_image_folder_path = os.path.join(self.benchmark_path,"No-Subfig-Img","train")+"/"
        with open(train_image_folder_path + "metadata.jsonl", 'w') as f:
            for caption in captions_list:
                f.write(json.dumps(caption) + "\n")
        f.close()
        #Prepare the val dataset
        file_idx_path = os.path.join(self.benchmark_path,"List-of-Files-for-Each-Experiments","First-Sentence","No-Subfig","val","file_idx.json")
        captions_list = []
        f = open(file_idx_path)
        file_idx = json.load(f)
        for image_name in tqdm(file_idx):
            file_metadata_path = os.path.join(self.benchmark_path,"Caption-All","val",os.path.splitext(image_name)[0]+".json")
            c = open(file_metadata_path)
            file_metadata = json.load(c)
            captions_list.append({"file_name":file_metadata['figure-ID'],
                                  "text":file_metadata['1-lowercase-and-token-and-remove-figure-index']['caption'],  
                                  "human_feedback": file_metadata['human-feedback']})
            c.close()
        f.close()
        print("There are " + str(len(captions_list)) + " validation samples")
        val_image_folder_path = os.path.join(self.benchmark_path,"No-Subfig-Img","val")+"/"
        with open(val_image_folder_path + "metadata.jsonl", 'w') as f:
            for caption in captions_list:
                f.write(json.dumps(caption) + "\n")
        f.close()
        #Prepare the test dataset
        file_idx_path = os.path.join(self.benchmark_path,"List-of-Files-for-Each-Experiments","First-Sentence","No-Subfig","test","file_idx.json")
        captions_list = []
        f = open(file_idx_path)
        file_idx = json.load(f)
        for image_name in tqdm(file_idx):
            file_metadata_path = os.path.join(self.benchmark_path,"Caption-All","test",os.path.splitext(image_name)[0]+".json")
            c = open(file_metadata_path)
            file_metadata = json.load(c)
            captions_list.append({"file_name":file_metadata['figure-ID'],
                                  "text":file_metadata['1-lowercase-and-token-and-remove-figure-index']['caption'],  
                                  "human_feedback": file_metadata['human-feedback']})
            c.close()
        f.close()
        print("There are " + str(len(captions_list)) + " test examples")
        test_image_folder_path = os.path.join(self.benchmark_path,"No-Subfig-Img","test")+"/"
        with open(test_image_folder_path + "metadata.jsonl", 'w') as f:
            for caption in captions_list:
                f.write(json.dumps(caption) + "\n")
        f.close()
        print("Successfully generated jsonl for train, test and validation")
     
    def __init__(self, benchmark_path) -> None:
        
        """
        Constructor for the FigCapsHF object
        :params benchmark_path: path of the dataset
        :type benchmark_path: String
        """
        
        self.benchmark_path = benchmark_path
        self.generate_jsonl()
        
    def get_image_caption_pair(self, data_split, image_name):
        
        """
        Visualize single figure-caption pair from the large dataset

        :param data_split: the data split (train/val/test) where the image is located
        :type data_split: String
        
        :param image_name: the name of the image to generate from (withou the .png suffix) 
        :type image_name: String

        """
        
        image = Image.open(os.path.join(self.benchmark_path, "No-Subfig-Img", data_split, image_name + ".png"))
        image.show()
        f = open(os.path.join(self.benchmark_path, "Caption-All", data_split, image_name + ".json"))
        data = json.load(f)
        print("The associated caption for this image is: " + data['1-lowercase-and-token-and-remove-figure-index']['caption'])
        f.close()

    def get_image_caption_pair_hf(self, image_name):
        
        """
        Visualize single figure-caption from the human annotated dataset (including HF factors)

        :param image_name: the name of the human annotated image to generate from (without the .png suffix)
        :type image_name: String

        """
        
        csv_path = os.path.join(self.benchmark_path, "human-feedback.csv")
        df = pd.read_csv(csv_path)
        data_split = None
        if os.path.exists(os.path.join(self.benchmark_path,"Caption-All","train",image_name +".json")):
            data_split = "train"
        elif os.path.exists(os.path.join(self.benchmark_path,"Caption-All","test",image_name +".json")):
            data_split = "test"
        elif os.path.exists(os.path.join(self.benchmark_path,"Caption-All","val",image_name +".json")):
            data_split = "val"
        else:
            data_split = "na"
        data_row = df[df['image-file-name'].str.contains(image_name+".png")].values
        if data_split!="na" and len(data_row)!=0:
            self.get_image_caption_pair(data_split,image_name)
            print("\n" + f"The associated human feedback for the figure-caption pair is: helpfulness: {data_row[0][7]}, ocr: {data_row[0][8]}, visual: {data_row[0][9]}, takeaway: {data_row[0][10]}")
    
    def generate_embeddings(self, image_paths, captions_list, embedding_model, MCSE_Path =  None):
        
        """
        Generate embeddings from a specified model for a list of given figure-caption pairs

        :param image_paths: paths to the figure image
        :type image_paths: List
        :param captions_list: list of captions for the corresponding figure images
        :type captions_list: List
        :param embedding_model: Name of the Model to generate embeddings for the figure caption pair. Choose from ['BLIP', 'BERT', 'SciBERT', 'MCSE']
        :type embedding_model: String
        :param MCSE_path: If MCSE is selected, supply path to the folder containing model weights
        :type MCSE_path: String

        :return: embeddings for each figure-caption pair 
        :rtype: (N*D) numpy array

        Note: for MCSE (flickr-mcse-roberta-base) , download the 'flickr-mcse-robert-base' model from 'https://github.com/uds-lsv/MCSE' 
        """
        
        if len(captions_list)!= len(image_paths):
            print("Number of Captions and Images does not match")
            return
        embeddings = []
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_dict = {'SciBERT': "allenai/scibert_scivocab_uncased", "BERT": 'bert-base-uncased', "BLIP": "Salesforce/blip-itm-base-coco"}
        if embedding_model == "SciBERT" or embedding_model == "BERT" or embedding_model == "MCSE":
            if embedding_model == "MCSE":
                embedding_model = MCSE_Path
            else:
                embedding_model = model_dict[embedding_model]
            tokenizer = AutoTokenizer.from_pretrained(embedding_model)
            model = AutoModel.from_pretrained(embedding_model).to(device)
            for caption,image_path in tqdm(zip(captions_list, image_paths), total = len(captions_list)):
                inputs = tokenizer(caption, add_special_tokens = True, max_length = 512, truncation = True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs['last_hidden_state'][0,0,:].cpu().detach().numpy()
                embeddings.append(embedding)
  
        elif embedding_model == "BLIP":
            processor = AutoProcessor.from_pretrained(model_dict["BLIP"])
            model = BlipForImageTextRetrieval.from_pretrained(model_dict["BLIP"]).to(device)
            for caption,image_path in tqdm(zip(captions_list, image_paths), total = len(captions_list)):
                image = Image.open(image_path)
                inputs = processor(images=image, text=caption, max_length = 512, truncation = True, return_tensors="pt").to(device)
                outputs = model(**inputs)
                embedding = outputs['last_hidden_state'][0,0,:].cpu().detach().numpy()
                embeddings.append(embedding)
        else:
            return  
        return np.array(embeddings)
            
    def train_scoring_model(self, hf_embeddings, scores):
        
        """
        Train a MLP Regressor Model to, given a figure-caption embedding, predict the desired human feedback score 

        :param hf_embeddings: Embeddings for the figure-caption pairs
        :type hf_embeddings: Numpy Array

        :param scores: (N,) numpy array containing the target human feedback scores 
        :type scores: Numpy Array

        :return: trained model which can predict human feedback score, given a figure-caption pair embedding
        :rtype: scikit-learn MLP Regressor
        """
            
        scoring_model = MLPRegressor(random_state = 1, max_iter = 500).fit(hf_embeddings,scores)
        return scoring_model
        
    def infer_hf(self, embeddings, scoring_model, quantization_levels = 2):
        
        """
        Using the trained scoring model, predicts the (inferred) human feedback scores for unseen figure-caption pairs
        
        :param embeddings: embeddings of the figure-caption pairs for which human feedback needs to be inferred
        :type embeddings: (N*D) numpy array
        :param scoring_model: model which has been trained to predict human feedback scores given a figure-caption pair embedding
        :type scoring_model: scikit-learn MLP Regressor
        :param quantization_levels: calculates different percentiles of the inferred scores and quantizes the scores into the selected number of levels/bins. Higher quantized score corresponds to a higher score.
        :type quantization_levels: natural number, default is 2
    
        :return: 
        
            inferred_scores: (N,) numpy array with the predicted scores of figure-caption pairs using the supplied model
        
            quantized_scores: (N,) numpy array with the quantized scores (calculated using the inferred scores)

        :rtype: Numpy Array, Numpy Array
        """
        
        N,D  = embeddings.shape
        if quantization_levels > N:
            print("Fewer samples than quantization levels")
            return
        inferred_scores = scoring_model.predict(embeddings)
        quantized_scores = np.digitize(inferred_scores,np.percentile(inferred_scores, np.linspace(0,100,quantization_levels+1,endpoint = True)[1:-1]))
        return (inferred_scores, quantized_scores)
    
    def generate_embeddings_hf_anno(self, hf_score_type, embedding_model, MCSE_path=None):
        
        """
        Generate embeddings from a specified model for the ~400 figure-caption pairs in the human-annotated dataset

        :param hf_score_type: the human-feedback score to be used from the human-annotated dataset. Select between ['helpfulness','ocr','visual','takeaway']
        :type hf_score_type: String
        :param embedding_model: Name of the Model to generate embeddings for the figure caption pair. Choose from ['BLIP', 'BERT', 'SciBERT', 'MCSE']
        :type embedding_model: String    
        :param MCSE_path: If MCSE is selected, supply path to the folder containing model weights 
        :type MCSE_path: String

        :return:
            hf_ds_embeddings: (N*D) numpy array containing embeddings for figure-caption pairs in the human annotated dataset. Note: pairs with an empty specified human-feedback score are removed.

            scores: (N,) numpy array containing the specified human-feedback scores (corresponding to the embeddings)
        :rtype: Numpy Array, Numpy Array

        Note: for MCSE (flickr-mcse-roberta-base) , download the 'flickr-mcse-robert-base' model from 'https://github.com/uds-lsv/MCSE'

        """
        
        #Generate embeddings for the annotated human feedback dataset
        csv_path = os.path.join(self.benchmark_path, "human-feedback.csv")
        df = pd.read_csv(csv_path)
        df_concise = df[['image-file-name', hf_score_type]]
        df_concise = df_concise[df[hf_score_type].notna()]
        scores = np.array(df_concise[hf_score_type].tolist())
        selected_images = df_concise['image-file-name'].tolist()
        hf_locations = []
        for example in selected_images:
            if os.path.exists(os.path.join(self.benchmark_path,"Caption-All","train",os.path.splitext(example)[0]+".json")):
                hf_locations.append("train")
            elif os.path.exists(os.path.join(self.benchmark_path,"Caption-All","test",os.path.splitext(example)[0]+".json")):
                hf_locations.append("test")
            elif os.path.exists(os.path.join(self.benchmark_path,"Caption-All","val",os.path.splitext(example)[0]+".json")):
                hf_locations.append("val")
            else:
                hf_locations.append("na")
                print("error in csv, corresponding caption file not found in dataset")
        hf_captions = []
        hf_image_paths = []
        for pair in zip(selected_images,hf_locations):
            image_name, location = pair
            file_metadata_path = os.path.join(self.benchmark_path,"Caption-All",location,os.path.splitext(image_name)[0]+".json")
            c = open(file_metadata_path)
            file_metadata = json.load(c)
            hf_captions.append(file_metadata['1-lowercase-and-token-and-remove-figure-index']['caption'])
            hf_image_paths.append(os.path.join(self.benchmark_path,"No-Subfig-Img",location,image_name))
            c.close()
        hf_ds_embeddings = self.generate_embeddings(hf_image_paths, hf_captions, embedding_model, MCSE_Path =  MCSE_path)
        return (hf_ds_embeddings,scores)
    
    def generate_embeddings_ds_split(self, embedding_model, MCSE_path=None, split_name = "train", max_num_samples = None):
        
        """
        Generate embeddings from a specified model for the figure-caption pairs in the larger dataset.

        :param embedding_model: Name of the Model to generate embeddings for the figure caption pair. Choose from ['BLIP', 'BERT', 'SciBERT', 'MCSE']
        :type embedding_model: String
        :param split_name: data split to generate embeddings from. Choose from ['train','test','val']
        :type split_name: String, default = 'train'
        :param max_num_samples: maximum number of samples from the specified split to generate embeddings for(picks the first N) 
        :type max_num_samples: natural number, default = all pairs in the specified split
        :param MCSE_path: If MCSE is selected, supply path to the folder containing model weights 
        :type MCSE_path: String

        :return:
            ds_split_embeddings: (N*D) numpy array containing embeddings for figure-caption pairs in the specified data split

            image_names: list containing the paths of the figure-images

            captions: list containing the captions corresponding to the figure-images
            
        :rtype: Numpy Array, List, List

        Note: for MCSE (flickr-mcse-roberta-base) , download the 'flickr-mcse-robert-base' model from 'https://github.com/uds-lsv/MCSE'

        """

        json_path = os.path.join(self.benchmark_path, "No-Subfig-Img",split_name,"metadata.jsonl")
        json_df = pd.read_json(json_path, lines= True)
        if max_num_samples!=None:
            json_df = json_df.head(max_num_samples)
        image_names = json_df['file_name']
        json_df['file_name'] = os.path.join(self.benchmark_path, "No-Subfig-Img", split_name) + "/" + json_df['file_name']
        captions = json_df['text'].tolist()
        image_paths = json_df['file_name'].tolist()
        ds_split_embeddings = self.generate_embeddings(captions, image_paths, embedding_model, MCSE_Path = MCSE_path)
        return (ds_split_embeddings, image_names, captions)
    
    def infer_hf_training_set(self, hf_score_type, embedding_model, MCSE_path = None, max_num_samples = None, quantization_levels = 2, mapped_hf_labels = ["bad", "good"]):
        
        """
        Predict the (inferred) human feedback scores for the training-split of the dataset, using a scoring model trained on the human-annotated dataset
        
        :param hf_score_type: the human-feedback score to be used from the human-annotated dataset. Select between ['helpfulness','ocr','visual','takeaway']
        :type hf_score_type: String
        :param embedding_model: Name of the Model to generate embeddings for the figure caption pair. Choose from ['BLIP', 'BERT', 'SciBERT', 'MCSE']
        :type embedding_model: String
        :param MCSE_path: If MCSE is selected, supply path to the folder containing model weights 
        :type MCSE_path: String
        :param max_num_samples: maximum number of samples from the specified split to generate embeddings for(picks the first N) 
        :type max_num_samples: natural number, default = all pairs in the specified split
        :param quantization_levels: calculates different percentiles of the inferred scores and quantizes the scores into the selected number of levels/bins. Higher score corresponds to a higher score.
        :type quantization_levels: natural number, default is 2
        :param mapped_hf_labels: a list containing string labels corresponding to each quantization level (quantization_levels number of labels needed) 
        :type mapped_hf_labels: List of strings, default is ["bad","good"]

        Note: for MCSE (flickr-mcse-roberta-base) , download the 'flickr-mcse-robert-base' model from 'https://github.com/uds-lsv/MCSE' 

        :return: inferred_hf_df containing rows with
                
                - file_name: image_name(with .png)
                
                - text: corresponding caption
                
                - inferred_hf: predicted human-feedback scores using a trained scoring model
                
                - quantized_hf: quantized inferred human-feedback scores
                
                - mapped_hf: mapped quantized human-feedback scores

        :rtype: Pandas DataFrame
        """

        hf_ds_embeddings, scores = self.generate_embeddings_hf_anno(hf_score_type, embedding_model, MCSE_path = MCSE_path)
        scoring_model = self.train_scoring_model(hf_ds_embeddings,scores)
        train_ds_embeddings, image_names, training_captions  = self.generate_embeddings_ds_split(embedding_model, MCSE_path, "train", max_num_samples)
        inferred_hf, quantized_hf = self.infer_hf(train_ds_embeddings, scoring_model, quantization_levels)
        #generate new df with name of image, text, and inferred human feedback, quantized feedback, mapped feedback
        inferred_hf_df = pd.DataFrame(list(zip(image_names,training_captions,inferred_hf,quantized_hf)), columns = ['file_name','text','inferred_hf','quantized_hf'])
        if mapped_hf_labels!=None and len(mapped_hf_labels)==quantization_levels:
            label_dict = {key:value for key,value in zip(np.arange(quantization_levels),mapped_hf_labels)}
            inferred_hf_df['mapped_hf'] = inferred_hf_df['quantized_hf']
            inferred_hf_df['mapped_hf'] = inferred_hf_df['mapped_hf'].map(label_dict)
        return inferred_hf_df


   
