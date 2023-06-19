# FigCaps-HF: A Figure-to-Caption Generative Framework and Benchmark with Human Feedback

<!-- [[`Paper`]()]  [[`BibTex`]()]  -->
[[Website](https://figcapshf.github.io/)] [[Benchmark Dataset](https://doi.org/10.6084/m9.figshare.23504517)]

To enable the generation of high-quality figure captions, we introduce **FigCaps-HF**, a new framework for figure-caption generation that can incorporate domain expert feedback in generating captions optimized for reader preferences. 
Our framework comprises of 1) an automatic method for evaluating quality of figure-caption pairs, 2) a novel reinforcement learning with human feedback (RLHF) method to optimize a generative figure-to-caption model for reader preferences.

We release a large-scale benchmark dataset with human feedback on figure-caption pairs to enable further evaluation and development of RLHF techniques for this problem.

## Benchmark Dataset 
The benchmark dataset can be downloaded here: [[Download Link](https://figshare.com/ndownloader/files/41222934)](8.34 GB)


### Folder Structure
```
├── No-Subfig-Img                       #contains figure-image files for each split of the dataset
│	├── Train
│	├── Val
│	└── Test
├── Caption-All                         #contains corresponding figure-captions and (precomputed) inferred human-feedback metadata
│	├── Train
│	├── Val
│	└── Test
├── human-feedback.csv                  #contains human evaluations of a subset of figure image-caption pairs
├── arxiv-metadata-oai-snapshot.json    #arXiv paper metadata (from arXiv dataset) 
└── List-of-Files-for-Each-Experiments  #list of figure names used in each experiment 
    ├── Single-Sentence-Caption
    │   ├── No-Subfig
    │   │   ├── Train
    │	│   ├── Val
    │	│   └── Test
    │	└── Yes-Subfig
    │       ├── Train
    │       ├── Val
    │       └── Test
    ├── First-Sentence                  #Same as in Single-Sentence-Caption
    └── Caption-No-More-Than-100-Tokens #Same as in Single-Sentence-Caption
```
### Human Feedback Benchmark Data 
The included human-feedback.csv contains human evaluations of 439 figure image-caption pairs from the dataset. These evaluations consist of ratings, for each image-caption pair, of the “helpfulness”, “OCR (quality)”, “takeaway” and “visual (descriptiveness)”, each scored on a 1-5 point scale (5 being the highest). Additionally, the annotations include a boolean indicating whether each pair “has-image-error”, “has-caption-error”, “has-classification-error” or “has-subfigure-error”. For convenience, the image-file name and url of the originating arXiv-paper are also included.

### Number of Figures in Each Subset

|                         |  Train  | Validate |  Test  |
|------------------------:|:-------:|:--------:|:------:|
| Benchmark                 | 106,834 |  13,354  | 13,355 |


## JSON Data Format (for each figure-caption in Caption-All)

### Example JSON
```
{
  "contains-subfigure": true, 
  "Img-text": ["(b)", "s]", "[m", "fs", "et", "e", "of", "T", "im", "Attack", "duration", "[s]", "350", "300", "250", "200", "150", "100", "50", "0", "50", "100", "150", "200", "250", "300", "0", "(a)", "]", "[", "m", "fs", "et", "e", "of", "ta", "nc", "D", "is", "Attack", "duration", "[s]", "10000", "9000", "8000", "7000", "6000", "5000", "4000", "3000", "2000", "1000", "0", "50", "100", "150", "200", "250", "300", "0"], 
  "paper-ID": "1001.0025v1", 
  "figure-ID": "1001.0025v1-Figure2-1.png", 
  "figure-type": "Graph Plot", 
  "human-feedback":{
    "helpfulness": {
      "score": XXXX,
      "label": "[GOOD]/[BAD]",
      "caption-prepend": "[GOOD]/[BAD] actual caption...",
    },
    "ocr": {
      "score": XXXX,
      "label": "[GOOD]/[BAD]",
      "caption-prepend": "[GOOD]/[BAD] actual caption...",
    },
    "visual": {
      "score": XXXX,
      "label": "[GOOD]/[BAD]",
      "caption-prepend": "[GOOD]/[BAD] actual caption...",
    },
    "takeaway": {
      "score": XXXX,
      "label": "[GOOD]/[BAD]",
      "caption-prepend": "[GOOD]/[BAD] actual caption...",
    },
  }
  "0-originally-extracted": "Figure 2: Impact of the replay attack, as a function of the spoofing attack duration. (a) Location offset or error: Distance between the attack-induced and the actual victim receiver position. (b) Time offset or error: Time difference between the attack-induced clock value and the actual time.", 
  "1-lowercase-and-token-and-remove-figure-index": {
    "caption": "impact of the replay attack , as a function of the spoofing attack duration . ( a ) location offset or error : distance between the attack-induced and the actual victim receiver position . ( b ) time offset or error : time difference between the attack-induced clock value and the actual time .", 
    "sentence": ["impact of the replay attack , as a function of the spoofing attack duration .", "( a ) location offset or error : distance between the attack-induced and the actual victim receiver position .", "( b ) time offset or error : time difference between the attack-induced clock value and the actual time ."], 
    "token": ["impact", "of", "the", "replay", "attack", ",", "as", "a", "function", "of", "the", "spoofing", "attack", "duration", ".", "(", "a", ")", "location", "offset", "or", "error", ":", "distance", "between", "the", "attack-induced", "and", "the", "actual", "victim", "receiver", "position", ".", "(", "b", ")", "time", "offset", "or", "error", ":", "time", "difference", "between", "the", "attack-induced", "clock", "value", "and", "the", "actual", "time", "."]
  }, 
  "2-normalized": {
    "2-1-basic-num": {
      "caption": "impact of the replay attack , as a function of the spoofing attack duration . ( a ) location offset or error : distance between the attack-induced and the actual victim receiver position . ( b ) time offset or error : time difference between the attack-induced clock value and the actual time .", 
      "sentence": ["impact of the replay attack , as a function of the spoofing attack duration .", "( a ) location offset or error : distance between the attack-induced and the actual victim receiver position .", "( b ) time offset or error : time difference between the attack-induced clock value and the actual time ."], 
      "token": ["impact", "of", "the", "replay", "attack", ",", "as", "a", "function", "of", "the", "spoofing", "attack", "duration", ".", "(", "a", ")", "location", "offset", "or", "error", ":", "distance", "between", "the", "attack-induced", "and", "the", "actual", "victim", "receiver", "position", ".", "(", "b", ")", "time", "offset", "or", "error", ":", "time", "difference", "between", "the", "attack-induced", "clock", "value", "and", "the", "actual", "time", "."]
      }, 
    "2-2-advanced-equation-bracket": {
      "caption": "impact of the replay attack , as a function of the spoofing attack duration . BRACKET-TK location offset or error : distance between the attack-induced and the actual victim receiver position . BRACKET-TK time offset or error : time difference between the attack-induced clock value and the actual time .", 
      "sentence": ["impact of the replay attack , as a function of the spoofing attack duration .", "BRACKET-TK location offset or error : distance between the attack-induced and the actual victim receiver position .", "BRACKET-TK time offset or error : time difference between the attack-induced clock value and the actual time ."], 
      "tokens": ["impact", "of", "the", "replay", "attack", ",", "as", "a", "function", "of", "the", "spoofing", "attack", "duration", ".", "BRACKET-TK", "location", "offset", "or", "error", ":", "distance", "between", "the", "attack-induced", "and", "the", "actual", "victim", "receiver", "position", ".", "BRACKET-TK", "time", "offset", "or", "error", ":", "time", "difference", "between", "the", "attack-induced", "clock", "value", "and", "the", "actual", "time", "."]
      }
    }
  }
```

### JSON Schema

- **contains-subfigure:** boolean (if figure-image contains subfigures)
- **paper-ID:** the unique paper ID in the arXiv dataset
- **figure-ID:** the extracted figure ID of paper (the index is not the same as the label in the caption)
- **figure-type:** the figure type
- **0-originally-extracted:** original figure-caption
- **1-lowercase-and-token-and-remove-figure-index:** Removed figure index and the captions in lowercase
- **2-normalized:** 
  - **2-1-basic-num:** caption after replacing the number
  - **2-2-advanced-euqation-bracket:** caption after replacing the equations and contents in the bracket
- **Img-text:** texts extracted from the figure, such as the texts for labels, legends ... etc.

Within the caption content, we have three attributes:

- **caption:** caption after each normalization
- **sentence:** a list of segmented sentences
- **token:** a list of tokenized words

Within the human-feedback field, we have the inferred human-feedback for the different metrics (helpfulness, ocr, takeaway, and visual). The tokens are decided based on the median score of the dataset on that metric.

**human-feedback:** 
  - **helpfulness:**   Expert's rating on how helpful a caption is to understand a scientific figure
    - **Score:**             predicted score                               
    - **Token:**             [Good]/[Bad]
    - **caption-prepend:**    1-lowercase-and-token-and-remove-figure-index caption with the token pre-pended
  - **takeaway:**      Expert's rating on the takeaway from the scientific image 
    - **Score:**             predicted score              
    - **Token:**             [Good]/[Bad]
    - **caption-prepend:**    1-lowercase-and-token-and-remove-figure-index caption with the token pre-pended
  - **ocr:**           Expert's rating on the OCRs expressiveness
    - **Score:**             predicted score              
    - **Token:**             [Good]/[Bad]
    - **caption-prepend:**    1-lowercase-and-token-and-remove-figure-index caption with the token pre-pended
  - **visual:**        Expert's rating on the visualness of the scientific figure 
    - **Score:**             predicted score              
    - **Token:**             [Good]/[Bad]
    - **caption-prepend:**    1-lowercase-and-token-and-remove-figure-index caption with the token pre-pended

## Installation 

```shell
#We first need to clone this repository, install the requirements, and download the benchmark dataset
pip install --upgrade pip
git clone https://github.com/FigCapsHF/FigCapsHF
pip install -r requirements.txt
wget  https://figshare.com/ndownloader/files/41222934 -O benchmark.zip
unzip benchmark.zip
```
## Example Usage

### RLHF Fine-tuning
```python
#Code edits to implement a baseline are also included in train_blip.py
#Preferred training on GPUs. If training on CPU, add "--cpu" flag.
python train_blip.py --mixed_precision fp16 --hf_score_type helpfulness --benchmark_path XX/benchmark

```
### Inference 
Our RLHF Fine-tuned BLIP Model can be downloaded here: [[Download Link](https://drive.google.com/file/d/1BtyBkk9bZeruzjttMAzWlTDnJCzLlmpc/view?usp=share_link)](2.5 GB)

```python
#Generate caption for a single image
python inference.py --figure_path /path/test_image.png --model_path /path/model.pth
```
```python
#Generate evaluation metrics on the test dataset
python test_blip.py --benchmark_path XX/benchmark --model_path /path/model.pth
```


### Visualization
#For the following sections, we initialize a FigCapsHF object
```python
from FigCapsHF import FigCapsHF
FigCapsHF = FigCapsHF("path/to/benchmark/data")
```
```python
#Visualize sample from dataset
FigCapsHF.get_image_caption_pair(data_split = "train", image_name = "1001.0025v1-Figure5-1")
```
```python
#Visualize sample from the human annotated dataset and associated metadata
FigCapsHF.get_image_caption_pair_hf(image_name = "1907.11521v1-Figure6-1")
```

### Human Feedback Generation

```python
#Generate human-feedback metadata for the dataset
inferred_hf_df = FigCapsHF.infer_hf_training_set(hf_score_type = "helpfulness", embedding_model = "BERT", max_num_samples = 100, quantization_levels = 3, mapped_hf_labels = ["Bad", "Neutral", "Good"])
```

```python
#Generate a human-feedback score for a single figure-caption pair
hf_ds_embeddings, scores = FigCapsHF.generate_embeddings_hf_anno(hf_score_type = "helpfulness", embedding_model = "BERT")
scoring_model = FigCapsHF.train_scoring_model(hf_ds_embeddings, scores)

image_path = "/path/1907.11521v1-Figure6-1.png"
caption = "the graph indicates the loss of the model over successive generations"

embedding = FigCapsHF.generate_embeddings([image_path], [caption], embedding_model = "BERT")
inferred_hf_score = scoring_model.predict(embedding)

```


<!-- 

| Model            | Parameters | ROUGE-L | BLEU   | Meteor |
|------------------|------------|---------|--------|--------|
| BLIP             | 0.25B      | 0.130   | 0.014  | 0.132  |
| Ours-BLIP-RLHF   | 0.25B      | 0.152   | 0.019  | 0.145  |
 -->
<!-- ## Retrieving the dataset
Our benchmark dataset can be downloaded from [[`here`](https://figshare.com/s/c034fd77bea9475319cb)]. -->

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
  
This dataset uses data in the [arXiv dataset](https://www.kaggle.com/Cornell-University/arxiv).
The [arXiv dataset](https://www.kaggle.com/Cornell-University/arxiv) uses the [CC0 1.0 Universal (CC0 1.0) Public Domain Dedication license](https://creativecommons.org/publicdomain/zero/1.0/) for the metadata, which grants permission to remix, remake, annotate, and publish the metadata.

<!-- Another way for Training. -->
<!-- Here we are using BLIP as a sample model for training using Pytorch's native DataLoader library combined with Huggingface's dataset class. It also has a training loop. User can provide arguments for their desired functionality as shown in the script below. To change the the number of epochs and learning rate, modify config variable in train_blip.py.
```shell
python train_blip.py --f16 --output_dir output

```

## Inference
（after downloading and setting up the dataset) To predict a scientific caption from an scientific image, and if the user desire to use other models for inference, please make the modification in inference.py. Below, we will provide a pre-trained BLIP model trained on this benchmark dataset and make simple inference using the model provided.
```shell 
./pred.sh
cd BLIP
```
Now use this [link](https://drive.google.com/file/d/1FZh95Xeyt3RlaYs_TeeiiSPwYvAuGogQ/view?usp=share_link) to download our model, and place it under directory BLIP, and use the following script to do an inference on the sample.png
```shell
python inference.py sample.png
```
If running on a CPU, the expected result is *the results of comparing oa and noa in terms of mean of error.* (on seed 42).

![Sample Scientific figure](/Figures/sample.png)  -->
