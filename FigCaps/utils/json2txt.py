import os, json
import pandas as pd

path_to_json = 'D:/Downloads/benchmark/benchmark/Caption-All/train'
# path_to_label = 'D:/Downloads/scicap_data/scicap_data/SciCap-Caption-All/train'
json_files = [path_to_json+"/"+pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

# label_files = [path_to_label+"/"+pos_json[:-5]+".txt" for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

lst = []
for i in range(len(json_files)):
    each = json_files[i]
    adict = json.load(open(each))
    dict = {k:adict[k] for k in ('figure-ID','human-feedback') if k in adict}
    dict["file_name"] = dict['figure-ID']
    dict['RLHF_case_caption'] = dict['human-feedback']['helpfulness']['caption-prepend']
    del dict['figure-ID']
    del dict['human-feedback']
    lst.append(dict)



with open('D:/Downloads/benchmark/benchmark/No-Subfig-Img/train/metadata.jsonl', 'w+') as f:
    for item in lst:
        f.write(json.dumps(item) + "\n")