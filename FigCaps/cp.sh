#!/bin/bash

#fixes a encoding problem 
mv "benchmark/No-Subfig-Img/val/1707.07466v2-FigureCกค1-1.png" "benchmark/No-Subfig-Img/val/1707.07466v2-Figure1-1.png"

mv "benchmark/Caption-All/val/1707.07466v2-FigureCกค1-1.json" "benchmark/Caption-All/val/1707.07466v2-Figure1-1.json"

# Rename and move the first file
cp -i "Metadata/metadata_Train.jsonl" "benchmark/No-Subfig-Img/train/metadata.jsonl"

# Rename and move the second file
cp -i "Metadata/metadata_Test.jsonl" "benchmark/No-Subfig-Img/test/metadata.jsonl"

# Rename and move the third file
cp -i "Metadata/metadata_Val.jsonl" "benchmark/No-Subfig-Img/val/metadata.jsonl"



