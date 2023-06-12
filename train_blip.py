import argparse
import os
import re
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from datasets import load_dataset 
from transformers import AutoProcessor, BlipForConditionalGeneration

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor, hf_score_type):
        self.dataset = dataset
        self.processor = processor
        self.hf_score_type = hf_score_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        encoding = self.processor(images=item["image"], text=item['human_feedback'][self.hf_score_type]['caption-prepend'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        # remove batch dimension
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        return encoding

def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision, project_dir=args.logging_dir)

    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    num_epochs = int(config["num_epochs"])
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])
    
    # Parse out whether we are saving every epoch or after a certain number of batches
    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
        f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed.")
    else:
        checkpointing_steps = None
        
    # Instantiate dataloaders.
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # path to the folder containing the images
    root = os.path.join(args.benchmark_path,"No-Subfig-Img","train/")
    dataset = load_dataset("imagefolder", data_dir=root, split="train")
    train_dataset = ImageCaptioningDataset(dataset, processor, args.hf_score_type)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=2)
    
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print('PARAMS:', sum(p.numel() for p in model.parameters()))

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    model = model.to(accelerator.device)

    # Instantiate optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    
    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # Now we train the model
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        total_loss = 0
        num_samples = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            overall_step += resume_step
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False, disable=not accelerator.is_local_main_process)
        for idx, batch in loop:
            # We could avoid this line since we set the accelerator with `device_placement=True`.
            #batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            input_ids = batch.pop("input_ids").to(accelerator.device)
            pixel_values = batch.pop("pixel_values").to(accelerator.device)
            outputs = model(input_ids=input_ids,pixel_values=pixel_values,labels=input_ids)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            num_samples += input_ids.shape[0]
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            overall_step += 1
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=total_loss.item()/ num_samples)
            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}")
        #accelerator.log({"train_loss": total_loss.item() / len(train_dataloader),"epoch": epoch},step=overall_step,)
        if checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            
def main():
    parser = argparse.ArgumentParser(description="BLIP with RLHF training script.")
    parser.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default='epoch',
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="blip_checkpoints",
        help="Optional save directory where all checkpoint folders will be stored.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Location on where to store experiment tracking logs",
    )
    parser.add_argument(
        "--hf_score_type",
        type=str,
        default="helpfulness",
        help="The human-feedback score type to be used for RLHF, choose from: helpfulness, ocr, visual, or takeaway",
    )
    parser.add_argument(
        "--benchmark_path",
        type=str,
        default=os.getcwd(),
        help="The path to the benchmark dataset (folder which contains the No-Subfig-Img folder)",
    )
    
    args = parser.parse_args()
    
    config = {"lr": 5e-5, "num_epochs": 10, "seed": 42, "batch_size": 2}
    training_function(config, args)

if __name__ == "__main__":
    main()
