import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)
    
def main():

    import torch
    import numpy as np
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor

    image_encoder_model = "google/vit-base-patch16-224-in21k"
    text_decode_model = "gpt2"

    # image feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
    # text tokenizer
    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        image_encoder_model, text_decode_model)

    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.is_decoder = True

    model.to(device)


    from datasets import load_dataset 
    root = 'benchmark/No-Subfig-Img'
    dataset = load_dataset("imagefolder", data_dir=root) #needs the metadata.jsonl to load a imagefolder with metadata

    dataset = dataset.map(lambda e: feature_extractor(e['image']), batched=True)
    dataset = dataset.map(lambda e: tokenizer(e['RLHF_lower_case_caption'], truncation=True, padding='max_length'), batched=True) 
    dataset = dataset.remove_columns("RLHF_lower_case_caption")
    dataset = dataset.remove_columns("attention_mask")
    dataset = dataset.remove_columns("image")
    dataset = dataset.rename_column("input_ids", "labels")


    #Training Arguments
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
    
    #Metrics 
    import evaluate
    metric = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load('meteor')

    import numpy as np

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    ignore_pad_token_for_loss = True
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                        decoded_labels)
        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels,
                                use_stemmer=True)
        result1 = bleu.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result2 = meteor.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result = {**result, **result1}
        result = {**result,**result2}
        # result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
        ]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        gradient_accumulation_steps=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        output_dir="output",
        learning_rate = 6e-5,
        weight_decay=0.01,
        num_train_epochs=5,
        save_strategy= "epoch",
        save_total_limit= 5,
        fp16 = True,
    )

    from transformers import default_data_collator

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=default_data_collator,
        
    )

    trainer.train()
    trainer.save_model("model")
    
if __name__ == '__main__':
    main()
