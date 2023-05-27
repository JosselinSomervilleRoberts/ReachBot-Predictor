import numpy as np
import torch
from transformers import (
    AutoImageProcessor,
    ResNetForImageClassification,
    Trainer,
    TrainingArguments,
    ViTFeatureExtractor,
    ViTForImageClassification,
)

from datasets import load_dataset, load_metric


def transform(example_batch, feature_extractor):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor(
        [x.resize((224, 224)) for x in example_batch["image"]], return_tensors="pt"
    )
    # Don't forget to include the labels!
    inputs["labels"] = example_batch["label"]
    return inputs


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


if __name__ == "__main__":
    model_type = "resnet"

    if model_type == "resnet":
        output_dir = "./resnet-cracks"
    elif model_type == "vit":
        output_dir = "./vit-cracks"

    dataset = load_dataset("imagefolder", data_dir="datasets/cracks_classification")
    labels = dataset["train"].features["label"].names

    if model_type == "resnet":
        feature_extractor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
            ignore_mismatched_sizes=True,
        )
    elif model_type == "vit":
        model_name_or_path = "google/vit-base-patch16-224-in21k"
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)
        model = ViTForImageClassification.from_pretrained(
            model_name_or_path,
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        )
    prepared_ds = dataset.with_transform(lambda x: transform(x, feature_extractor))
    metric = load_metric("accuracy")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="wandb",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=feature_extractor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    metrics = trainer.evaluate(prepared_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
