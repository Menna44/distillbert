---
license: other
tags:
- generated_from_trainer
model-index:
- name: finetuned-distilbert-adult-content-detection
  results: []
---

### finetuned-distilbert-news-article-catgorization

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the adult_content dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0065
- F1_score(weighted): 0.99
### Model description
 More information needed
### Intended uses & limitations
More information needed
### Training and evaluation data
The model was trained on some subset of the adult_content dataset and it was validated on the remaining subset of the data
### Training procedure
More information needed
### Training hyperparameters
The following hyperparameters were used during training:
- learning_rate: 5e-5
- train_batch_size: 5
- eval_batch_size: 5
- seed: 17
- optimizer: AdamW(lr=1e-5 and epsilon=1e-08)
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 0
- num_epochs: 5
### Training results
| Training Loss | Epoch |  Validation Loss | f1 score   |
|:-------------:|:-----:|:---------------: |:------:|
| 0.0065        | 1.0   | 0.0669           | 0.9880 |
| 0.0429        | 2.0   | 0.0530           | 0.9907 |
| 0.0177        | 3.0   | 0.0305           | 0.9933 |
| 0.0123        | 4.0   | 0.0362           | 0.9933 |
| 0.0065        | 5.0   | 0.0176           | 0.9960 |