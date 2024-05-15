from datasets import load_dataset, load_metric
from transformers import AlbertTokenizerFast, TrainingArguments, Trainer, AlbertConfig# , BertForPreTraining
from svt_bert_pretrain import BertForPreTraining
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy, BatchEncoding
from typing import Optional, Union, List, Dict, Tuple
import torch
from torch import nn
from functools import partial
from svt_pretrain import SVTForPretraining
from dataclasses import dataclass
def compute_metrics(eval_predictions):
    labels, next_sentence_label=eval_predictions.label_ids
    prediction_scores, seq_relationship_score=eval_predictions.predictions
    prediction_scores=prediction_scores.argmax(-1).reshape(-1)
    labels=labels.reshape(-1)
    mlm_acc = prediction_scores.eq(labels)
    mask = torch.logical_not(labels.eq(-100))
    mlm_acc = mlm_acc.logical_and(mask)
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()  # 需要预测的个数
    mlm_acc = float(mlm_correct) / mlm_total  # 计算被mask掉的词语的个数然后计算准确率

    nsp_correct = (seq_relationship_score.argmax(1) == next_sentence_label).float().sum()
    nsp_total = len(next_sentence_label)
    nsp_acc = float(nsp_correct) / nsp_total
    return {
        "mlm_acc": mlm_acc,
        "nsp_acc": nsp_acc
    }

@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: int = 256
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        nsp_labels = [example.pop("next_sentence_label") for example in examples]
        batch_size = len(examples)
        flattened_features = [{k: v[:230] for k, v in feature.items()} for feature in
                              examples]
        max_length = max([len(feature["input_ids"]) for feature in examples])
        if max_length%2==0:
            max_length+=0
        else:
            max_length+=1
        max_length=max_length if max_length<230 else 230
        batch = self.tokenizer.pad(flattened_features, padding = self.padding, max_length = max_length,return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        batch["next_sentence_label"] = torch.tensor(nsp_labels, dtype=torch.long)

        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        '''
        inputs必须为tensor类型，
        '''

        # 这里的labels指的是哪些字要被mask
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        # 生成一个全为0.15 的矩阵，维度和inputs的大小一样。
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # 这里的spcial_tokens指的是cls，sep,pad等，special_tokens_mask的维度和inputs一样，
        # 特殊token的位置是true，其他位置是false
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()
        # masked_fill_函数的作用是会根据special_tokens_mask中true的位置，将probability_matrix中对于的位置置为0
        # 前面special_tokens_mask已经将cls等特殊字符的位置置为true,也就是通过将probability_matrix中特殊字符的位置置0，从而
        # 不会mask掉特殊字符。
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        # Bert是将句子中15%的词随机mask掉，也就是每个词有15%的概率被mask掉（1），有85%的概率不被mask掉（0），
        # 因此可以建模为伯努利分布，参数p=0.15，
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
tokenizer=AlbertTokenizerFast.from_pretrained("/root/autodl-tmp/xlnet-base")# model config directory 
datasets = load_dataset('csv', data_files={"train":"./data/total_pretrain1.csv", "val":"./data/c4_test.csv"}, cache_dir="./")# pretrain corpus
def preprocess_function(examples):
    first_sentences=[str(sent1) for sent1 in examples['sent1']]
    second_sentences=[str(sent2) for sent2 in examples['sent2']]
    tokenizer_examples=tokenizer(first_sentences,second_sentences,truncation=True)
    return {k: v for k, v in tokenizer_examples.items()}
print(len(datasets['train']))
encodings_datasets=datasets.map(preprocess_function, remove_columns=["sent1",'sent2'],batched=True)
training_args=TrainingArguments(
    output_dir="./albert_ftwt_result",# "./prediction_bert_result",
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    warmup_steps=5000,# 5000,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    save_strategy="steps",
    save_steps=200000,
    logging_dir="./albert_ftwt_log",#,'./pretrained_bert_logs',
    logging_steps=5000,
    save_safetensors = False,
    # label_names=["labels", "next_sentence_label"],
    # remove_unused_columns=False,
    # include_inputs_for_metrics=True
)
config=AlbertConfig.from_pretrained("/root/autodl-tmp/xlnet-base") # model config directory 
model = SVTForPretraining(config)
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=encodings_datasets['train'],
    eval_dataset=encodings_datasets['val'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer),
    # compute_metrics=compute_metrics
)
trainer.train(resume_from_checkpoint=True)
trainer.evaluate()
trainer.save_model()# ("./result_svt_bert")
