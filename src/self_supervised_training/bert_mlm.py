import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from typing import List
from transformers import BertTokenizer, DataCollatorForLanguageModeling, BertForMaskedLM, TrainingArguments, Trainer


class Tweets(Dataset):
    def __init__(self, tweets: List[str], tokenizer: BertTokenizer, seq_len: int):
        """
        :param tweets: list of tweets (text)
        :type tweets: List[str]

        :param tokenizer: used to transform text into tokens
        :type tokenizer: BertTokenizer

        :param seq_len: length of bert sequence
        :type seq_len: int
        """
        super(Tweets, self).__init__()
        tweets_tokenized = []
        for tweet in tweets:
            tweets_tokenized.append(tokenizer.encode(tweet,
                                                     add_special_tokens=True,
                                                     truncation=True,
                                                     max_length=seq_len))
        self.tweets_tokenized = tweets_tokenized
        self.len = len(tweets_tokenized)

    def __len__(self):
        return self.len

    def __getitem__(self, item: int):
        return self.tweets_tokenized[item]


class TweetHandler:
    def __init__(self, seq_len: int, train_size: float, mlm_p: 0.15, tokenizer: BertTokenizer, n_samples: int = 160):
        """
        :param seq_len: length of transformer sequence
        :type seq_len: int

        :param train_size: share of tweets used as training data
        :type train_size: float

        :param mlm_p: mask out probability for masked language modeling
        :type mlm_p: float

        :param tokenizer: used to tokenized textual data
        :type tokenizer: BertTokenizer

        :param n_samples: number of samples used
        :type n_samples: int
        """
        data = pd.read_csv("../data/tweets.csv")
        print("using", n_samples, "of", len(data), "data points")
        self.tweets = data["content"].values[:n_samples]
        self.seq_len = seq_len

        self.collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_p)

        # todo train val test split
        self.tweets_train = Tweets(tweets=self.tweets[:int(len(self.tweets) * train_size)],
                                   tokenizer=tokenizer,
                                   seq_len=seq_len)
        self.tweets_val = Tweets(tweets=self.tweets[int(len(self.tweets) * train_size):],
                                 tokenizer=tokenizer,
                                 seq_len=seq_len)


bert_checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_checkpoint)
bert = BertForMaskedLM.from_pretrained(bert_checkpoint)
seq_len = 64
train_size = 0.8
mlm_p = 0.15
n_samples = 1600
batch_size = 64
n_epochs = 5

tweet_handler = TweetHandler(seq_len=seq_len,
                             train_size=train_size,
                             mlm_p=mlm_p,
                             tokenizer=tokenizer,
                             n_samples=n_samples)
train_args = TrainingArguments(num_train_epochs=n_epochs,
                               per_device_train_batch_size=batch_size,
                               per_device_eval_batch_size=batch_size,
                               save_strategy="epoch",
                               evaluation_strategy="epoch",
                               logging_strategy="epoch",
                               output_dir="../monitoring/checkpoints",
                               save_total_limit=1,
                               overwrite_output_dir=True)
trainer = Trainer(model=bert,
                  args=train_args,
                  train_dataset=tweet_handler.tweets_train,
                  eval_dataset=tweet_handler.tweets_val,
                  data_collator=tweet_handler.collator)

train_result = trainer.train()

losses_train = []
losses_val = []
for i, elem in enumerate(trainer.state.log_history):
    print(elem)
    if elem.keys().__contains__("eval_loss"):
        losses_val.append(elem["eval_loss"])
    elif elem.keys().__contains__("loss"):
        losses_train.append(elem["loss"])

x = torch.arange(len(losses_train))
plt.plot(x, losses_train, label="training data")
plt.plot(x, losses_val, label="validation data")
plt.title("MLM fine tuning")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.tight_layout()
plt.savefig("../monitoring/results.png")
