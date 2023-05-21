import boto3
import torch
import torch.nn as nn
from io import StringIO
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from own_transformers.bert_config import BertConfig
from own_transformers.model import modeling_bert
from typing import List, Optional, Tuple, Union
from torch import cuda
import numpy as np

"""
python -m own_transformers.test_train_bert

BertModel(
  (bert_embedding): BertEmbedding(
    (word_embedding): Embedding(30522, 768)
    (position_embedding): Embedding(512, 768)
    (token_embedding): Embedding(1, 768)
    (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (bert_encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): BertLayer(
        (bert_attention): BertAttention(
          (self_attention): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (self_output): BertSelfOutput(
            (linear): Linear(in_features=768, out_features=768, bias=True)
            (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (act): GELU(approximate=none)
        )
        (bert_output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (bert_pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activate): Tanh()
  )
)
"""


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BertClassification(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.bert = modeling_bert.BertModel(config)
        self.dense = nn.Linear(config.hidden_size, 6)

    def forward(
        self,
        input: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bert_output = self.bert(input, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.dense(bert_output)


client = boto3.client('s3')
def load_csv_from_s3():
    bucket_name = 'datausers'
    object_key = 'xgui/test-bert/train.csv'
    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))


if __name__ == "__main__":
    print('loading data')
    df = load_csv_from_s3()

    df['list'] = df[df.columns[2:]].values.tolist()
    new_df = df[['comment_text', 'list']].copy()

    # Defining some key variables that will be used later on in the training
    MAX_LEN = 200
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    EPOCHS = 20
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Creating the dataset and dataloader for the neural network

    train_size = 0.8
    train_dataset=new_df.sample(frac=train_size,random_state=200)
    test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)


    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    train_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    test_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    training_loader = DataLoader(
        train_set,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )
    testing_loader = DataLoader(
        test_set,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
    )

    device = 'cuda' if cuda.is_available() else 'cpu'
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position=512,
        token_types_size=2,
        hidden_size=768,
        layer_norm_eps=1e-12,
        dropout_p=0.1,
        attention_head_size=12,
        intermediate_size=3072,
        num_hidden_layers=12,
    )
    model = BertClassification(config).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

    epoch = 20
    for i in range(epoch):
        loss_values = []
        data_loader_step = 0
        for data in training_loader:
            optimizer.zero_grad()
            id = data['ids'].to(device, dtype=torch.long)
            attention_mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            model_output = model(id, attention_mask=attention_mask, token_type_ids=token_type_ids)
            targets = data['targets'].to(device, dtype=torch.float)
            loss_value = loss_func(model_output, targets)
            loss_value.backward()
            optimizer.step()
            loss_values.append(loss_value.item())
            data_loader_step += 1
            if data_loader_step % 500 == 0:
                print(f'\tstep={data_loader_step}, loss_value={loss_value}')
        print(f'epoch={i}, loss={np.mean(loss_value)}')
