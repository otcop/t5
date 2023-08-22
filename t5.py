# fine tuning T5 pretrained model using the wikihow data
import torch 
import torch.nn as nn 

# load model from huggingface
from transformers import T5Tokenizer, T5ForConditionalGeneration 

model = T5ForConditionalGeneration.from_pretrained('t5-base') # The model 
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# load the wikihow dataset
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
dataset = load_dataset('wikihow', 'all', data_dir='../examples/data',split="train")



# define model configs
from dataclasses import dataclass
@dataclass
class data_config:
    input_length: int = 512
    output_length: int = 150
    num_samples: int = 2
    batch_size: int = 1
# define a dataset that using tokenizer
class wikihow(Dataset):
    def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):         
        self.dataset =  load_dataset('wikihow', 'all', data_dir='data/', split=type_path)
        if num_samples:
            self.dataset = self.dataset.select(list(range(0, num_samples)))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.print_text = print_text
  
    def __len__(self):
        return self.dataset.shape[0]
    
    def clean_text(self, text):
        text = text.replace('Example of text:', '')
        text = text.replace('Example of Summary:', '')
        text = text.replace('\n','')
        text = text.replace('``', '')
        text = text.replace('"', '')
        
        return text
    
    
    def convert_to_features(self, example_batch):
        # Tokenize contexts and questions (as pairs of inputs)
        
        if self.print_text:
            print("Input Text: ", self.clean_text(example_batch['text']))
        input_ = self.clean_text(example_batch['text']) + " </s>"
        target_ = self.clean_text(example_batch['headline']) + " </s>"
        
        input_ = self.clean_text(example_batch['text'])
        target_ = self.clean_text(example_batch['headline'])

        
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
    
       
        return source, targets
  
    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        
dataset =  wikihow(tokenizer=tokenizer, type_path='train', num_samples=data_config.num_samples,  input_length=data_config.input_length, 
                        output_length=data_config.output_length)
print(len(dataset))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=data_config.batch_size, shuffle=False)

for batch in train_loader:
    print(batch["source_ids"].shape)
    output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
    print(output.keys())
    print(output['loss'])
    break

def train(model, train_loader, optimizer):
    model.train()
    losses = torch.zeros(2)
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        losses[0] += loss.item()
        losses[1] += len(batch)
    train_loss = losses[0] / losses[1]
    return train_loss
optimizer = torch.optim.AdamW(model.parameters())
train(model, train_loader, optimizer)

epochs = 2
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer)
    print(train_loss)

