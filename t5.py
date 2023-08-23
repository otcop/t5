# fine tuning T5 pretrained model using the wikihow data
import torch 
import torch.nn as nn 
import os
import functools
# load model from huggingface
from transformers import T5Tokenizer, T5ForConditionalGeneration 

# model = T5ForConditionalGeneration.from_pretrained('t5-base') # The model 
# tokenizer = T5Tokenizer.from_pretrained('t5-base')

# load the wikihow dataset
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

datafolder = '../examples/data'

# add distributed package
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers.models.t5.modeling_t5 import T5Block
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
'''
Dataset is made up of three parts: text, headline, title
the text is the input and headline is the target we want to generate 
'''

# define model configs
from dataclasses import dataclass
@dataclass
class data_config:
    input_length: int = 512
    output_length: int = 150
    num_samples: int = 2
    batch_size: int = 2
    num_val_samples: int = 1
    test_batch_size: int = 2

@dataclass
class model_config:
    device: str = 'cpu'
    lr: float = 0.001


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
        


# Train the network
def train(model, train_loader, optimizer):
    model.train()
    losses = torch.zeros(2)
    device = os.environ['device']
    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        losses[0] += loss.item()
        losses[1] += len(batch)
    train_loss = losses[0] / losses[1]
    return train_loss

@torch.inference_mode()
def validation(model, val_loader):
    losses = torch.zeros(2)

    for batch in train_loader:
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"],labels=batch["target_ids"] )
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        losses[0] += loss.item()
        losses[1] += len(batch)
    val_loss = losses[0] / losses[1]
    return val_loss



def fsdp_main(args):
    model = T5ForConditionalGeneration.from_pretrained('t5-base') # The model 
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dataset = load_dataset('wikihow', 'all', data_dir=datafolder,split="train")

    dataset =  wikihow(tokenizer=tokenizer, type_path='train', num_samples=data_config.num_samples,  input_length=data_config.input_length, 
                            output_length=data_config.output_length)
    print(len(dataset))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=data_config.batch_size, shuffle=False)

    val_dataset =  wikihow(tokenizer=tokenizer, type_path='train', num_samples=data_config.num_val_samples,  input_length=data_config.input_length, 
                            output_length=data_config.output_length)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=data_config.test_batch_size)

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)

    device = "cuda"
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=model_config.lr)


    t5_auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        T5Block,
    })

    model = FSDP(model,
    # process_group=process_group_fsdp,       
    # auto_wrap_policy=t5_auto_wrap_policy,
    # mixed_precision=mixed_precision_policy,
    # sharding_strategy=fsdp_config.sharding_strategy,
    device_id=torch.cuda.current_device())#,
    # limit_all_gathers=fsdp_config.limit_all_gathers)

