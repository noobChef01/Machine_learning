"""
Main agent for NMT attention based.
name: nmt_attention.py
date: Jan 2020
author: Sajid Mashroor
"""
import torch
from torch import nn
import torch.optim as optim

from graphs.models.layers.encoder import Encoder
from graphs.models.layers.decoder import Decoder    
from graphs.models.network import Seq2Seq
from datasets.de_en import DeEnDataLoader

from torchtext.datasets import Multi30k

import numpy as np
import shutil
from tqdm import tqdm
import math
import time

from tensorboardX import SummaryWriter

class NmtAttnAgent:

    def __init__(self, config):
        self.config = config
        # self.logger              

        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config["cuda"]

        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.config["seed"])
            torch.cuda.set_device(self.config["gpu_device"])
            #self.logger.info("Operation will be on *****GPU-CUDA***** ")
            print("Operation will be on *****GPU-CUDA***** ")
        else:
            self.device = torch.device("cpu")
            torch.cuda.manual_seed_all(self.config["seed"])
            #self.logger.info("Operation will be on *****CPU***** ")
            print("Operation will be on *****CPU***** ")

        self.model = Seq2Seq(Encoder(self.config["encoder"]), 
                             Decoder(self.config["decoder"]), 
                             self.device)
        self.optmiser = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        self.dataloader = DeEnDataLoader(self.config, self.device)
        TRG_PAD_IDX = self.dataloader.TRG.vocab.stoi[self.dataloader.TRG.pad_token]        
        self.loss = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)   
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)
        # self.check_pth = self.load_checkpoint(self.config["chck_pth"])
        # Tensorboard Writer
        self.sum_wrt = SummaryWriter(log_dir=self.config["log_dir"], comment='NMT')

    def run(self):
        """
        Run in given mode
        """

        try:
            if self.config["mode"] == "train":
                self.train()
            else:
                self.validate()
        
        except KeyboardInterrupt:
                #self.logger.info("You have entered CTRL+C.. Wait to finalize")
                print("You have entered CTRL+C.. Wait to finalize")


    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        best_epoch = 0
        cur_best_valid_loss = float('inf')
        for epoch in tqdm(range(self.config["no_epochs"])):
            start_time = time.time()
            train_loss = self.train_one_epoch()
            valid_loss = self.validate()
            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < cur_best_valid_loss:
                cur_best_valid_loss = valid_loss
                best_epoch = epoch
                self.save_checkpoint(epoch, self.model.state_dict())
            
            self.sum_wrt.add_scalar("train\loss", train_loss, epoch)
            self.sum_wrt.add_scalar("train\perplexity", math.exp(train_loss), epoch)
            self.sum_wrt.add_scalar(f"{self.config['mode']}\loss", valid_loss, epoch)
            self.sum_wrt.add_scalar(f"{self.config['mode']}\perplexity", math.exp(valid_loss), epoch)

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
            print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}")
        self.save_checkpoint(best_epoch, is_best=True)
        

    def train_one_epoch(self):
        """
        One epoch of training
        """

        self.model.train()
        epoch_loss = 0
        iterator = self.dataloader.train_iterator
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            self.optmiser.zero_grad()
            
            output = self.model(src, trg)
            
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]
            
            loss = self.loss(output, trg)
            if np.isnan(float(loss.item())):
                raise ValueError('Loss is nan during training...')
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip"])
            
            self.optmiser.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def validate(self):
        """
        One cycle of model validation
        """

        self.model.eval()    
        epoch_loss = 0    
        assert self.config.mode != "train", "mode is train in evaluation mode"
        iterator = self.dataloader.test_iterator if self.config.mode == "test" \
            else self.dataloader.valid_iterator
        with torch.no_grad():    
            for i, batch in enumerate(iterator):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, 0) #turn off teacher forcing

                #trg = [trg len, batch size]
                #output = [trg len, batch size, output dim]

                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)

                #trg = [(trg len - 1) * batch size]
                #output = [(trg len - 1) * batch size, output dim]

                loss = self.loss(output, trg)
                if np.isnan(float(loss.item())):
                    raise ValueError('Loss is nan during {self.config.mode}...')

                epoch_loss += loss.item()
            
        return epoch_loss / len(iterator)

    def load_checkpoint(self, epoch):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        """
        return torch.load(f"{self.config['model_pth']}{self.model.__class__.__name__}_{epoch}.pth.tar")

    def save_checkpoint(self, epoch, state_dict=None, is_best=False):
        torch.save(state_dict, f"{self.config['model_pth']}{self.model.__class__.__name__}_{epoch}.pth.tar")
        if is_best:
                shutil.copyfile(f"{self.config['model_pth']}{self.model.__class__.__name__}_{epoch}.pth.tar",
                                f"{self.config['model_pth']}{self.model.__class__.__name__}_best.pth.tar")

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def finalize(self):
        #self.logger.info("Please wait while finalizing the operation.. Thank you")
        print("Please wait while finalizing the operation.. Thank you")
        self.sum_wrt.export_scalars_to_json(f"{self.config['log_dir']}all_scalars.json".format())
        self.sum_wrt.close()