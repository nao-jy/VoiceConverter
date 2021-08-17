import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle
from model import Generator, Discriminator
from tqdm import tqdm
from make_dataset import DataSet
import torch.nn as nn

class CycleGANTraining(object):
    def __init__(self, source_speaker_dir, target_speaker_dir, model_checkpoint=None):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.file_name = "train_log.txt"
        self.source_speaker = DataSet(source_speaker_dir)
        self.target_speaker = DataSet(target_speaker_dir)

        self.num_epochs = 500000
        self.batch_size = 1

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
                   list(self.discriminator_B.parameters())

        # Initial learning rates
        self.generator_lr = 2e-4  # 0.0002
        self.discriminator_lr = 1e-4  # 0.0001

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []
        self.discriminator_loss_store = []
        
        if model_checkpoint == None:
            self.start_epoch = 1
        else:
            self.start_epoch = self.loadModel(model_checkpoint)

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'source_mean': self.source_speaker.mean,
            'source_std': self.source_speaker.std,
            'target_mean': self.target_speaker.mean,
            'target_std': self.target_speaker.std
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch
    
    def train(self):

        for epoch in tqdm(range(self.start_epoch, self.num_epochs + 1)):
            start_time_epoch = time.time()
            # Constants
            cycle_loss_lambda = 10
            identity_loss_lambda = 1
            hinge_m = 0.5
            hinge = nn.ReLU()

            real_A = self.source_speaker.getitem()
            real_B = self.target_speaker.getitem()

            for _ in range(self.batch_size - 1):
                real_A = torch.cat([real_A, self.source_speaker.getitem()], dim=0)
                real_B = torch.cat([real_B, self.target_speaker.getitem()], dim=0)

            real_A = real_A.to(self.device).float()
            real_B = real_B.to(self.device).float()

            # Generator Training
            self.reset_grad()
            fake_B = self.generator_A2B(real_A)
            cycle_A = self.generator_B2A(fake_B)

            fake_A = self.generator_B2A(real_B)
            cycle_B = self.generator_A2B(fake_A)

            identity_A = self.generator_B2A(real_A)
            identity_B = self.generator_A2B(real_B)

            d_fake_A = self.discriminator_A(fake_A)
            d_fake_B = self.discriminator_B(fake_B)

            # Generator Cycle loss
            cycleLoss = torch.mean(torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

            # Generator Identity Loss
            identiyLoss = torch.mean(torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

            # Adversarial Loss
            generator_loss_A2B = torch.mean(hinge(-d_fake_B))
            generator_loss_B2A = torch.mean(hinge(-d_fake_A))

            # generator_loss_A2B = torch.mean((d_fake_B - 1) ** 2) / 2
            # generator_loss_B2A = torch.mean((d_fake_A - 1) ** 2) / 2

            # Total Generator Loss
            generator_loss = generator_loss_A2B + generator_loss_B2A + cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
            self.generator_loss_store.append(generator_loss.item())

            # Backprop for Generator
            generator_loss.backward()
            self.generator_optimizer.step()

            # Discriminator Training
            self.reset_grad()
            d_real_A = self.discriminator_A(real_A)
            d_real_B = self.discriminator_B(real_B)

            fake_A = self.generator_B2A(real_B)
            d_fake_A = self.discriminator_A(fake_A)

            fake_B = self.generator_A2B(real_A)
            d_fake_B = self.discriminator_B(fake_B)

            # Loss Functions
            d_loss_A_real = torch.mean(hinge(hinge_m - d_real_A))
            d_loss_A_fake = torch.mean(hinge(hinge_m + d_fake_A))
            d_loss_A = d_loss_A_real + d_loss_A_fake

            d_loss_B_real = torch.mean(hinge(hinge_m - d_real_B))
            d_loss_B_fake = torch.mean(hinge(hinge_m + d_fake_B))
            d_loss_B = d_loss_B_real + d_loss_B_fake

            # d_loss_A_real = torch.mean((d_real_A - 1) ** 2) / 2
            # d_loss_A_fake = torch.mean((d_fake_A - 0) ** 2) / 2
            # d_loss_A = d_loss_A_real + d_loss_A_fake

            # d_loss_B_real = torch.mean((d_real_B - 1) ** 2) / 2
            # d_loss_B_fake = torch.mean((d_fake_B - 0) ** 2) / 2
            # d_loss_B = d_loss_B_real + d_loss_B_fake

            # Final Loss for discriminator
            d_loss = d_loss_A + d_loss_B 
            self.discriminator_loss_store.append(d_loss.item())

            # Backprop for Discriminator
            d_loss.backward()

            self.discriminator_optimizer.step()

            if epoch % 2000 == 0 and epoch != 0:
                end_time = time.time()
                store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
                self.store_to_file(store_to_file)
                print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                    epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch))

                # Save the Entire model
                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(store_to_file)
                self.saveModelCheckPoint(epoch, 'CycleGAN_CheckPoint')
                print("Model Saved!")

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

if __name__ == "__main__":
    cycleGAN = CycleGANTraining(source_speaker_dir="source", target_speaker_dir="target", model_checkpoint=None)
    cycleGAN.train()