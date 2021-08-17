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

        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        # Generator and Discriminator
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A_G = Discriminator(input_nc=1, n_layers=6).to(self.device)
        self.discriminator_B_G = Discriminator(input_nc=1, n_layers=6).to(self.device)
        self.discriminator_A_L = Discriminator(input_nc=1, n_layers=4).to(self.device)
        self.discriminator_B_L = Discriminator(input_nc=1, n_layers=4).to(self.device)

        # Optimizer
        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_G_params = list(self.discriminator_A_G.parameters()) + \
                   list(self.discriminator_B_G.parameters())
        d_L_params = list(self.discriminator_A_L.parameters()) + \
                   list(self.discriminator_B_L.parameters())

        # Initial learning rates
        self.generator_lr = 2e-4  # 0.0002
        self.discriminator_G_lr = 1e-4  # 0.0001
        self.discriminator_L_lr = 1e-4  # 0.0001

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_G_optimizer = torch.optim.Adam(
            d_G_params, lr=self.discriminator_G_lr, betas=(0.5, 0.999))
        self.discriminator_L_optimizer = torch.optim.Adam(
            d_L_params, lr=self.discriminator_L_lr, betas=(0.5, 0.999))

        # Storing Discriminatior and Generator Loss
        self.generator_loss_store = []
        self.discriminator_loss_store = []
        
        if model_checkpoint == None:
            self.start_epoch = 1
        else:
            self.start_epoch = self.loadModel(model_checkpoint)

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_G_optimizer.zero_grad()
        self.discriminator_L_optimizer.zero_grad()

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA_G': self.discriminator_A_G.state_dict(),
            'model_discriminatorB_G': self.discriminator_B_G.state_dict(),
            'model_discriminatorA_L': self.discriminator_A_L.state_dict(),
            'model_discriminatorB_L': self.discriminator_B_L.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_G_optimizer': self.discriminator_G_optimizer.state_dict(),
            'discriminator_L_optimizer': self.discriminator_L_optimizer.state_dict(),
            'source_mean': self.source_speaker.mean,
            'source_std': self.source_speaker.std,
            'target_mean': self.target_speaker.mean,
            'target_std': self.target_speaker.std
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH, map_location=self.device)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A_G.load_state_dict(
            state_dict=checkPoint['model_discriminatorA_G'])
        self.discriminator_B_G.load_state_dict(
            state_dict=checkPoint['model_discriminatorB_G'])
        self.discriminator_A_L.load_state_dict(
            state_dict=checkPoint['model_discriminatorA_L'])
        self.discriminator_B_L.load_state_dict(
            state_dict=checkPoint['model_discriminatorB_L'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_G_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_G_optimizer'])
        self.discriminator_L_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_L_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch
    
    def train(self):

        for epoch in tqdm(range(self.start_epoch, self.num_epochs + 1)):
            start_time_epoch = time.time()
            # Constants
            cycle_loss_lambda = 10
            identity_loss_lambda = 5
            cam_loss_lambda = 1000

            real_A = self.source_speaker.getitem()
            real_B = self.target_speaker.getitem()

            # dummy_mask = torch.ones((1, 1, real_A.shape[2], real_A.shape[3])).to(self.device).float()
            # mask_A = self.source_speaker.getmask().to(self.device).float()
            # mask_B = self.target_speaker.getmask().to(self.device).float()


            # if epoch > 10000:
            #     identity_loss_lambda = 0

            real_A = real_A.to(self.device).float()
            real_B = real_B.to(self.device).float()

            # Generator Training
            self.reset_grad()

            # Re-Reconstruction Loss

            fake_B, fake_A2B_cam_logit, _  = self.generator_A2B(real_A)
            cycle_A, _, _  = self.generator_B2A(fake_B)

            fake_B, _, _  = self.generator_A2B(cycle_A)
            cycle_A, _, _  = self.generator_B2A(fake_B)

            fake_A, fake_B2A_cam_logit, _  = self.generator_B2A(real_B)
            cycle_B, _, _  = self.generator_A2B(fake_A)

            fake_A, _, _  = self.generator_B2A(real_B)
            cycle_B, _, _  = self.generator_A2B(fake_A)

            identity_A, identity_A_cam_logit, _ = self.generator_B2A(real_A)
            identity_B, identity_B_cam_logit, _  = self.generator_A2B(real_B)

            d_fake_A_G, d_fake_A_G_cam, _ = self.discriminator_A_G(fake_A)
            d_fake_B_G, d_fake_B_G_cam, _ = self.discriminator_B_G(fake_B)

            # Use 2nd Step Adversarial Loss
            d_fake_A_L, d_fake_A_L_cam, _ = self.discriminator_A_L(cycle_A)
            d_fake_B_L, d_fake_B_L_cam, _ = self.discriminator_B_L(cycle_B)

            # Generator Cycle loss
            cycleLoss = torch.mean(torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

            # Generator Identity Loss
            identiyLoss = torch.mean(torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

            # Adversarial Loss
            generator_loss_A2B_G = torch.mean((d_fake_B_G - 1) ** 2) / 2 + torch.mean((d_fake_B_G_cam - 1) ** 2) / 2
            generator_loss_B2A_G = torch.mean((d_fake_A_G - 1) ** 2) / 2 + torch.mean((d_fake_A_G_cam - 1) ** 2) / 2
            generator_loss_A2B_L = torch.mean((d_fake_B_L - 1) ** 2) / 2 + torch.mean((d_fake_B_L_cam - 1) ** 2) / 2
            generator_loss_B2A_L = torch.mean((d_fake_A_L - 1) ** 2) / 2 + torch.mean((d_fake_A_L_cam - 1) ** 2) / 2
            adversarial_loss = generator_loss_A2B_G + generator_loss_B2A_G + generator_loss_A2B_L + generator_loss_B2A_L

            # CAM Loss: indicates what makes each speaker different
            # If the Generator can grasp each characteristics, this loss converges
            # TODO: When CAM Loss converge, increase cycle loss lambda to preserve lingual information
            # Adversarial loss is not enough, because it doesn't care weather the input is language

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(identity_A_cam_logit, torch.zeros_like(identity_A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(identity_B_cam_logit, torch.zeros_like(identity_B_cam_logit).to(self.device))
            cam_loss = G_cam_loss_A + G_cam_loss_B
            
            # Total Generator Loss
            generator_loss = adversarial_loss + cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss + cam_loss_lambda * cam_loss
            self.generator_loss_store.append(generator_loss.item())

            # for debug
            if epoch % 1 == 0:
                print("cycle loss: " + str((cycle_loss_lambda * cycleLoss).item()))
                print("cam loss:   " + str((cam_loss_lambda * cam_loss).item()))
                print("adv loss:   " + str(adversarial_loss.item()))
                print("id  loss:   " + str(identiyLoss.item()))

            # Backprop for Generator
            generator_loss.backward()
            self.generator_optimizer.step()

            # Discriminator Training
            self.reset_grad()
            fake_A, _, _ = self.generator_B2A(real_B)
            fake_B, _, _ = self.generator_A2B(real_A)
            cycle_A, _, _  = self.generator_B2A(fake_B)
            cycle_B, _, _  = self.generator_A2B(fake_A)

            d_real_A_G, d_real_A_G_cam, _ = self.discriminator_A_G(real_A)
            d_real_B_G, d_real_B_G_cam, _ = self.discriminator_B_G(real_B)
            d_fake_A_G, d_fake_A_G_cam, _ = self.discriminator_A_G(fake_A)
            d_fake_B_G, d_fake_B_G_cam, _ = self.discriminator_B_G(fake_B)

            d_real_A_L, d_real_A_L_cam, _ = self.discriminator_A_L(real_A)
            d_real_B_L, d_real_B_L_cam, _ = self.discriminator_B_L(real_B)
            d_fake_A_L, d_fake_A_L_cam, _ = self.discriminator_A_L(cycle_A)
            d_fake_B_L, d_fake_B_L_cam, _ = self.discriminator_B_L(cycle_B)

            # Loss Functions
            d_loss_A_real_G = torch.mean((d_real_A_G - 1) ** 2) / 2 + torch.mean((d_real_A_G_cam - 1) ** 2) / 2
            d_loss_A_fake_G = torch.mean((d_fake_A_G - 0) ** 2) / 2 + torch.mean((d_fake_A_G_cam - 0) ** 2) / 2
            d_loss_A_real_L = torch.mean((d_real_A_L - 1) ** 2) / 2 + torch.mean((d_real_A_L_cam - 1) ** 2) / 2
            d_loss_A_fake_L = torch.mean((d_fake_A_L - 0) ** 2) / 2 + torch.mean((d_fake_A_L_cam - 0) ** 2) / 2
            d_loss_A = d_loss_A_real_G + d_loss_A_fake_G + d_loss_A_real_L + d_loss_A_fake_L

            # Loss Functions
            d_loss_B_real_G = torch.mean((d_real_B_G - 1) ** 2) / 2 + torch.mean((d_real_B_G_cam - 1) ** 2) / 2
            d_loss_B_fake_G = torch.mean((d_fake_B_G - 0) ** 2) / 2 + torch.mean((d_fake_B_G_cam - 0) ** 2) / 2
            d_loss_B_real_L = torch.mean((d_real_B_L - 1) ** 2) / 2 + torch.mean((d_real_B_L_cam - 1) ** 2) / 2
            d_loss_B_fake_L = torch.mean((d_fake_B_L - 0) ** 2) / 2 + torch.mean((d_fake_B_L_cam - 0) ** 2) / 2
            d_loss_B = d_loss_B_real_G + d_loss_B_fake_G + d_loss_B_real_L + d_loss_B_fake_L


            # Final Loss for discriminator with the second step adverserial loss
            d_loss = d_loss_A + d_loss_B
            self.discriminator_loss_store.append(d_loss.item())

            # Backprop for Discriminator
            d_loss.backward()

            self.discriminator_G_optimizer.step()
            self.discriminator_L_optimizer.step()

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
    cycleGAN = CycleGANTraining(source_speaker_dir="source", target_speaker_dir="target", model_checkpoint="CycleGAN_CheckPoint")
    cycleGAN.train()