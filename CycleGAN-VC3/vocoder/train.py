import os
import math
import tqdm
import torch
import itertools
import traceback
import numpy as np
import model.ModifiedGenerator as ModifiedGenerator
import model.MultiScaleDiscriminator as MultiScaleDiscriminator
import stft_loss.MultiResolutionSTFTLoss as MultiResolutionSTFTLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def num_params(model, print_out=True):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    if print_out:
        print('Trainable Parameters: %.3fM' % parameters)

def train(pt_dir, chkpt_path, trainloader, valloader):
    model_g = ModifiedGenerator(80, 4).to(device)
    print("Generator : \n")
    num_params(model_g)
    model_d = MultiScaleDiscriminator().to(device)
    print("Discriminator : \n")
    num_params(model_d)
    optim_g = torch.optim.Adam(model_g.parameters(),
        lr=0.0001, betas=(0.5, 0.9))
    optim_d = torch.optim.Adam(model_d.parameters(),
        lr=0.0001, betas=(0.5, 0.9))

    init_epoch = -1
    step = 0

    if chkpt_path is not None:
        print("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path)
        model_g.load_state_dict(checkpoint['model_g'])
        model_d.load_state_dict(checkpoint['model_d'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        optim_d.load_state_dict(checkpoint['optim_d'])
        step = checkpoint['step']
        init_epoch = checkpoint['epoch']

    else:
        print("Starting new training run.")

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    try:
        model_g.train()
        model_d.train()
        stft_loss = MultiResolutionSTFTLoss()
        criterion = torch.nn.MSELoss().to(device)

        for epoch in itertools.count(init_epoch+1):

            trainloader.dataset.shuffle_mapping()
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
            avg_g_loss = []
            avg_d_loss = []
            avg_adv_loss = []
            for (melG, audioG), \
                (melD, audioD) in loader:
                melG = melG.to(device)      # torch.Size([16, 80, 64])
                audioG = audioG.to(device)  # torch.Size([16, 1, 16000])
                melD = melD.to(device)      # torch.Size([16, 80, 64])
                audioD = audioD.to(device)  #torch.Size([16, 1, 16000]
                # generator
                optim_g.zero_grad()
                fake_audio = model_g(melG)  # torch.Size([16, 1, 12800])
                fake_audio = fake_audio[:, :, :16000]


                sc_loss, mag_loss = stft_loss(fake_audio[:, :, :audioG.size(2)].squeeze(1), audioG.squeeze(1))
                loss_g = sc_loss + mag_loss

                adv_loss = 0.0

                if step > 100000:

                    disc_real = model_d(audioG)
                    disc_fake = model_d(fake_audio)
                    # for multi-scale discriminator

                    for feats_fake, score_fake in disc_fake:
                        # adv_loss += torch.mean(torch.sum(torch.pow(score_fake - 1.0, 2), dim=[1, 2]))
                        adv_loss += criterion(score_fake, torch.ones_like(score_fake))
                    adv_loss = adv_loss / len(disc_fake)  # len(disc_fake) = 3

                    # adv_loss = 0.5 * adv_loss

                    # loss_feat = 0
                    # feat_weights = 4.0 / (2 + 1) # Number of downsample layer in discriminator = 2
                    # D_weights = 1.0 / 7.0 # number of discriminator = 7
                    # wt = D_weights * feat_weights
                    if hp.model.feat_loss:
                        for (feats_fake, score_fake), (feats_real, _) in zip(disc_fake, disc_real):
                            for feat_f, feat_r in zip(feats_fake, feats_real):
                                adv_loss += hp.model.feat_match * torch.mean(torch.abs(feat_f - feat_r))

                    loss_g += hp.model.lambda_adv * adv_loss
            

                loss_g.backward()
                optim_g.step()

                # discriminator
                loss_d_avg = 0.0
                if step > 100000:
                    fake_audio = model_g(melD)[:, :, :16000]
                    fake_audio = fake_audio.detach()
                    loss_d_sum = 0.0
                    for _ in range(1):
                        optim_d.zero_grad()
                        disc_fake = model_d(fake_audio)
                        disc_real = model_d(audioD)
                        loss_d = 0.0
                        loss_d_real = 0.0
                        loss_d_fake = 0.0
                        for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
                            loss_d_real += criterion(score_real, torch.ones_like(score_real))
                            loss_d_fake += criterion(score_fake, torch.zeros_like(score_fake))
                        loss_d_real = loss_d_real / len(disc_real)  # len(disc_real) = 3
                        loss_d_fake = loss_d_fake / len(disc_fake)  # len(disc_fake) = 3
                        loss_d = loss_d_real + loss_d_fake
                        loss_d.backward()
                        optim_d.step()
                        loss_d_sum += loss_d
                    loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                    loss_d_avg = loss_d_avg.item()

                step += 1
                # logging
                loss_g = loss_g.item()
                avg_g_loss.append(loss_g)
                avg_d_loss.append(loss_d_avg)
                avg_adv_loss.append(adv_loss)
                if any([loss_g > 1e8, math.isnan(loss_g), loss_d_avg > 1e8, math.isnan(loss_d_avg)]):
                    print("loss_g %.01f loss_d_avg %.01f at step %d!" % (loss_g, loss_d_avg, step))
                    raise Exception("Loss exploded")

                if step % 1 == 0:
                    print("Avg : g %.04f d %.04f ad %.04f| step %d" % (sum(avg_g_loss) / len(avg_g_loss),
                                                                                sum(avg_d_loss) / len(avg_d_loss),
                                                                                sum(avg_adv_loss) / len(avg_adv_loss),
                                                                                step))
            if epoch % 20 == 0:
                save_path = os.path.join(pt_dir, '%04d.pt'
                    % (epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                }, save_path)
                print("Saved checkpoint to: %s" % save_path)

    except Exception as e:
        print("Exiting due to exception: %s" % e)
        traceback.print_exc()