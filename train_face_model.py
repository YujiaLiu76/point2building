import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from tqdm import tqdm
from src.modules.data_modules import PolygenDataModule, CollateMethod
from src.modules.face_model import FaceModel
from src.modules.data_modules import load_dataloaders

def load_f_models(device, split='train'):
    if split == 'train':
        face_transformer_config = {
            'hidden_size': 256,
            'fc_size': 1024,
            'num_layers': 12,
            'dropout_rate': 0.2,
        }
    elif split == 'test':
        face_transformer_config = {
            'hidden_size': 256,
            'fc_size': 1024,
            'num_layers': 12,
            'dropout_rate': 0.
        }

    face_model = FaceModel(encoder_config = face_transformer_config,
                           decoder_config = face_transformer_config,
                           class_conditional = False,
                           max_seq_length = 500,
                           quantization_bits = 8,
                           decoder_cross_attention = True,
                           use_discrete_vertex_embeddings = True,
                           learning_rate = 3e-4,
                           gamma = 0.995,
                           device = device,
                          )

    return face_model

def train_f_models(face_dataloader, device="cuda"):
    epochs = 200
    
    checkpoint_save_path = './saved_model/{}/face_model'.format(CITY)
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)
    face_model = load_f_models(device=device, split='train')
    face_model = face_model.to(device)
    configured_optimizers = face_model.configure_optimizers(total_steps=len(face_dataloader)*epochs)
    face_model_optimizer, face_model_scheduler_warmup = configured_optimizers["optimizer"], configured_optimizers["scheduler_warmup"]
    for i in tqdm(range(0, epochs)):
        total_face_loss = 0
        for j, (face_batch) in enumerate(face_dataloader):
            face_model_optimizer.zero_grad()
            for k in face_batch:
                if k != 'filenames' and k != 'files_list':
                    face_batch[k] = face_batch[k].to(device)
            face_logits = face_model(face_batch)

            face_pred_dist = Categorical(logits = face_logits)

            face_loss = -torch.sum(face_pred_dist.log_prob(face_batch["faces"]) * face_batch["faces_mask"])

            face_loss.backward()

            nn.utils.clip_grad_norm_(face_model.parameters(), max_norm=1.0)

            face_model_optimizer.step()

            total_face_loss += face_loss.item()
            face_model_scheduler_warmup.step()

        avg_face_loss = total_face_loss/len(face_dataloader)
        if ((i + 1) % 20 == 0):
            print(f"Epoch {i + 1}: Face loss = {avg_face_loss:.1f}, lr: {face_model_optimizer.param_groups[0]['lr']}")
            with open(f'{checkpoint_save_path}/log.txt', 'a') as f:
                f.write(f"Epoch {i + 1}: Face loss = {avg_face_loss:.1f}\n")
            checkpoint = {
                'epoch': i + 1,
                'state_dict': face_model.state_dict(),
                'optimizer': face_model_optimizer.state_dict()
            }
            torch.save(checkpoint, '{}/checkpoint-{}.pth'.format(checkpoint_save_path, i + 1))

    return face_model


if __name__=="__main__":
    CITY = "Zurich"
    face_dataloader = load_dataloaders(batch_size=16, preprocess=True, CITY=CITY, stage=2)
    train_f_models(face_dataloader, device='cuda')