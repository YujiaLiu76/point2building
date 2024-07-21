import os
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm

from src.modules.data_modules import load_dataloaders
from src.modules.pointcloud_encoder import PointCloudToVertexModel


def load_v_models(device, learning_rate=3e-4, split='train'):
    if split == 'train':
        pc_decoder_config = {
            "hidden_size": 384,
            "fc_size": 1024,
            "num_layers": 18,
            'dropout_rate': 0.2
        }
    elif split == 'test':
        pc_decoder_config = {
            "hidden_size": 384,
            "fc_size": 1024,
            "num_layers": 18,
            'dropout_rate': 0.0
        }

    pc_model = PointCloudToVertexModel(
        decoder_config = pc_decoder_config,
        max_num_input_verts = 100,
        quantization_bits = 8,
        learning_rate = learning_rate,
        gamma = 1.,
        device=device
    )

    return pc_model

def train_v_models(pc_dataloader, device="cuda"):
    epochs = 200
    lr_option = 'cos'
    max_lr = 3e-4
    warmup_steps = 10000
    
    pc_vertex_model = load_v_models(device, learning_rate=max_lr)
    pc_vertex_model = pc_vertex_model.to(device)
    if lr_option == 'cos':
        # using CosineAnnealingLR
        configured_optimizers = pc_vertex_model.configure_optimizers(total_steps=len(pc_dataloader)*epochs, warmup_steps=warmup_steps)
        pc_vertex_model_optimizer, pc_vertex_model_scheduler_warmup = configured_optimizers["optimizer"], configured_optimizers["scheduler_warmup"]
    elif lr_option == 'const':
        #using Constant LR
        pc_vertex_model_optimizer = torch.optim.Adam(pc_vertex_model.parameters(), lr=max_lr)

    checkpoint_save_path = './saved_model/{}/vertex_model'.format(CITY)
    if not os.path.exists(checkpoint_save_path):
        os.makedirs(checkpoint_save_path)
    
    for i in tqdm(range(0, epochs)):
        total_pc_vertex_loss = 0
        for j, (pc_vertex_batch) in enumerate(pc_dataloader):
            for k in pc_vertex_batch:
                if k != 'filenames':
                    pc_vertex_batch[k] = pc_vertex_batch[k].to(device)
            pc_vertex_logits = pc_vertex_model(pc_vertex_batch)
            pc_vertex_pred_dist = Categorical(logits = pc_vertex_logits)
            pc_vertex_loss = -torch.sum(pc_vertex_pred_dist.log_prob(pc_vertex_batch["vertices_flat"]) * pc_vertex_batch["vertices_flat_mask"])

            pc_vertex_loss.backward()

            nn.utils.clip_grad_norm_(pc_vertex_model.parameters(), max_norm=1.0)

            pc_vertex_model_optimizer.step()

            total_pc_vertex_loss += pc_vertex_loss.item()

            if lr_option == 'cos':
                pc_vertex_model_scheduler_warmup.step()                

        avg_pc_vertex_loss = total_pc_vertex_loss/len(pc_dataloader)
        if ((i + 1) % 10 == 0):
            print(f"Epoch {i + 1}: PC Vertex loss = {avg_pc_vertex_loss:.1f}, lr: {pc_vertex_model_optimizer.param_groups[0]['lr']}")
            with open(f'{checkpoint_save_path}/log.txt', 'a') as f:
                f.write(f"Epoch {i + 1}: PC Vertex loss = {avg_pc_vertex_loss:.1f}\n")
            checkpoint = {
                'epoch': i + 1,
                'state_dict': pc_vertex_model.state_dict(),
                'optimizer': pc_vertex_model_optimizer.state_dict(),
                'scheduler': pc_vertex_model_scheduler_warmup.state_dict(),
            }
            torch.save(checkpoint, '{}/checkpoint-{}.pth'.format(checkpoint_save_path, i + 1))


    return pc_vertex_model

if __name__=="__main__":
    CITY = "Zurich"
    pc_dataloader = load_dataloaders(batch_size=16, preprocess=True, data_split='train', CITY=CITY, stage=1)
    train_v_models(pc_dataloader, device='cuda')