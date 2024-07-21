from enum import Enum
import os
import numpy as np
from typing import List, Dict, Optional
import json
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import src.utils.data_utils as data_utils
import MinkowskiEngine as ME


class PVDataset(Dataset):
    def __init__(self, all_pointcloud_files: List[str], data_info_file: str, preprocess:bool=False, rotatexy=True) -> None:
        """Initializes PointCloud Dataset

        Args:
            training_dir: Where model files along with renderings are located
            preprocess: Whether to apply filter
        """
        self.pointclouds = all_pointcloud_files
        self.rotatexy = rotatexy
        with open(data_info_file) as f:
            self.data_info = json.load(f)

        if preprocess:
            selected_file_idx = []
            for idx in tqdm(range(len(all_pointcloud_files))):
                pts_file = all_pointcloud_files[idx]
                mesh_file = pts_file.replace('/pointclouds/', '/meshes/').replace('.xyz', '.obj')
                if data_utils.filter_mesh_obj_by_info(self.data_info, mesh_file):
                    selected_file_idx.append(idx)
            self.pointclouds = [all_pointcloud_files[idx] for idx in selected_file_idx]
            print('======num files: {}========'.format(len(self.pointclouds)))

    def __len__(self) -> int:
        return len(self.pointclouds)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Gets pointcloud object along with associated mesh

        Args:
            idx: Index of pointcloud to retrieve

        Returns:
            mesh_dict: Dictionary containing vertices, faces of .obj file and image tensor
        """
        pc_file = self.pointclouds[idx]
        mesh_file = pc_file.replace('/pointclouds/', '/meshes/').replace('.xyz', '.obj')

        vertices, faces = data_utils.load_obj(mesh_file)
        pts = data_utils.load_xyz(pc_file)

        minz = -np.max(vertices[:,2])
        added_pts = pts.copy()
        added_pts[:,2] = minz
        pts = np.vstack((pts, added_pts))

        vertices = torch.from_numpy(vertices).float()
        pts = torch.from_numpy(pts).float()

        if self.rotatexy:
            angle_rad = np.random.uniform(0, 2*np.pi)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            vertices = data_utils.rotate_points(vertices, rotation_2d)
            pts = data_utils.rotate_points(pts, rotation_2d)

            vertices = torch.from_numpy(vertices).float()

        vertices, vertices_scale = data_utils.normalize_vertices_scale(vertices, return_scale=True)
        pts = pts / vertices_scale

        pts = pts.float()
        pts = torch.clamp(pts, -0.5, 0.5)
        vertices = torch.clamp(vertices, -0.5, 0.5)

        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        vertices = vertices.to(torch.int32)

        mesh_dict = {"vertices": vertices, "pointcloud": pts, "filename": pc_file}

        return mesh_dict
    
class VFDataset(Dataset):
    def __init__(
        self,
        all_mesh_files: Optional[List[str]] = None,
        data_info_file: str = None,
        preprocess: bool = False,
        rotatexy: bool = True
    ) -> None:
        """
        Args:
            training_dir: Root folder of shapenet dataset
        """

        self.all_mesh_files = all_mesh_files
        with open(data_info_file) as f:
            self.data_info = json.load(f)
        self.rotatexy = rotatexy
        
        if preprocess:
            selected_file_idx = []
            for idx in tqdm(range(len(self.all_mesh_files))):
                mesh_file = self.all_mesh_files[idx]
                if data_utils.filter_mesh_obj_by_info(self.data_info, mesh_file):
                    selected_file_idx.append(idx)
            self.all_mesh_files = [self.all_mesh_files[idx] for idx in selected_file_idx]
            print('======num files: {}========'.format(len(self.all_mesh_files)))

    def __len__(self) -> int:
        """Returns number of 3D objects"""
        return len(self.all_mesh_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns processed vertices, faces and class label of a mesh
        Args:
            idx: Which 3D object we're retrieving
        Returns:
            mesh_dict: Dictionary containing vertices, faces and class label
        """
        mesh_file = self.all_mesh_files[idx]
        
        vertices, faces = data_utils.load_obj(mesh_file)
        vertices = torch.from_numpy(vertices)
        if self.rotatexy:
            angle_rad = np.random.uniform(0, 2*np.pi)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            vertices = data_utils.rotate_points(vertices, rotation_2d)
            vertices = torch.from_numpy(vertices).float()
        vertices, vertices_scale = data_utils.normalize_vertices_scale(vertices, return_scale=True)
        vertices = torch.clamp(vertices, -0.5, 0.5)
        vertices, faces, _ = data_utils.quantize_process_mesh(vertices, faces)
        faces = data_utils.flatten_faces(faces)
        vertices = vertices.to(torch.int32)
        faces = faces.to(torch.int32)

        mesh_dict = {"vertices": vertices, "faces": faces, "filename": mesh_file}

        return mesh_dict
    

class CollateMethod(Enum):
    VERTICES = 1
    FACES = 2

class PolygenDataModule(nn.Module):
    def __init__(
        self,
        collate_method: CollateMethod,
        batch_size: int,
        quantization_bits: int = 8,
        apply_random_shift_vertices: bool = True,
        apply_random_shift_faces: bool = True,
        shuffle_vertices: bool = True,
        apply_preprocess: bool = False, 
        rotatexy: bool = True, 
        all_pointcloud_files: Optional[List[str]] = None,
        all_mesh_files: Optional[List[str]] = None,
        data_info_file: str = None,
    ) -> None:
        """
        Args:
            data_dir: Root directory for dataset
            collate_method: Whether to collate vertices or faces
            batch_size: How many 3D objects in one batch
            quantization_bits: How many bits we are using to quantize the vertices
            apply_random_shift_vertices: Whether or not we're applying random shift to vertices for vertex model
            apply_random_shift_faces: Whether or not we're applying random shift to vertices for face model
            shuffle_vertices: Whether or not we're shuffling the order of vertices during batch generation for face model
            apply_preprocess: Filter out objs that have too many vertices or faces
            rotatexy: Whether or not we're rotating inputs
            all_pointcloud_files: List of all .xyz files.
        """
        super().__init__()

        self.batch_size = batch_size

        self.quantization_bits = quantization_bits
        self.apply_random_shift_vertices = apply_random_shift_vertices
        self.apply_random_shift_faces = apply_random_shift_faces
        self.shuffle_vertices = shuffle_vertices

        if collate_method == CollateMethod.VERTICES:
            self.pv_dataset = PVDataset(all_pointcloud_files=all_pointcloud_files, data_info_file=data_info_file, preprocess=apply_preprocess, rotatexy=rotatexy)
            self.collate_fn = self.collate_vertex_model_batch
        elif collate_method == CollateMethod.FACES:
            self.vf_dataset = VFDataset(all_mesh_files=all_mesh_files, data_info_file=data_info_file, preprocess=apply_preprocess, rotatexy=rotatexy)
            self.collate_fn = self.collate_face_model_batch

    def collate_vertex_model_batch(self, ds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        pc_vertex_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]

        max_vertices = max(num_vertices_list)
        num_elements = len(ds)
        vertices_flat = torch.zeros([num_elements, max_vertices * 3 + 1])
        vertices_flat_mask = torch.zeros_like(vertices_flat)
        filenames = []

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            initial_vertex_size = vertices.shape[0]
            padding_size = max_vertices - initial_vertex_size
            vertices_permuted = torch.stack([vertices[..., 2], vertices[..., 1], vertices[..., 0]], dim=-1)
            curr_vertices_flat = vertices_permuted.reshape([-1])
            vertices_flat[i] = F.pad(curr_vertices_flat + 1, [0, padding_size * 3 + 1])[None]

            vertices_flat_mask[i] = torch.zeros_like(vertices_flat[i], dtype=torch.float32)
            vertices_flat_mask[i, : initial_vertex_size * 3 + 1] = 1

            filenames.append(element["filename"])        

        pc_coords_list = []
        pc_feats_list = []
        for item in ds:
            voxel_dict = {}
            pc_coords_tmp = data_utils.quantize_verts(item["pointcloud"])

            for m in range(pc_coords_tmp.shape[0]):
                coord_tuple = (pc_coords_tmp[m,0].item(), pc_coords_tmp[m,1].item(), pc_coords_tmp[m,2].item())
                if(coord_tuple not in voxel_dict):
                    voxel_dict[coord_tuple] = torch.concat([item["pointcloud"][m],torch.tensor([1.0])], dim=-1)
                else:
                    voxel_dict[coord_tuple] += torch.concat([item["pointcloud"][m],torch.tensor([1.0])], dim=-1)

            locations = torch.tensor(list(voxel_dict.keys()))
            features = torch.stack(list(voxel_dict.values()))
            features = features/features[:,3:]

            pc_coords_list.append(locations)
            features = torch.cat([data_utils.quantize_verts(features[:,:3]), features[:,-1:]], dim=-1)
            pc_feats_list.append(features)

        pc_coords, pc_feats = ME.utils.sparse_collate(pc_coords_list, pc_feats_list)

        pc_vertex_model_batch["vertices_flat"] = vertices_flat
        pc_vertex_model_batch["vertices_flat_mask"] = vertices_flat_mask
        pc_vertex_model_batch["pc_coords"] = pc_coords
        pc_vertex_model_batch["pc_feats"] = pc_feats
        pc_vertex_model_batch["filenames"] = filenames
        return pc_vertex_model_batch
    
    def collate_face_model_batch(
        self,
        ds: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Applies padding to different length face sequences so we can batch them
        Args:
            ds: List of dictionaries with each dictionary containing info about a specific 3D object

        Returns:
            face_model_batch: A single dictionary which represents the whole face model batch
        """
        face_model_batch = {}
        num_vertices_list = [shape_dict["vertices"].shape[0] for shape_dict in ds]
        max_vertices = max(num_vertices_list)
        num_faces_list = [shape_dict["faces"].shape[0] for shape_dict in ds]
        max_faces = max(num_faces_list)
        num_elements = len(ds)

        shuffled_faces = torch.zeros([num_elements, max_faces], dtype=torch.int32)
        face_vertices = torch.zeros([num_elements, max_vertices, 3])
        face_vertices_mask = torch.zeros([num_elements, max_vertices], dtype=torch.int32)
        faces_mask = torch.zeros_like(shuffled_faces, dtype=torch.int32)
        filenames = []

        for i, element in enumerate(ds):
            vertices = element["vertices"]
            num_vertices = vertices.shape[0]
            if self.apply_random_shift_faces:
                vertices = data_utils.random_shift(vertices)

            if self.shuffle_vertices:
                permutation = torch.randperm(num_vertices)
                vertices = vertices[permutation]
                vertices = vertices.unsqueeze(0)
                face_permutation = torch.cat(
                    [
                        torch.Tensor([0, 1]).to(torch.int32),
                        torch.argsort(permutation).to(torch.int32) + 2,
                    ],
                    dim=0,
                )
                curr_faces = face_permutation[element["faces"].to(torch.int64)][None]
            else:
                curr_faces = element["faces"][None]

            vertex_padding_size = max_vertices - num_vertices
            initial_faces_size = curr_faces.shape[1]
            face_padding_size = max_faces - initial_faces_size
            shuffled_faces[i] = F.pad(curr_faces, [0, face_padding_size, 0, 0])
            curr_verts = data_utils.dequantize_verts(vertices, self.quantization_bits)
            face_vertices[i] = F.pad(curr_verts, [0, 0, 0, vertex_padding_size])
            face_vertices_mask[i] = torch.zeros_like(face_vertices[i][..., 0], dtype=torch.float32)
            face_vertices_mask[i, :num_vertices] = 1
            faces_mask[i] = torch.zeros_like(shuffled_faces[i], dtype=torch.float32)
            faces_mask[i, : initial_faces_size + 1] = 1
            filenames.append(element["filename"])

        face_model_batch["faces"] = shuffled_faces
        face_model_batch["vertices"] = face_vertices
        face_model_batch["vertices_mask"] = face_vertices_mask
        face_model_batch["faces_mask"] = faces_mask
        face_model_batch["filenames"] = filenames

        return face_model_batch

        
    def train_pv_dataloader(self, num_workers=8) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.pv_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
        )
    
    def test_pv_dataloader(self, num_workers=8) -> DataLoader:
        """
        Returns:
            test_dataloader: Dataloader used to load test batches
        """
        return DataLoader(
            self.pv_dataset,
            self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
        )    
    
    def train_vf_dataloader(self, num_workers=8) -> DataLoader:
        """
        Returns:
            train_dataloader: Dataloader used to load training batches
        """
        return DataLoader(
            self.vf_dataset,
            self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=True,
        )

def load_dataloaders(batch_size=4, preprocess=False, data_split='train', CITY='Zurich', stage=1):
    data_dir = os.path.join('datasets', CITY, '{}set'.format(data_split))
    rotatexy = data_split == 'train'
    data_info_file = os.path.join(data_dir, 'info.json')

    if stage == 1:
        all_pointcloud_files = glob(os.path.join(data_dir, 'pointclouds/*.xyz'))
        pv_data_module = PolygenDataModule(
                                        collate_method = CollateMethod.VERTICES,
                                        batch_size = batch_size,
                                        apply_random_shift_vertices = False,
                                        apply_preprocess=preprocess,
                                        rotatexy=rotatexy,
                                        all_pointcloud_files=all_pointcloud_files,
                                        data_info_file=data_info_file,
        )
        if data_split == 'train':
            pv_dataloader = pv_data_module.train_pv_dataloader(num_workers=8)
        elif data_split == 'test':
            pv_dataloader = pv_data_module.test_pv_dataloader(num_workers=8)

        return pv_dataloader

    elif stage == 2:
        all_mesh_files = glob(os.path.join(data_dir, 'meshes/*.obj'))
        apply_random_shift_faces = data_split == 'train'
        face_data_module = PolygenDataModule(
                                            collate_method = CollateMethod.FACES,
                                            batch_size = batch_size,
                                            apply_random_shift_faces = apply_random_shift_faces,
                                            apply_preprocess=preprocess,
                                            shuffle_vertices = False,
                                            rotatexy=rotatexy,
                                            all_mesh_files=all_mesh_files,
                                            data_info_file=data_info_file,                            
        )

        vf_dataloader = face_data_module.train_vf_dataloader()

        return vf_dataloader