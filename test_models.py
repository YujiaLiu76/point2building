import os
import numpy as np
from tqdm import tqdm
from glob import glob
import json
import torch
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import src.utils.data_utils as data_utils
from src.modules.data_modules import load_dataloaders
from train_vertex_model import load_v_models
from train_face_model import load_f_models

def process_data(predv):
    pred_vertices = torch.from_numpy(predv['vertices'][:predv['num_vertices']]).float().to(device)
    pred_vertices = data_utils.quantize_verts(pred_vertices)
    pred_vertices, inv = torch.unique(pred_vertices, dim=0, return_inverse=True)
    sort_inds = data_utils.torch_lexsort(pred_vertices.T)
    pred_vertices = pred_vertices[sort_inds]
    pred_vertices = pred_vertices.to(torch.int32)
    pred_vertices = data_utils.dequantize_verts(pred_vertices)

    face_batch = {}
    face_batch["vertices"] = pred_vertices.unsqueeze(0)
    face_batch["vertices_mask"] = torch.ones_like(pred_vertices[..., 0], dtype=torch.float32).unsqueeze(0)
    face_batch["files_list"] = [item.replace('/pointclouds/', '/meshes/').replace('.xyz', '.obj') for item in pc_vertex_batch['filenames']]

    return face_batch

def sample_vertices(vertex_model, vertex_batch):
    with torch.no_grad():
        vertex_samples = vertex_model.sample_mask(context = vertex_batch, num_samples = vertex_batch["vertices_flat"].shape[0],
                                            max_sample_length = 100, top_p = 0.9, recenter_verts = False, only_return_complete = False)
    out_dict = {}
    out_dict["vertices"] = vertex_samples["vertices"][0].cpu().numpy()
    out_dict["num_vertices"] = vertex_samples["num_vertices"][0].cpu().numpy()
    return out_dict

def sample_faces(face_model, face_batch, top_p=0.9, return_mesh=False):
    with torch.no_grad():
        face_samples = face_model.sample_mask(context = face_batch, max_sample_length = 500, top_p = top_p, only_return_complete = False)
    curr_faces = face_samples["faces"][0]
    num_face_indices = face_samples['num_face_indices'][0]
    pred_faces = data_utils.unflatten_faces(curr_faces[:num_face_indices].detach().cpu().numpy())

    return pred_faces

def v_have_stop_token(vs):
    if len(vs) < 100:
        return True
    else:
        return False
    
def f_have_stop_token(fs):
    def compute_len_fs(fs):
        return len([item for sublist in fs for item in sublist])+len(fs)+1
    len_f = compute_len_fs(fs)
    if len(fs) < 500:
        return True
    else:
        return False

def is_floor_covering_pointcloudxy(vs, pts, info, coverage_rate_thres=0.7):
    vs_floor_inds = np.where(vs[:,-1]<vs[:,-1].min()+0.5/info['scale'])[0]
    points1 = vs[vs_floor_inds][:,:2]
    points2 = pts[:,:2]

    if len(points1) < 3:
        return False, None

    hull1 = ConvexHull(points1)
    hull2 = ConvexHull(points2)

    poly1 = Polygon(points1[hull1.vertices])
    poly2 = Polygon(points2[hull2.vertices])

    intersection = poly1.intersection(poly2)
    coverage_rate = intersection.area / poly2.area

    return coverage_rate > coverage_rate_thres, vs_floor_inds

def are_missing_floor_vertices(vs, vs_floor_inds):
    vs_floor = vs[vs_floor_inds][:,:2]
    hull = ConvexHull(vs_floor)
    hull_points = vs_floor[hull.vertices]

    angles = []
    num_points = len(hull_points)
    for i in range(num_points):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % num_points]
        p3 = hull_points[(i + 2) % num_points]
        angle = data_utils.calculate_angle(p1, p2, p3)
        angles.append(angle)

    return any(angle < 60 for angle in angles)

def are_missing_floor_faces(vs, fs, vs_floor_inds):
    floor_f_bool_list = [all(element in list(vs_floor_inds) for element in f) for f in fs]
    
    if np.sum(floor_f_bool_list) == 0:
        return True
    else:
        result_floor_fs = [fs[_] for _, mask_value in enumerate(floor_f_bool_list) if mask_value]
        valid_poly_bools = []
        for one_result_floor_fs in result_floor_fs:
            valid_poly_bools.append(Polygon(vs[:,:2][one_result_floor_fs]).is_valid)
        return not(all(valid_poly_bools))



if __name__ == '__main__':
    CITY = 'Zurich'
    device = 'cuda'
    model_base_dir = './saved_model'
    test_data_dir = 'datasets/{}/testset'.format(CITY)

    with open(os.path.join('datasets', CITY, 'testset/info.json')) as f:
        data_info = json.load(f)

    checkpoint_v_dir = os.path.join(model_base_dir, CITY, 'vertex_model')
    checkpoint_v_pth = os.path.join(checkpoint_v_dir, 'checkpoint_v.pth')
    checkpoint_v = torch.load(checkpoint_v_pth)
    pc_vertex_model = load_v_models(device=device)
    pc_vertex_model = pc_vertex_model.to(device)
    pc_vertex_model.load_state_dict(checkpoint_v['state_dict'])

    checkpoint_f_dir = os.path.join(model_base_dir, CITY, 'face_model')
    checkpoint_f_pth = os.path.join(checkpoint_f_dir, 'checkpoint_f.pth')
    checkpoint_f = torch.load(checkpoint_f_pth)
    face_model = load_f_models(device=device)
    face_model = face_model.to(device)
    face_model.load_state_dict(checkpoint_f['state_dict'])

    all_pointcloud_files = sorted(glob(os.path.join(test_data_dir, 'pointclouds', '*.xyz')))
    pc_dataloader = load_dataloaders(batch_size=1, preprocess=True, data_split='test', CITY=CITY, stage=1)

    for j, (pc_vertex_batch) in enumerate(tqdm(pc_dataloader)):
        for k in pc_vertex_batch:
            if k != 'filenames':
                pc_vertex_batch[k] = pc_vertex_batch[k].to(device)
        one_data_info = data_info[os.path.split(pc_vertex_batch['filenames'][0])[-1].split('.')[0]]
        run_v_id = 0
        while run_v_id < 10:
            predv = sample_vertices(pc_vertex_model, pc_vertex_batch)
            if not v_have_stop_token(predv['vertices']):
                run_v_id += 1
                continue
            run_f_id = 0
            face_batch = process_data(predv)
            while run_f_id < 10:
                # print('run {} iterations'.format(run_v_id*10+run_f_id))
                predf = sample_faces(face_model, face_batch, return_mesh=True)
                if not f_have_stop_token(predf):
                    run_f_id += 1
                    continue
                if_floor_cover, vs_floor_inds = is_floor_covering_pointcloudxy(predv['vertices'], data_utils.dequantize_verts(pc_vertex_batch['pc_coords'][:,1:], 8).detach().cpu().numpy(), one_data_info)
                if vs_floor_inds is None:
                    break
                if not if_floor_cover:
                    if are_missing_floor_vertices(predv['vertices'], vs_floor_inds):
                        break
                    elif are_missing_floor_faces(predv['vertices'], predf, vs_floor_inds):
                        run_f_id += 1
                        continue

                out_file = os.path.join('results', CITY, os.path.split(face_batch['files_list'][0])[-1])
                data_utils.process_and_save_mesh(vertices=face_batch['vertices'][0].detach().cpu().numpy(), faces=predf, file_path=out_file, precess_dup=True)
                run_v_id = 10  
                break

            run_v_id += 1