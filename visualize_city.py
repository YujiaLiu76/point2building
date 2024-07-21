import os
import numpy as np
from glob import glob
from tqdm import tqdm
import json
import src.utils.data_utils as data_utils


if __name__ == '__main__':
    CITY = "Zurich"
    mesh_files = sorted(glob(os.path.join('results', CITY, '*.obj')))
    with open(os.path.join('datasets', CITY, 'testset/info.json')) as f:
        data_info = json.load(f)

    material_list = []
    vs_list = []
    fs_list = []

    for i, mesh_file in enumerate(tqdm(mesh_files)):
        vertices, faces = data_utils.load_obj(mesh_file)
        one_data_info = data_info[os.path.split(mesh_file)[-1].split('.')[0]]
        vertices = vertices * one_data_info['scale'] + one_data_info['center']
        material_list.append(i)
        vs_list.append(vertices)
        fs_list.append(faces)

    offset_f = np.hstack([1, np.cumsum([len(item) for item in vs_list])[:-1]+1])
    new_fs_list = []
    for i, fs in enumerate(fs_list):
        new_fs_list.append([[x + offset_f[i] for x in row] for row in fs])
    vs_list = np.vstack(vs_list)

    with open(os.path.join('results', '{}_pred.obj'.format(CITY)), 'w') as wf:
        wf.write('mtllib cad.mtl\n')
        for v in vs_list:
            wf.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for j, new_fs in enumerate(new_fs_list):
            wf.write('usemtl Material{}\n'.format(material_list[j]))
            for new_f in new_fs:
                face_indices = ' '.join(str(idx) for idx in new_f)
                wf.write(f"f {face_indices}\n")



        print('ok')
    #TODO
    
    
    print('ok')



