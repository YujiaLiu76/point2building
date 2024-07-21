import os
import numpy as np
from typing import List, Tuple, Dict, Optional
import networkx as nx
import torch
from .truncated_normal import TruncatedNormal

MIN_RANGE = -0.5
MAX_RANGE = 0.5

def load_obj(poly_file):
    vs = []
    fs = []
    with open(poly_file) as f:
        for line in f.readlines():
            if line[0] == 'v':
                vxyz = line.strip().split(' ')
                vs.append([float(vxyz[1]), float(vxyz[2]), float(vxyz[3])])
            elif line[0] == 'f':
                a = line.strip().split(' ')
                fs.append([int(f)-1 for f in a[1:]])
    return np.array(vs), fs

def load_xyz(xyz_file):
    """Load point cloud file"""
    pts = []
    with open(xyz_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line == '':
                continue
            xyz = line.split(' ')
            pts.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
    return np.array(pts).astype(np.float32)

def center_vertices_np(vertices, return_center=False):
    """Translate vertices so that the bounding box is centered at zero

    Args:
        vertices: np array of shape (num_vertices, 3)

    Returns:
        centered_vertices: centered vertices in array of shape (num_vertices, 3)
    """
    vert_min = np.min(vertices, axis=0)
    vert_max = np.max(vertices, axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    centered_vertices = vertices - vert_center
    if return_center:
        return centered_vertices, vert_center
    else:
        return centered_vertices
    
def normalize_vertices_scale_np(vertices, return_scale=False):
    """Scale vertices so that the long diagonal of the bounding box is one.

    Args:
        vertices: unscaled vertices of shape (num_vertices, 3)

    Returns:
        scaled_vertices: scaled vertices of shape (num_vertices, 3)
    """
    vert_min = np.min(vertices, axis=0)
    vert_max = np.max(vertices, axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents ** 2))
    scaled_vertices = vertices / scale
    if return_scale:
        return scaled_vertices, scale
    else:
        return scaled_vertices

def normalize_vertices_scale(vertices: torch.Tensor, return_scale=False) -> torch.Tensor:
    """Scale vertices so that the long diagonal of the bounding box is one

    Args:
        vertices: unscaled vertices of shape (num_vertices, 3)
    Returns:
        scaled_vertices: scaled vertices of shape (num_vertices, 3)
    """
    vert_min, _ = torch.min(vertices, dim=0)
    vert_max, _ = torch.max(vertices, dim=0)
    extents = vert_max - vert_min
    scale = torch.sqrt(torch.sum(extents ** 2))
    scaled_vertices = vertices / scale
    if return_scale:
        return scaled_vertices, scale
    else:
        return scaled_vertices
    
def save_mesh(out_file, vertices, faces):
    with open(out_file, 'w') as wf:
        wf.write('mtllib cad.mtl\n')
        for v in vertices:
            wf.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            a = "f"
            for i in face:
                a += " {}".format(i+1)
            a += "\n"
            wf.write(a)

def save_pointcloud(out_file, pts):
    np.savetxt(out_file, pts)

def process_and_save_mesh(
    vertices: np.ndarray,
    faces: List[List[int]],
    file_path: str,
    transpose: bool = False,
    scale: float = 1.0,
    precess_dup = False,
) -> None:
    """Writes vertices and faces to .obj file to represent 3D object
    Args:
        vertices: array of shape (num_vertices, 3) representing vertex indices
        faces: List of vertex indices representing vertex connectivity
        file_path: Where to save .obj file
        transpose: boolean representing whether to change traditional order of (x, y, z)
        scale: Factor by which to scale vertices
    """
    if len(faces) == 0:
        return
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
    vertices *= scale
    # if precess_dup:
    #     new_faces = []
    #     for lst in faces:
    #         # merged_lst = [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i-1]]
    #         arr = np.array(lst)
    #         unique, index, inverse = np.unique(arr, return_index=True, return_inverse=True)
    #         unique_sorted = unique[np.argsort(index)]
    #         merged_lst = list(unique_sorted)
    #         new_faces.append(merged_lst)
    #     faces = new_faces
    #=============
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts
            if c_length > 2:
                d = argmin(c)
                # Cyclically permute faces so that the first index is the smallest
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    #=============
    if faces is not None and len(faces) != 0:
        if min(min(faces)) == 0:
            f_add = 1
        else:
            f_add = 0
    with open(file_path, "w") as f:
        f.write('mtllib cad.mtl\n')
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        # f.write('usemtl Material{}\n'.format(str(int(file_path.split('/')[-1].split('_')[0]))))
        for face in faces:
            line = "f"
            for i in face:
                line += " {}".format(i + f_add)
            line += "\n"
            f.write(line)
    return


def filter_mesh_obj_by_info(data_info, mesh_file:str, count_vs=100)->bool:
    mesh_filename = mesh_file.split('/')[-1].split('.')[0]

    if mesh_filename not in data_info.keys():
        return False

    if data_info[mesh_filename]['count_verts'] > count_vs:
        return False

    return True

def rotate_points(points_3d, rotation_matrix_2d):
    # Extract 2D points (ignoring z)
    points_2d = points_3d[:, :2]

    # Apply the rotation to the 2D points
    rotated_2d = np.dot(points_2d, rotation_matrix_2d.T)
    
    # Replace x and y coordinates in the 3D point cloud, keeping z the same
    rotated_3d = np.hstack((rotated_2d, points_3d[:, 2].reshape(-1, 1)))
    
    return rotated_3d

def quantize_verts(verts: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    """Convert floating point vertices to discrete values in [0, 2 ** n_bits - 1]

    Args:
        verts: np array of floating points vertices
        n_bits: number of quantization bits
    Returns:
        quantized_verts: np array representing quantized verts
    """
    range_quantize = 2 ** n_bits - 1
    quantized_verts = (verts - MIN_RANGE) * range_quantize / (MAX_RANGE - MIN_RANGE)
    return quantized_verts.to(torch.int32)

def dequantize_verts(verts: torch.Tensor, n_bits: int = 8, add_noise: bool = False) -> torch.Tensor:
    """Undoes quantization process and converts from [0, 2 ** n_bits - 1] to floats

    Args:
        verts: quantized representation of verts
        n_bits: number of quantization bits
        add_noise: adds random values from uniform distribution if set to true
    Returns:
        dequantized_verts: np array representing floating point verts
    """
    range_quantize = 2 ** n_bits - 1
    dequantized_verts = verts * (MAX_RANGE - MIN_RANGE) / range_quantize + MIN_RANGE
    if add_noise:
        dequantized_verts = torch.rand(size=dequantized_verts.shape) * (1 / range_quantize)
    return dequantized_verts

def torch_lexsort(a: torch.Tensor, dim=-1) -> torch.Tensor:
    """Pytorch implementation of np.lexsort (https://discuss.pytorch.org/t/numpy-lexsort-equivalent-in-pytorch/47850/3)

    Args:
        a: Tensor of shape (n, m)

    Returns:
        lex_sorted_tensor: Tensor of shape (n, m) after lexsort has been applied
    """

    assert dim == -1
    assert a.ndim == 2
    a_unq, inv = torch.unique(a.flip(0), dim=dim, sorted=True, return_inverse=True)
    return torch.argsort(inv)

def argmin(arr: List[float]) -> int:
    """Helper method to return argmin of a python list without numpy for code quality

    Args:
        arr: List of numbers

    Returns:
        argmin: Location of minimum element in list
    """
    return min(range(len(arr)), key=lambda x: arr[x])


def face_to_cycles(faces: List[int]) -> List[int]:
    """Find cycles in faces list

    Args:
        faces: List of vertex indices representing connectivity

    Returns:
        cycle_basis: All cycles in faces graph
    """
    g = nx.Graph()

    for v in range(len(faces) - 1):
        g.add_edge(faces[v], faces[v + 1])
    g.add_edge(faces[-1], faces[0])
    return list(nx.cycle_basis(g))

def quantize_process_mesh(
    vertices: torch.Tensor,
    faces: List[List[int]],
    tris: Optional[List[int]] = None,
    quantization_bits: int = 8,
) -> Tuple[torch.Tensor, List[List[int]], Optional[torch.Tensor]]:
    """Quantize vertices, remove resulting duplicates and reindex faces

    Args:
        vertices: torch tensor of shape (num_vertices, 3)
        faces: Unflattened faces
        tris: List of triangles
        quantization_bits: number of quantization bits

    Returns:
        vertices: processed vertices
        faces: processed faces
        triangles: list of triangles in 3D object
    """
    vertices = quantize_verts(vertices, quantization_bits)
    vertices, inv = torch.unique(vertices, dim=0, return_inverse=True)

    # Sort vertices by z then y then x
    sort_inds = torch_lexsort(vertices.T)
    vertices = vertices[sort_inds]

    # Re-index faces and tris to re-ordered vertices
    faces = [torch.argsort(sort_inds)[inv[f]] for f in faces]
    if tris is not None:
        tris = torch.Tensor([torch.argsort(sort_inds)[inv[t]] for t in tris])

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g. [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.

    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f.tolist())
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts
            if c_length > 2:
                d = argmin(c)
                # Cyclically permute faces so that the first index is the smallest
                sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])

    faces = sub_faces
    if tris is not None:
        tris = torch.Tensor([v for v in tris if len(set(v)) == len(v)])

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    faces = [torch.Tensor(f).to(torch.int64) for f in faces]
    if tris is not None:
        tris = tris.tolist()
        tris.sort(key=lambda f: tuple(sorted(f)))
        tris = torch.Tensor(tris)

    # After removing degenerate faces some vertices are now unreferenced
    # Remove these
    num_verts = vertices.shape[0]
    vert_connected = torch.eq(torch.arange(num_verts)[:, None], torch.hstack(faces)[None]).any(dim=-1)
    vertices = vertices[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = torch.arange(num_verts) - torch.cumsum((1 - vert_connected.to(torch.int32)), dim=-1)
    faces = [vert_indices[f].tolist() for f in faces]
    if tris is not None:
        tris = torch.Tensor([vert_indices[t].tolist() for t in tris])
    return vertices, faces, tris

def flatten_faces(faces: List[List[int]]) -> torch.Tensor:
    """Converts from list of faces to flat face array with stopping indices

    Args:
        faces: List of list of vertex indices

    Returns:
        flattened_faces: A 1D list of faces with stop tokens indicating when to move to the next face
    """
    if not faces:
        return torch.Tensor([0]).to(torch.int32)
    else:
        l = [f + [-1] for f in faces[:-1]]
        l += [faces[-1] + [-2]]
    return (torch.Tensor([item for sublist in l for item in sublist]) + 2).to(torch.int32)

def unflatten_faces(flat_faces: torch.Tensor) -> List[List[int]]:
    """Converts from flat face sequence to a list of separate faces

    Args:
        flat_faces: A 1D list of vertex indices with stopping tokens

    Returns:
        faces: A 2D list of face indices where each face is its own list
    """

    def group(seq):
        g = []
        for el in seq:
            if el == 0 or el == -1:
                yield g
                g = []
            else:
                g.append(el - 1)
        yield g

    outputs = list(group(flat_faces - 1))[:-1]
    return [o for o in outputs if len(o) > 2]

def random_shift(vertices: torch.Tensor, shift_factor: float = 0.25) -> torch.Tensor:
    """Randomly shift vertices in a cube according to some shift factor

    Args:
        vertices: tensor of shape (num_vertices, 3) representing current vertices
        shift_factor: float representing how much vertices should be shifted

    Returns:
        vertices: Shifted vertices
    """

    max_positive_shift = (255 - torch.max(vertices, dim=0)[0]).to(torch.float32)
    positive_condition_tensor = max_positive_shift > 1e-9
    max_positive_shift = torch.where(positive_condition_tensor, max_positive_shift, torch.Tensor([1e-9, 1e-9, 1e-9]))

    max_negative_shift = torch.min(vertices, dim=0)[0].to(torch.float32)
    negative_condition_tensor = max_negative_shift > 1e-9
    max_negative_shift = torch.where(negative_condition_tensor, max_negative_shift, torch.Tensor([1e-9, 1e-9, 1e-9]))
    normal_dist = TruncatedNormal(
        loc=torch.zeros((1, 3)),
        scale=shift_factor * 255,
        a=-max_negative_shift,
        b=max_positive_shift,
    )
    shift = normal_dist.sample().to(torch.int32)
    vertices += shift
    return vertices

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def extract_edges_from_faces(faces):
    edges = set()
    
    for face in faces:
        for i in range(len(face)):
            edge = tuple(sorted([face[i], face[(i + 1) % len(face)]]))
            edges.add(edge)
    
    return list(edges)

def is_almost_vertical(point1, point2, angle_threshold_degrees):
    """
    Check if the line segment defined by two endpoints is almost vertical,
    within a specified angle threshold from the z-axis.

    Args:
    - point1: A tuple (x1, y1, z1) representing the first endpoint of the line segment.
    - point2: A tuple (x2, y2, z2) representing the second endpoint of the line segment.
    - angle_threshold_degrees: The maximum angle (in degrees) allowed from the z-axis.

    Returns:
    - A boolean indicating whether the line segment is almost vertical.
    """
    
    # Create vectors from points and the z-axis
    line_vector = np.array(point2) - np.array(point1)
    if line_vector[-1] < 0:
        line_vector = np.array(point1) - np.array(point2)
    z_axis_vector = np.array([0, 0, 1])
    
    # Calculate the angle between the line_vector and the z-axis
    cosine_angle = np.dot(line_vector, z_axis_vector) / (np.linalg.norm(line_vector) * np.linalg.norm(z_axis_vector))
    
    # Ensure the cosine value is within valid range [-1, 1] to avoid NaN errors due to floating-point arithmetic
    cosine_angle = np.clip(cosine_angle, -1, 1)
    
    angle_radians = np.arccos(cosine_angle)
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    # Check if the angle is within the threshold
    return angle_degrees <= angle_threshold_degrees
