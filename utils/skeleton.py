import numpy as np
import open3d
import cv2
import trimesh
import torch


def get_sphere(position, radius=1.0, color=(0.1, 0.1, 0.7)):
    mesh_sphere: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.paint_uniform_color(color)

    # translate to position
    mesh_sphere = mesh_sphere.translate(position, relative=False)
    return mesh_sphere

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_cylinder(start_point, end_point, radius=0.3, color=(0.1, 0.9, 0.1)):
    center = (start_point + end_point) / 2
    height = np.linalg.norm(start_point - end_point)
    mesh_cylinder: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh_cylinder.paint_uniform_color(color)

    # translate and rotate to position
    # rotate vector
    rot_vec = end_point - start_point
    rot_vec = rot_vec / np.linalg.norm(rot_vec)
    rot_0 = np.array([0, 0, 1])
    rot_mat = rotation_matrix_from_vectors(rot_0, rot_vec)
    # if open3d.__version__ >= '0.9.0.0':
    #     rotation_param = rot_mat
    # else:
    #     rotation_param = Rotation.from_matrix(rot_mat).as_euler('xyz')
    rotation_param = rot_mat
    mesh_cylinder = mesh_cylinder.rotate(rotation_param)
    mesh_cylinder = mesh_cylinder.translate(center, relative=False)
    return mesh_cylinder


class Skeleton:
    heatmap_sequence = ["Head", # 0
                        "Neck", # 1
                        "Right_shoulder", # 2 
                        "Right_elbow", # 3
                        "Right_wrist", # 4
                        "Left_shoulder", # 5
                        "Left_elbow", # 6
                        "Left_wrist", # 7
                        "Right_hip", # 8
                        "Right_knee", # 9
                        "Right_ankle", # 10
                        "Right_foot", # 11
                        "Left_hip", # 12 
                        "Left_knee", # 13
                        "Left_ankle", #14
                        "Left_foot"] # 15
    lines = [(0, 1),  
             (1, 2), (2, 3), (3, 4), 
             (1, 5), (5, 6), (6, 7), 
             (2, 8), (8, 9), (9, 10), (10, 11),
             (5, 12), (12, 13), (13, 14), (14, 15),
             (8, 12)]

    c1, c2, c3, c4 = (236, 252, 5), (5, 244, 255), (89, 235, 52), (0, 106, 255)

    colors = [c1, 
              c2, c2, c2, 
              c3, c3, c3, 
              c2, c2, c2, c2,
              c3, c3, c3, c3,
              c4]
    
    kinematic_parents = [0, 1, 1, 2, 3, 1, 5, 6, 2, 8, 9, 10, 5, 12, 13, 14]
    
    def __init__(self, color=(0.1, 0.9, 0.1)):
        
        self.skeleton = None
        self.skeleton_mesh = None
        self.color = color
                
    def joints_2_mesh(self, joints_3d):
        if isinstance(joints_3d, list):
            joints_3d = np.asarray(joints_3d)
            
        elif isinstance(joints_3d, torch.Tensor):
            joints_3d = joints_3d.cpu().numpy()
        
        self.skeleton = joints_3d
        self.skeleton_to_mesh()
        skeleton_mesh = self.skeleton_mesh
        self.skeleton_mesh = None
        self.skeleton = None
        return skeleton_mesh
    
    def joint_list_2_mesh_list(self, joints_3d_list):
        mesh_list = []
        for joints_3d in joints_3d_list:
            mesh_list.append(self.joints_2_mesh(joints_3d))
        return mesh_list
    
    def get_skeleton_mesh(self):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            return self.skeleton_mesh
    
    def save_skeleton(self, out_path):
        if self.skeleton_mesh is None:
            raise Exception("Skeleton is not prepared.")
        else:
            open3d.io.write_triangle_mesh(out_path, mesh=self.skeleton_mesh)

    @classmethod
    def save_mesh(cls, mesh, out_path):
        open3d.io.write_triangle_mesh(out_path, mesh=mesh)

    def render(self):
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        open3d.visualization.draw_geometries([self.skeleton_mesh, mesh_frame])
    
    def skeleton_to_mesh(self):
        final_mesh = open3d.geometry.TriangleMesh()
        for i in range(len(self.skeleton)):
            keypoint_mesh = get_sphere(position=self.skeleton[i], radius=0.02)
            final_mesh = final_mesh + keypoint_mesh
        
        for line in self.lines:
            line_start_i = line[0]
            line_end_i = line[1]
            
            start_point = self.skeleton[line_start_i]
            end_point = self.skeleton[line_end_i]
            
            line_mesh = get_cylinder(start_point, end_point, radius=0.005, color=self.color)
            final_mesh += line_mesh
        self.skeleton_mesh = final_mesh
        return final_mesh
    
    @classmethod
    def draw_2d_skeleton(cls, img, j2d, color=None, lines=True, line_width=2):
        img = img.copy()
        
        if isinstance(j2d, list):
            j2d = np.asarray(j2d)
        
        if isinstance(j2d, torch.Tensor):
            j2d = j2d.cpu().numpy() 

        j2d = j2d.astype(dtype=np.int32)

        for x, y in j2d:
            cv2.circle(img, (x, y), 3, (255, 255, 255), -1)

        if not lines:
            return img    
        
        for line, limb_color in zip(cls.lines, cls.colors):
            line_start_i = line[0]
            line_end_i = line[1]
            
            start_point = j2d[line_start_i]
            end_point = j2d[line_end_i]

            if color is None:
                c = limb_color
            else:
                c = color
            
            cv2.line(img, (start_point[0], start_point[1]), (end_point[0], end_point[1]), c, line_width)

        return img

    @classmethod
    def to_trimesh(cls, mesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
    
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)
        else:
            vertex_colors = None

        if mesh.has_vertex_normals():
            vertex_normals = np.asarray(mesh.vertex_normals)
        else:
            vertex_normals = None

        return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors, vertex_normals=vertex_normals)


    def joints_2_trimesh(self, joints_3d, coord_frame=False, ground_plane=False):
        mesh = self.joints_2_mesh(joints_3d)
        mesh = self.to_trimesh(mesh)
        
        if coord_frame:
            coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
            coord = self.to_trimesh(coord)
            
            mesh = trimesh.util.concatenate([mesh, coord])

        return mesh
    
    def transform(self, transorm_matrix_4x4):
        n_joints = self.skeleton.shape[0]

        pose_gt_homo = np.ones((n_joints, 4))
        pose_gt_homo[:, :3] = self.skeleton
        pose_gt_homo = transorm_matrix_4x4.dot(pose_gt_homo.T).T
        transformed_pose_gt = pose_gt_homo[:, :3].astype(np.float32)

        self.skeleton = transformed_pose_gt

        return transformed_pose_gt
    
    @classmethod
    def transform_joints(cls, transorm_matrix_4x4, j3d):
        n_joints = j3d.shape[0]

        pose_gt_homo = np.ones((n_joints, 4))
        pose_gt_homo[:, :3] = j3d
        pose_gt_homo = transorm_matrix_4x4.dot(pose_gt_homo.T).T
        transformed_pose_gt = pose_gt_homo[:, :3].astype(np.float32)

        return transformed_pose_gt