import math
import torch
import traceback
import numpy as np
import torchvision
import cv2
import pyrender
import trimesh

from settings import config
from utils.heatmap import get_max_preds
from utils.skeleton import Skeleton


def save_batch_images(batch_image, writer, tag, global_step, nrow=8, padding=2, n_images=4, shuffle=False):
    '''
    batch_image: [batch_size, channel, height, width]
    '''
    if batch_image.shape[1] == 2:
        N, C, H, W = batch_image.shape
        batch_image = torch.cat([batch_image, torch.zeros(N, 1, H, W).to(batch_image.device)], dim=1)
    elif batch_image.shape[1] == 1:
        N, C, H, W = batch_image.shape
        batch_image = torch.cat([batch_image, batch_image, batch_image], dim=1)

    if shuffle:
        idx = np.random.choice(batch_image.shape[0], n_images, replace=False)
        batch_image = batch_image[idx]
    else:
        batch_image = batch_image[:n_images]

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    
    writer.add_image(tag, ndarr.transpose(2, 0, 1), global_step)


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 writer, tag, global_step, nrow=8, padding=2, n_images=4):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    if batch_image.shape[1] == 2:
        N, C, H, W = batch_image.shape
        batch_image = torch.cat([batch_image, torch.zeros(N, 1, H, W).to(batch_image.device)], dim=1)

    batch_image = batch_image[:n_images]
    batch_joints = batch_joints[:n_images]
    batch_joints_vis = batch_joints_vis[:n_images]

    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                jx = x * width + padding + joint[0]
                jy = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(jx), int(jy)), 2, [255, 0, 0], 2)
            k = k + 1

    # cv2.imwrite(file_name, ndarr)
    writer.add_image(tag, ndarr.transpose(2, 0, 1), global_step)

def save_batch_location_maps(gt_lms, pred_lms, tag, writer, global_step, n_images=4):
    pred_lms = pred_lms[:n_images].detach()
    gt_lms = gt_lms[:n_images]

    batch_size = gt_lms.size(0)
    num_joints = gt_lms.size(1)
    coor_dim = gt_lms.size(2)
    lm_height = gt_lms.size(3)
    lm_width = gt_lms.size(4)

    tag = tag + '_lm'

    def _get_grid_image(lms):	 
        grid_image = np.zeros((batch_size*lm_height, num_joints * lm_width, 3), dtype=np.uint8)

        for i in range(batch_size):
            heatmaps = lms[i].mul(255).clamp(0, 255).cpu().permute(0, 2, 3, 1)
            
            heatmaps = heatmaps.numpy().astype(np.uint8)
            
            height_begin = lm_height * i
            height_end = lm_height * (i + 1)
            for j in range(num_joints):
                heatmap = heatmaps[j, :, :]
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
                width_begin = lm_width * j
                width_end = lm_width * (j+1)
                grid_image[height_begin:height_end, width_begin:width_end, :] = \
                    colored_heatmap
            
        return grid_image
    
    writer.add_image(tag + '_pred', _get_grid_image(pred_lms), global_step, dataformats='HWC')
    writer.add_image(tag + '_gt', _get_grid_image(gt_lms), global_step, dataformats='HWC')

def save_batch_heatmaps(batch_image, batch_heatmaps, writer, tag, global_step, 
                        normalize=True, n_images=4):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if batch_image.shape[1] == 2:
        N, C, H, W = batch_image.shape
        batch_image = torch.cat([batch_image, torch.zeros(N, 1, H, W).to(batch_image.device)], dim=1)

    batch_image = batch_image[:n_images]
    batch_heatmaps = batch_heatmaps[:n_images]

    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    # cv2.imwrite(file_name, grid_image)
    writer.add_image(tag, grid_image.transpose(2, 0, 1), global_step)

def save_debug_images(config, input, meta, target, joints_pred, output, prefix, writer, global_step):
    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['j2d'], meta['vis_j2d'],
            writer, f'{prefix}_gt', 
            global_step
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['vis_j2d'],
            writer, f'{prefix}_pred',
            global_step
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, writer, f'{prefix}_hm_gt',
            global_step
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, writer, f'{prefix}_hm_pred',
            global_step
        )


renderer = pyrender.OffscreenRenderer(1440, 1080)

scene = pyrender.Scene(ambient_light=(0.9, 0.9, 0.9))
camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0)


device_centric_camera_pose  = [
    [-0.50501284, -0.14524132, 0.85080373, 1.02667734],
    [ 0.85942626, -0.17561686, 0.48015125, 0.79955088],
    [ 0.07967768, 0.97368562, 0.21351297, -0.39187423],
    [ 0., 0., 0., 1.]]

camera_pose = np.array(device_centric_camera_pose)
    
camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
scene.add_node(camera_node)


def generate_skeleton_image(gt_j3d, pred_j3d):
    gt_skeleton = Skeleton((0.1, 0.9, 0.1)) # Green - RGB
    pred_skeleton = Skeleton((0.9, 0.1, 0.1)) # Red - RGB

    gt_node = None
    try:
        gt_mesh = gt_skeleton.joints_2_trimesh(gt_j3d)    
        gt_mesh = pyrender.Mesh.from_trimesh(gt_mesh)
        gt_node = pyrender.Node(mesh=gt_mesh)
        scene.add_node(gt_node)
    except:
        traceback.print_exc()   
    
    pred_node = None
    try:
        pred_mesh = pred_skeleton.joints_2_trimesh(pred_j3d)
        pred_mesh = pyrender.Mesh.from_trimesh(pred_mesh)
        pred_node = pyrender.Node(mesh=pred_mesh)
        scene.add_node(pred_node)
    except:
        traceback.print_exc()

    color, depth = renderer.render(scene)

    if gt_node is not None:
        scene.remove_node(gt_node)
    
    if pred_node is not None:
        scene.remove_node(pred_node)

    return color


def save_debug_3d_joints(config, inp, meta, gt_j3d, pred_j3d, prefix, writer, global_step):    
    idx = np.random.randint(0, inp.shape[0])
    
    if isinstance(gt_j3d, torch.Tensor):
        gt_j3d = gt_j3d.detach().cpu().numpy()
        
    if isinstance(pred_j3d, torch.Tensor):
        pred_j3d = pred_j3d.detach().cpu().numpy()

    gt_j3d = gt_j3d[idx] / 1000 # mm -> m
    pred_j3d = pred_j3d[idx] / 1000 # mm -> m

    color = generate_skeleton_image(gt_j3d, pred_j3d)
    writer.add_image(prefix + '_j3d', color, global_step, dataformats='HWC')

    return color


def save_debug_segmenation(config, inp, meta, gt_seg, pred_seg, prefix, writer, global_step):
    save_batch_images(gt_seg, writer, f'{prefix}_seg_gt', global_step)
    save_batch_images(pred_seg, writer, f'{prefix}_seg_pred', global_step)


def save_debug_eros(config, inp, meta, eros, prefix, writer, global_step):
    save_batch_images(eros, writer, f'{prefix}_eros', global_step, shuffle=True)
 
 
def plot_heatmaps(image, heatmaps):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.cpu().numpy()
    
    H, W, C = image.shape
    
    hm_comp = np.zeros((H, W, C), dtype=np.float32)
    N = heatmaps.shape[0]
    for i in range(N):
        hm = heatmaps[i] * 255
        hm = np.clip(hm, 0, 255).astype(np.uint8)
        
        hm = cv2.resize(hm, (W, H))
        hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)        
        hm_comp += hm

    hm_comp = np.clip(hm_comp, 0, 255).astype(np.uint8)    
    image = cv2.addWeighted(image, 0.5, hm_comp, 0.5, 0)

    return image


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.colors = o3d.utility.Vector3dVector(colors)


def create_ground_plane(ground_plane=None, length=25.0, color0=[0.8, 0.9, 0.9], color1=[0.6, 0.7, 0.7], tile_width=0.5, xyz_orig=None, alpha=1.0):
        '''
        If ground_plane is none just places at origin with +z up.
        If ground_plane is given (a, b, c, d) where a,b,c is the normal, then this is rendered. To more accurately place the floor
        provid an xyz_orig = [x,y,z] that we expect to be near the point of focus.
        '''
        color0 = np.array(color0 + [alpha])
        color1 = np.array(color1 + [alpha])
        # make checkerboard
        radius = length / 2.0
        num_rows = num_cols = int(length / tile_width)
        vertices = []
        faces = []
        face_colors = []
        for i in range(num_rows):
            for j in range(num_cols):
                start_loc = [-radius + j*tile_width, radius - i*tile_width]
                cur_verts = np.array([[start_loc[0], start_loc[1], 0.0],
                                      [start_loc[0], start_loc[1]-tile_width, 0.0],
                                      [start_loc[0]+tile_width, start_loc[1]-tile_width, 0.0],
                                      [start_loc[0]+tile_width, start_loc[1], 0.0]])
                cur_faces = np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)
                cur_faces += 4 * (i*num_cols + j) # the number of previously added verts
                use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
                cur_color = color0 if use_color0 else color1
                cur_face_colors = np.array([cur_color, cur_color])

                vertices.append(cur_verts)
                faces.append(cur_faces)
                face_colors.append(cur_face_colors)

        vertices = np.concatenate(vertices, axis=0)
        faces = np.concatenate(faces, axis=0)
        face_colors = np.concatenate(face_colors, axis=0)

        if ground_plane is not None:            
            # compute transform between identity floor and passed in floor
            a, b, c, d = ground_plane
            # rotation
            old_normal = np.array([0.0, 0.0, 1.0])
            new_normal = np.array([a, b, c])
            new_normal = new_normal / np.linalg.norm(new_normal)
            v = np.cross(old_normal, new_normal)
            ang_sin = np.linalg.norm(v)
            ang_cos = np.dot(old_normal, new_normal)
            skew_v = np.array([[0.0, -v[2], v[1]],
                            [v[2], 0.0, -v[0]],
                            [-v[1], v[0], 0.0]])
            R = np.eye(3) +  skew_v + np.matmul(skew_v, skew_v)*((1.0 - ang_cos) / (ang_sin**2))
            # translation
            # project point of focus onto plane
            if xyz_orig is None:
                xyz_orig = np.array([0.0, 0.0, 0.0])
            # project origin onto plane
            plane_normal = np.array([a, b, c])
            plane_off = d
            direction = -plane_normal
            s = (plane_off - np.dot(plane_normal, xyz_orig)) / np.dot(plane_normal, direction)
            itsct_pt = xyz_orig + s*direction
            t = itsct_pt

            # transform floor
            vertices = np.dot(R, vertices.T).T + t.reshape((1, 3))

        if xyz_orig is None:
            xyz_orig = np.array([0.0, 0.0, 0.0])
            
        vertices += xyz_orig.reshape((1, 3))

        ground_mesh = trimesh.creation.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors, process=False)
        
        return ground_mesh


def main():
    import open3d as o3d
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    mat = rendering.Material()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, len(points.points)):
        widget3d.add_3d_label(points.points[idx], "{}".format(idx))
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()
