import numpy as np


def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def compute_similarity_transform_batch(S1_batch, S2_batch):
    batch_size = S1_batch.shape[0]
    assert S1_batch.shape == S2_batch.shape

    transformed_batch = np.empty_like(S1_batch)

    for i in range(batch_size):
        S1 = S1_batch[i]
        S2 = S2_batch[i]

        transformed_batch[i] = compute_similarity_transform(S1, S2)

    return transformed_batch


def align_by_pelvis(joints, return_pelvis=False):
    left_id = 12
    right_id = 8

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    pelvis = pelvis[None, ...]
    
    rr_joints = joints - pelvis
    
    if return_pelvis:
        return rr_joints, pelvis

    return rr_joints


def align_by_pelvis_batch(joints, return_pelvis=False):
    left_id = 12
    right_id = 8

    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.0 # middle of left and right hip
    pelvis = pelvis[:, None, ...]
    
    rr_joints = joints - pelvis

    if return_pelvis:
        return rr_joints, pelvis

    return rr_joints


# def compute_errors(gt3ds, preds):
#     """
#     Gets MPJPE after pelvis alignment + MPJPE after Procrustes.
#     Evaluates on the 14 common joints.
#     Inputs:
#       - gt3ds: N x 14 x 3
#       - preds: N x 14 x 3
#     """
#     errors, errors_pa = [], []
#     for i, (gt3d, pred) in enumerate(zip(gt3ds, preds)):
#         gt3d = gt3d.reshape(-1, 3)
#         # Root align.
#         gt3d = align_by_pelvis(gt3d)
#         pred3d = align_by_pelvis(pred)

#         joint_error = np.sqrt(np.sum((gt3d - pred3d)**2, axis=1))
#         errors.append(np.mean(joint_error))

#         # Get PA error.
#         pred3d_sym = compute_similarity_transform(pred3d, gt3d)
#         pa_error = np.sqrt(np.sum((gt3d - pred3d_sym)**2, axis=1))
#         errors_pa.append(np.mean(pa_error))

#     return errors, errors_pa



def compute_3d_errors_batch(gt3ds, preds, valid_j3d):
    valid_j3d = np.sum(valid_j3d, -1).mean(-1)
    valid_j3d = valid_j3d > 0
    
    gt3ds = gt3ds[valid_j3d, :, :]
    preds = preds[valid_j3d, :, :]
    
    cnt = gt3ds.shape[0]
    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))
    errors = np.sum(joint_error, 0) / cnt

    pred3d_sym = compute_similarity_transform_batch(preds, gt3ds)
    joint_error = np.sqrt(np.sum((gt3ds - pred3d_sym)**2, axis=-1))
    errors_pa = np.sum(joint_error, 0) / cnt

    return errors, errors_pa

def compute_3d_errors_joints(gt3ds, preds, valid_j3d):
    valid_j3d = np.sum(valid_j3d, -1).mean(-1)
    valid_j3d = valid_j3d > 0
    
    gt3ds = gt3ds[valid_j3d, :, :]
    preds = preds[valid_j3d, :, :]
    
    cnt = gt3ds.shape[1]
    joint_error = np.sqrt(np.sum((gt3ds - preds)**2, axis=-1))
    errors = np.sum(joint_error, 1) / cnt

    pred3d_sym = compute_similarity_transform_batch(preds, gt3ds)
    joint_error = np.sqrt(np.sum((gt3ds - pred3d_sym)**2, axis=-1))
    errors_pa = np.sum(joint_error, 1) / cnt

    return errors, errors_pa