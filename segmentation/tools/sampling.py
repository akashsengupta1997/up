import numpy as np
import cv2


def uniform_random_rot_matrix():
    """
    Uniform sampling of a random 3D rotation matrix using QR decomposition.
    Source: https://arxiv.org/pdf/math-ph/0609050.pdf
    """
    Z = np.random.randn(3, 3)
    Q, R = np.linalg.qr(Z)
    d = np.diagonal(R)
    ph = d/np.absolute(d)
    # matmul with diagonal matrix L equivalent to element-wise mul with broad-casted vector l
    Q = np.multiply(Q, ph, Q)
    return Q


def uniform_random_unit_vector():
    """
    Uniform sampling of 3D unit-vector, i.e. point on surface of unit sphere.
    """
    e = np.random.randn(3)
    e = e/np.linalg.norm(e)

    return e


def uniform_sample_smpl_shape_deviation(range, fat_and_height_range, num):
    """Uniform sampling of shape parameter deviations from mean."""
    fat_and_height_params = np.random.uniform(low=fat_and_height_range[0],
                                              high=fat_and_height_range[1],
                                              size=(num, 2))
    smpl_shapes = np.random.uniform(low=range[0], high=range[1], size=(num, 8))
    shapes = np.concatenate([fat_and_height_params, smpl_shapes], axis=1)
    return shapes


def uniform_rodrigues_sample_pose_deviation(range, global_rot_range, num):
    """Uniform sampling of pose parameter deviations from mean (i.e. uniform in Rodrigues)
    This is not actually uniform in the 3D rotation space."""
    glob_rot = np.random.uniform(low=global_rot_range[0],
                                 high=global_rot_range[1],
                                 size=(num, 3))
    smpl_poses = np.random.uniform(low=range[0],
                                   high=range[1],
                                   size=(num, 69))
    poses = np.concatenate([glob_rot, smpl_poses], axis=1)
    return poses


def uniform_axis_angle_sample_pose_deviation(orig_rodrigues_pose, axis_deviation_scale,
                                             angle_deviation_range, num):
    poses = []
    for i in range(num):
        pose = np.zeros(69)
        for joint in range(23):
            e_deviation = axis_deviation_scale * uniform_random_unit_vector()
            angle_deviation = np.random.uniform(low=angle_deviation_range[0],
                                                high=angle_deviation_range[1])

            orig_r = orig_rodrigues_pose[joint*3:(joint+1)*3]
            orig_angle = np.linalg.norm(orig_r)
            orig_e = orig_r/orig_angle

            e = orig_e + e_deviation
            angle = orig_angle + angle_deviation
            r = e * angle
            pose[joint*3:(joint+1)*3] = r

        poses.append(pose)

    return poses


def uniform_rotation_sample_global_pose(global_pose_max_angle, flip_init_global_pose, num):
    r_globals = []
    for i in range(num):
        accept_rotation = False
        while not accept_rotation:
            R = uniform_random_rot_matrix()
            r = cv2.Rodrigues(R)[0]
            angle = np.linalg.norm(r + 1e-8, ord=2)
            if angle < global_pose_max_angle:
                accept_rotation = True
                if flip_init_global_pose:
                    R_flip = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
                    R_final = np.matmul(R, R_flip)
                    r_final = cv2.Rodrigues(R_final)[0]
                    r_globals.append(np.squeeze(r_final))

    return r_globals


def uniform_rotation_sample_pose(global_pose_max_angle, pose_max_angle, flip_init_global_pose):
    """Uniform sampling of rotation matrices for pose - uniform in 3D rotation space
    (i.e. SO(3))"""
    rodrigues_vectors = []
    for joint in range(24):
        accept_rotation = False
        if joint == 0:
            r_global = uniform_rotation_sample_global_pose(global_pose_max_angle,
                                                           flip_init_global_pose,
                                                           1)[0]
            rodrigues_vectors.append(np.squeeze(r_global))
        else:
            while not accept_rotation:
                R = uniform_random_rot_matrix()
                r = cv2.Rodrigues(R)[0]
                angle = np.linalg.norm(r + 1e-8, ord=2)
                if angle < pose_max_angle:
                    accept_rotation = True
                    rodrigues_vectors.append(np.squeeze(r))

    pose_params = np.concatenate(rodrigues_vectors)
    return pose_params
