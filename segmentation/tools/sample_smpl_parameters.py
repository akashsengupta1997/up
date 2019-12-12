import deepdish as dd
import os
import pickle
import numpy as np
from sampling import uniform_sample_smpl_shape_deviation, uniform_rotation_sample_pose, \
    uniform_rodrigues_sample_pose_deviation


def load_mean_params(mean_params_path, flip_init_glob_pose):
    """Load mean shape and pose parameters."""
    mean_smpl = dd.io.load(mean_params_path)

    mean_pose = mean_smpl['pose']
    # Ignore the global rotation.
    mean_pose[:3] = 0.
    if flip_init_glob_pose:
        mean_pose[0] = np.pi
    mean_shape = mean_smpl['shape']

    return mean_shape, mean_pose


def uniform_rotation_sample_smpl(num,
                                 split,
                                 two_main_shape_range,
                                 shape_range,
                                 global_pose_max_angle,
                                 pose_max_angle,
                                 to_save_folder,
                                 num_glob_vert_rot=None,
                                 flip_init_glob_pose=True):
    mean_shape, _ = load_mean_params(
        '/Users/Akash_Sengupta/Documents/GitHub/pytorch_indirect_learning/models/smpl_requisites/neutral_smpl_mean_params.h5',
        flip_init_glob_pose=flip_init_glob_pose)
    shape_deviations = uniform_sample_smpl_shape_deviation(shape_range,
                                                           two_main_shape_range,
                                                           num)
    shapes = mean_shape + shape_deviations
    for i in range(num):
        poses = uniform_rotation_sample_pose(global_pose_max_angle, pose_max_angle,
                                             flip_init_glob_pose)
        print('shapes, poses', shapes.shape, poses.shape)

        with open(os.path.join("/Users/Akash_Sengupta/Documents/Datasets/SMPL_synthetic",
                               to_save_folder,
                               split,
                               "{}_body.pkl".format(str(i).zfill(6))),
                  'wb') as f:
            pickle.dump({'shape': shapes[i],
                         'pose': poses},
                        f,
                        protocol=2)


def uniform_sample_smpl(num,
                        split,
                        two_main_shape_range,
                        shape_range,
                        global_pose_range,
                        pose_range,
                        to_save_folder,
                        num_glob_vert_rot=None,
                        flip_init_glob_pose=True):

    mean_shape, mean_pose = load_mean_params(
        '/Users/Akash_Sengupta/Documents/GitHub/pytorch_indirect_learning/models/smpl_requisites/neutral_smpl_mean_params.h5',
        flip_init_glob_pose=flip_init_glob_pose)
    shape_deviations = uniform_sample_smpl_shape_deviation(shape_range,
                                                           two_main_shape_range,
                                                           num)

    if num_glob_vert_rot is None:
        pose_deviations = uniform_rodrigues_sample_pose_deviation(pose_range,
                                                                  global_pose_range,
                                                                  num)
    else:
        num_per_rot = int(num / num_glob_vert_rot)
        pose_deviations_no_rot = uniform_sample_smpl_shape_deviation(pose_range,
                                                                     (0, 0),
                                                                     num_per_rot)
        rot_pose_deviations_list = [pose_deviations_no_rot]
        for i in range(1, num_glob_vert_rot):
            pose_deviations_rot = np.copy(pose_deviations_no_rot)
            pose_deviations_rot[:, 1] = i * (2 * np.pi) / num_glob_vert_rot
            rot_pose_deviations_list.append(pose_deviations_rot)
        pose_deviations = np.concatenate(rot_pose_deviations_list, axis=0)

    shapes = mean_shape + shape_deviations
    poses = mean_pose + pose_deviations

    print('shapes', shapes.shape, 'poses', poses.shape)
    for i in range(shapes.shape[0]):
        with open(os.path.join("/Users/Akash_Sengupta/Documents/Datasets/SMPL_synthetic",
                               to_save_folder,
                               split,
                               "{}_body.pkl".format(str(i).zfill(6))),
                  'wb') as f:
            pickle.dump({'shape': shapes[i],
                         'pose': poses[i]},
                        f,
                        protocol=2)


# uniform_sample_smpl(16, 'train', (-4, 4), (-1.5, 1.5), (-1.0, 1.0), (-0.4, 0.4),
#                     'hard_shape_hard_pose', flip_init_glob_pose=True)
# uniform_sample_smpl(4, 'val', (-4, 4), (-1.5, 1.5), (-1.0, 1.0), (-0.4, 0.4),
#                     'hard_shape_hard_pose', flip_init_glob_pose=True)

# uniform_sample_smpl(16, 'train', (-4, 4), (-4, 4), (-0.2, 0.2), (-0.2, 0.2),
#                     'hard2_shape_easy_pose', flip_init_glob_pose=True)
# uniform_sample_smpl(4, 'val', (-1, 1), (-1, 1), (-0.2, 0.2), (-0.2, 0.2),
#                     'hard2_shape_easy_pose', flip_init_glob_pose=True)

# uniform_sample_smpl(16, 'train', (-4, 4), (-1.5, 1.5), (-2.0, 2.0), (-0.7, 0.7),
#                     'hard_shape_hard2_pose', flip_init_glob_pose=True)
# uniform_sample_smpl(4, 'val', (-4, 4), (-1.5, 1.5), (-2.0, 2.0), (-0.7, 0.7),
#                     'hard_shape_hard2_pose', flip_init_glob_pose=True)

# uniform_sample_smpl(16, 'train', (-4, 4), (-1.5, 1.5), (-2.0, 2.0), (-0.4, 0.4),
#                     'hard_shape_hard_glob_pose', flip_init_glob_pose=True)
# uniform_sample_smpl(4, 'val', (-4, 4), (-1.5, 1.5), (-2.0, 2.0), (-0.4, 0.4),
#                     'hard_shape_hard_glob_pose', flip_init_glob_pose=True)

uniform_rotation_sample_smpl(16, 'train', (-1, 1), (-1, 1), np.pi/3.0, np.pi/3.0,
                             'uniform_rot_sample_test', flip_init_glob_pose=True)