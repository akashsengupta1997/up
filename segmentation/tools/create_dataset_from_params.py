#!/usr/bin/env python2
"""
Render given SMPL parameters and create a synthetic dataset with matching part-segmentation and
joints.
"""
import os
import os.path as path
import sys
import logging
import pickle
from copy import copy as _copy
from collections import OrderedDict

import numpy as np
import scipy
import click
import cv2
import opendr.camera as _odr_c

from clustertools.log import LOGFORMAT
from clustertools.visualization import apply_colormap
from up_tools.model import (robust_person_size, six_region_groups,
                            regions_to_classes, get_crop, landmark_mesh_91)
import up_tools.visualization as vs

from sampling import uniform_rotation_sample_global_pose, uniform_sample_smpl_shape_deviation, \
    uniform_axis_angle_sample_pose_deviation

from up_tools.render_segmented_views import render, render_body_impl, MODEL_NEUTRAL, _TEMPLATE_MESH
sys.path.insert(0, path.join(path.dirname(__file__), '..', '..'))
from config import SEG_DATA_FP, UP3D_FP


LOGGER = logging.getLogger(__name__)
DSET_ROOT_FP = SEG_DATA_FP

if not path.exists(DSET_ROOT_FP):
    os.mkdir(DSET_ROOT_FP)


def get_landmark_positions(betas, pose, trans, resolution, landmarks, rt, t, f):
    """Get landmark positions for a given image."""
    # Pose the model.
    model = MODEL_NEUTRAL
    model.betas[:len(betas)] = betas
    model.pose[:] = pose
    model.trans[:] = trans
    mesh = _copy(_TEMPLATE_MESH)
    mesh.v = model.r
    mesh_points = mesh.v[tuple(landmarks.values()),]
    J_onbetas = model.J_regressor.dot(model.r)
    skeleton_points = J_onbetas[(8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20),]
    # Do the projection.
    camera = _odr_c.ProjectPoints(
        rt=rt,
        t=t,
        f=(f, f),
        c=np.array(resolution) / 2.,
        k=np.zeros(5))
    camera.v = np.vstack((skeleton_points, mesh_points))
    return camera.r.T.copy()


def check_landmark_visibility(landmarks, resolution_wh):
    vis = np.ones(landmarks.shape[-1])
    vis[landmarks[0] > resolution_wh] = 0
    vis[landmarks[1] > resolution_wh] = 0
    vis[landmarks[0] < 0] = 0
    vis[landmarks[1] < 0] = 0

    return vis


def add_dataset(dset_fp, landmarks, partspec, resolution_wh=256, num_zfill_in_name=5,
                start=0, num=0,
                num_augment_per_sample=0,
                global_pose_augment=False, global_pose_max_angle=2*np.pi, flip_init_glob_pose=True,
                shape_augment=False, shape_range=[-2, 2], fat_height_range=[-4, 4],
                pose_augment=False, pose_axis_deviation_scale=0.1, pose_angle_deviation_range=[-np.pi/6, np.pi/6]):
    """Add a dataset to the collection."""
    ids_list = [str(f[:num_zfill_in_name]) for f in sorted(os.listdir(dset_fp))
                if f.endswith("_body.pkl") and '-' not in f]
    to_render_ids = ids_list[start:start+num]
    LOGGER.info("Writing dataset. Shape augment: {}, "
                "Glob pose augment: {}, "
                "Pose augment: {}".format(str(shape_augment),
                                          str(global_pose_augment),
                                          str(pose_augment)))

    for im_idx in to_render_ids:
        print('Index', im_idx)
        smpl_path = path.join(dset_fp, '{}_body.pkl'.format(im_idx))

        with open(smpl_path, 'rb') as f:
            smpl_data = pickle.load(f)
        pose = smpl_data['pose']
        if 'betas' in smpl_data.keys():
            betas = smpl_data['betas']
        elif 'shape' in smpl_data.keys():
            betas = smpl_data['shape']

        rt = np.array([0.0, 0.0, 0.0])
        f = np.array(5000.0)
        trans = np.array([0.0, 0.0, 0.0])
        t = np.array([0.0, 0.0, 40.0])

        # ------- First render original, un-augmented data (if doing UP3D augmentation) -------
        camera = {'rt': rt, 'f': f, 'trans': trans, 't': t, 'betas': betas, 'pose': pose}
        resolution = (resolution_wh, resolution_wh)
        factor = 1.0

        renderings = render(MODEL_NEUTRAL,
                            (np.asarray(resolution) * 1. / factor).astype('int'),
                            camera,
                            1,
                            False,
                            use_light=False)
        renderings = [scipy.misc.imresize(renderim,
                                          (resolution[1],
                                           resolution[0]),
                                          interp='nearest')
                      for renderim in renderings]
        rendering = renderings[0]  # Only rendering one rotated view - single element in list

        landmark_pos = get_landmark_positions(betas, pose, trans, resolution, landmarks, rt, t, f)
        vis = check_landmark_visibility(landmark_pos, resolution_wh)

        class_groups = six_region_groups if partspec == '6' else None
        annotation = regions_to_classes(rendering, class_groups, warn_id=im_idx)
        if partspec == '1':
            annotation = (annotation > 0).astype('uint8')
        # assert np.max(annotation) <= int(partspec), (
        #     "Wrong annotation value (%s): %s!" % (im_idx, str(np.unique(annotation))))
        # if int(im_idx) == 0:
        #     assert np.max(annotation) == int(partspec), ("Probably an error in the number of parts!")
        pose_vis_im = vs.visualize_pose(cv2.cvtColor(annotation*8, cv2.COLOR_GRAY2RGB),
                                        landmark_pos,
                                        scale=1.)
        scipy.misc.imsave(path.join(dset_fp, '{}_ann.png'.format(im_idx)),
                          annotation)
        scipy.misc.imsave(path.join(dset_fp, '{}_seg_ann_vis.png'.format(im_idx)),
                          apply_colormap(annotation, vmax=int(partspec)))
        scipy.misc.imsave(path.join(dset_fp, '{}_pose_ann_vis.png'.format(im_idx)),
                          pose_vis_im)

        landmark_pos_with_vis = np.concatenate([landmark_pos,  vis[None, :]], axis=0)
        np.save(str(path.join(dset_fp, '{}_joints.npy'.format(im_idx))),
                landmark_pos_with_vis,
                allow_pickle=False)

        # --------------- Render augmented data (if doing UP3D augmentation) ---------------
        # UP3D Augmentation by random sampling
        if num_augment_per_sample > 0:
            if global_pose_augment:
                assert 'global_pose' in dset_fp, "Dataset path is probably wrong!"
                # Random sampling of global rotations
                new_r_globals = uniform_rotation_sample_global_pose(
                    global_pose_max_angle=global_pose_max_angle,
                    flip_init_global_pose=flip_init_glob_pose,
                    num=num_augment_per_sample-1)
                # First global rotation augmentation is set to be a backwards facing one
                # since this is the second most common global pose modality after front-facing
                R_global_backface = np.matmul(cv2.Rodrigues(np.array([0, np.pi, 0]))[0],
                                              cv2.Rodrigues(np.array([np.pi, 0, 0]))[0])
                r_global_backface = np.squeeze(cv2.Rodrigues(R_global_backface)[0])
                r_global_backface += 0.1*np.random.randn(3)  # add some random noise
                new_r_globals.insert(0, r_global_backface)

            if shape_augment:
                assert 'shape' in dset_fp, "Dataset path is probably wrong!"
                # Random sampling of shape deviations from original
                betas_delta = uniform_sample_smpl_shape_deviation(range=shape_range,
                                                                  fat_and_height_range=fat_height_range,
                                                                  num=num_augment_per_sample)
            if pose_augment:
                assert 'pose' in dset_fp, "Dataset path is probably wrong!"
                # Random sampling of axis and angle deviations from original pose
                new_poses = uniform_axis_angle_sample_pose_deviation(pose[3:],
                                                                     pose_axis_deviation_scale,
                                                                     pose_angle_deviation_range,
                                                                     num)

        for aug_idx in range(num_augment_per_sample):
            print('Aug', aug_idx)
            aug_pose = np.copy(pose)
            aug_betas = np.copy(betas)

            if global_pose_augment:
                aug_pose[:3] = new_r_globals[aug_idx]
            if shape_augment:
                aug_betas = aug_betas + betas_delta[aug_idx]
            if pose_augment:
                aug_pose[3:] = new_poses[aug_idx]

            aug_camera = {'rt': rt, 'f': f, 'trans': trans, 't': t, 'betas': aug_betas,
                          'pose': aug_pose}

            resolution = (resolution_wh, resolution_wh)
            factor = 1.0
            aug_renderings = render(MODEL_NEUTRAL,
                                    (np.asarray(resolution) * 1. / factor).astype('int'),
                                    aug_camera,
                                    1,
                                    False,
                                    use_light=False)
            aug_renderings = [scipy.misc.imresize(renderim,
                                                  (resolution[1],
                                                   resolution[0]),
                                                  interp='nearest')
                              for renderim in aug_renderings]
            aug_rendering = aug_renderings[0]  # Rendering 1 rotated view - 1 element in list

            aug_landmark_pos = get_landmark_positions(aug_betas, aug_pose, trans, resolution,
                                                      landmarks, rt, t, f)
            aug_vis = check_landmark_visibility(aug_landmark_pos, resolution_wh)

            class_groups = six_region_groups if partspec == '6' else None
            aug_annotation = regions_to_classes(aug_rendering, class_groups, warn_id=im_idx)
            if partspec == '1':
                aug_annotation = (aug_annotation > 0).astype('uint8')
            # assert np.max(annotation) <= int(partspec), (
            #     "Wrong annotation value (%s): %s!" % (im_idx, str(np.unique(annotation))))
            # if int(im_idx) == 0:
            #     assert np.max(annotation) == int(partspec), ("Probably an error in the number of parts!")
            aug_pose_vis_im = vs.visualize_pose(cv2.cvtColor(aug_annotation * 8, cv2.COLOR_GRAY2RGB),
                                                aug_landmark_pos,
                                                scale=1.)
            scipy.misc.imsave(path.join(dset_fp, '{}-{}_ann.png'.format(im_idx, aug_idx)),
                              aug_annotation)
            scipy.misc.imsave(path.join(dset_fp, '{}-{}_seg_ann_vis.png'.format(im_idx, aug_idx)),
                              apply_colormap(aug_annotation, vmax=int(partspec)))
            scipy.misc.imsave(path.join(dset_fp, '{}-{}_pose_ann_vis.png'.format(im_idx, aug_idx)),
                              aug_pose_vis_im)

            aug_landmark_pos_with_vis = np.concatenate([aug_landmark_pos, aug_vis[None, :]],
                                                       axis=0)
            np.save(str(path.join(dset_fp, '{}-{}_joints.npy'.format(im_idx, aug_idx))),
                    aug_landmark_pos_with_vis,
                    allow_pickle=False)
            aug_smpl_save_path = path.join(dset_fp, '{}-{}_body.pkl'.format(im_idx, aug_idx))
            with open(aug_smpl_save_path, 'wb') as aug_f:
                pickle.dump({'betas': aug_betas,
                             'pose': aug_pose},
                            aug_f,
                            protocol=2)


@click.command()
@click.argument("dset_fp", type=click.STRING)
@click.argument("partspec", type=click.Choice(['1', '6', '31']))
@click.argument("start", type=click.INT)
@click.argument("num", type=click.INT)
@click.option("--gpaug", "global_pose_augment", is_flag=True)
@click.option("--saug", "shape_augment", is_flag=True)
@click.option("--paug", "pose_augment", is_flag=True)
def cli(dset_fp, partspec, start, num, global_pose_augment=False, shape_augment=False,
        pose_augment=False, core_joints=True, resolution_wh=256):
    """Create segmentation datasets from select SMPL fits."""
    # np.random.seed(1)
    landmark_mapping = landmark_mesh_91
    if core_joints:
        LOGGER.info("Using the core joints.")
        # Order is important here! This way, we maintain LSP compatibility.
        landmark_mapping = OrderedDict([('neck', landmark_mapping['neck']),
                                        ('head_top', landmark_mapping['head_top']), ])
    n_landmarks = len(landmark_mapping) + 12
    LOGGER.info("Creating segmentation and pose dataset for {} classes and {} landmarks.".format(partspec, n_landmarks))

    LOGGER.info("Processing...")
    add_dataset(
        dset_fp,
        landmark_mapping,
        partspec,
        resolution_wh=resolution_wh,
        start=start,
        num=num,
        global_pose_augment=global_pose_augment,
        shape_augment=shape_augment,
        pose_augment=pose_augment)

    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("opendr.lighting").setLevel(logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter
