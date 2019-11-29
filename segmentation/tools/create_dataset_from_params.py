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


def add_dataset(dset_fp, landmarks, partspec, resolution_wh=256, num_zfill_in_name=6,
                start=0, num=0):
    """Add a dataset to the collection."""
    ids_list = [str(f[:num_zfill_in_name]) for f in sorted(os.listdir(dset_fp))
                if f.endswith("_body.pkl")]
    print(ids_list)
    to_render_ids = ids_list[start:start+num]
    print(to_render_ids)
    LOGGER.info("Writing dataset...")

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
        landmark_pos = np.concatenate((landmark_pos, np.ones((1, landmark_pos.shape[-1]))))
        # TODO currently marking all joints as visible, would be easy to simple check if they
        # actually are.
        np.save(str(path.join(dset_fp, '{}_joints.npy'.format(im_idx))),
                landmark_pos,
                allow_pickle=False)


@click.command()
@click.argument("dset_fp", type=click.STRING)
@click.argument("partspec", type=click.Choice(['1', '6', '31']))
@click.argument("start", type=click.INT)
@click.argument("num", type=click.INT)
def cli(dset_fp, partspec, start, num, core_joints=True, resolution_wh=256):
    """Create segmentation datasets from select SMPL fits."""
    np.random.seed(1)
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
        num=num)

    LOGGER.info("Done.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format=LOGFORMAT)
    logging.getLogger("opendr.lighting").setLevel(logging.WARN)
    cli()  # pylint: disable=no-value-for-parameter
