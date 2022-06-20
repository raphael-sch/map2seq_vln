import io
import os
import pickle

import PIL
from PIL import Image
import numpy as np

from graph_loader import GraphLoader
from nfov import NFOV

import tensorflow.compat.v2 as tf
import tf_image_processor

pano_dir = './panos'
draw_dir = 'output_draw'
output_dir = 'output'

do_draw = False

graph_nodes = GraphLoader('dataset', lambda s: s.split()).construct_graph().nodes
fov = 60
h_stride = 45
v_stride = 45
feature_height = 1
imsize = 224

horizontal_fov_frac = float(fov) / 360
vertical_fov_frac = 2 * horizontal_fov_frac
nfov_projector = NFOV(imsize, imsize, fov=(horizontal_fov_frac, vertical_fov_frac))

module_path = 'https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/3'
im_processor = tf_image_processor.TFImageProcessor(tf_hub_module_path=module_path)

if do_draw:
    os.makedirs(draw_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def run(panoid, pano_filename):
    print(panoid)

    img = PIL.Image.open(pano_filename)
    img = np.array(img)

    pano_heading_features = dict()

    pano_yaw_angle = graph_nodes[panoid].pano_yaw_angle
    global_headings = {0}
    global_headings.update(graph_nodes[panoid].neighbors.keys())
    for global_heading in global_headings:
        pano_heading = global_heading - pano_yaw_angle
        if pano_heading < 0:
            pano_heading += 360
        pano_heading = pano_heading - 180
        if pano_heading < 0:
            pano_heading += 360

        print(global_heading, pano_heading)

        heading_features = list()
        pano_heading_scaled = pano_heading / 360.0
        slice_centers = [pano_heading_scaled - 0.25, pano_heading_scaled - 0.125, pano_heading_scaled, pano_heading_scaled + 0.125, pano_heading_scaled + 0.25]
        assert len(slice_centers) == 5
        for i, slice_center in enumerate(slice_centers):
            if slice_center < 0:
                slice_center += 1
            if slice_center > 1:
                slice_center -= 1

            #print(slice_center)
            img_slice_npy = nfov_projector.to_nfov(img, np.array([slice_center, 0.5]))
            image_buffer = io.BytesIO()
            pil_image = PIL.Image.fromarray(img_slice_npy)
            pil_image.save(image_buffer, format="jpeg")
            view_features = im_processor.process(image_buffer.getvalue())
            heading_features.append(view_features)

            if do_draw:
                draw_image(img_slice_npy, panoid + '_' + str(int(global_heading)) + '_' + str(i))

        pano_heading_features[global_heading] = np.asarray(heading_features, dtype=np.float32)
        #print(panoid, global_heading, pano_heading)
    #print(panoid, pano_heading_features.keys())
    return pano_heading_features


def draw_image(img, name):
    img = Image.fromarray(img, mode='RGB')
    img.save(os.path.join(draw_dir, f'{name}.png'))


if __name__ == '__main__':
    all_pano_heading_features = dict()
    pano_ids = list(graph_nodes.keys())

    n_processed = 0
    for i, pano_id in enumerate(pano_ids):

        pano_filename = os.path.join(pano_dir, pano_id + '.jpg')
        if os.path.isfile(pano_filename):
            pano_heading_features = run(pano_id, pano_filename)
            all_pano_heading_features[pano_id] = pano_heading_features
            n_processed += 1
            print(n_processed, 'of', len(pano_ids))

    with open(os.path.join(output_dir, 'resnet_last_layer.pickle'), 'wb') as f:
        pickle.dump(all_pano_heading_features, f)

    assert n_processed == len(pano_ids)