import os
import pickle
import sys

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from cut_panos import get_slices
from graph_loader import GraphLoader


panos_dir = './panos'
output_dir = './output'
#panos_dir = 'panos/jpegs_manhattan_touchdown'
graph = GraphLoader('./dataset').construct_graph()

os.makedirs(output_dir, exist_ok=True)
panoid_finished = set([f[:-4] for f in os.listdir(output_dir)])

# Load the pretrained model
model = models.resnet18(pretrained=True) # Use the model object to select the desired layer
#print(model)
newmodel = torch.nn.Sequential(*(list(model.children())[:-4]))
#print('\n\n')
#print(newmodel)
#exit()
newmodel.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def roll_img(image_feature, panoid, heading):
    shift_angle = graph.nodes[panoid].pano_yaw_angle - heading
    width = image_feature.shape[1]
    shift = int(width * shift_angle / 360)
    image_feature = np.roll(image_feature, shift, axis=1)  # like 'abcd' -> 'bcda'
    return image_feature


all_pano_heading_features = dict()
n_processed = 0
for i, (panoid, node) in enumerate(sorted(graph.nodes.items())):

    pano_filepath = os.path.join(panos_dir, panoid + '.jpg')
    if not os.path.isfile(pano_filepath):
        continue

    slices = list()
    for slice in get_slices(pano_filepath):
        slice = normalize(to_tensor(slice))
        slices.append(slice)

    batch = Variable(torch.stack(slices, dim=0))
    print(batch.shape)
    output = newmodel(batch)
    print(output.shape)

    features = torch.cat(list(output), dim=-2)
    #print(features.shape)
    features = torch.mean(features, dim=0)
    #print(features.shape)
    features = features.detach().numpy()
    print(features.shape)


    headings = list(node.neighbors.keys()) + [0]

    features_heading = dict()

    for heading in headings:
        image_feature = roll_img(features.copy(), panoid, heading)
        image_feature = image_feature[182:282, :]
        features_heading[heading] = image_feature
        print(image_feature.shape)
        assert image_feature.shape == (100, 100)

    all_pano_heading_features[panoid] = features_heading

    n_processed += 1
    print(n_processed, 'of', len(graph.nodes))

with open(os.path.join(output_dir, 'resnet_fourth_layer.pickle'), 'wb') as f:
    pickle.dump(all_pano_heading_features, f)

assert n_processed == len(graph.nodes)
