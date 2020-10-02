# System libs
import warnings
import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch, PIL.Image, torchvision.transforms
import cv2
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

warnings.simplefilter("ignore", UserWarning)

#  read frames from video 
def read_frame_from_videos(vname, w, h):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image.resize((w, h)))
        success, image = vidcap.read()
        count += 1
    return frames

# generate mask based on list of labels
def get_mask(scores, idxs, threshold=0.5):
    pred = np.zeros(scores.shape[1:])
    for idx in idxs:
        pred += scores[idx].numpy()
    pred[pred > threshold] = 0.9999
    return pred

def main():
    # read labels
    labels = pd.read_csv('data/object150_info.csv')
    labels = list(labels['Name'])
    labels = [lbl.split(';')[0] for lbl in labels]
    labels_str = 'Available labels: \n' + '\n'.join([f' - {i:3}: {lbl}' for i, lbl in enumerate(labels)])

    # define input args
    parser = argparse.ArgumentParser(description='run semantic segmentation on video', epilog=labels_str)
    parser.add_argument('input_path', type=str, help='input video path')
    parser.add_argument('--dest_path', type=str, help='masks path', default=None)
    parser.add_argument('--dest_orig_path', type=str, help='orig frames path', default=None)
    parser.add_argument('--step', type=int, help='frame skip', default=1)
    parser.add_argument('--bs', type=int, help='batch size', default=16)
    parser.add_argument('--labels', type=list, nargs='+', help='use --show_labels for full list', default=[12, 116, 20])
    args = parser.parse_args()

    # process paths
    dest_path = args.dest_path or (osp.splitext(args.input_path)[0] + '_masks')
    dest_orig_path = args.dest_orig_path or (osp.splitext(args.input_path)[0] + '_orig')
    if not osp.exists(dest_path):
                 os.makedirs(dest_path) 
    if not osp.exists(dest_orig_path):
                 os.makedirs(dest_orig_path) 
    print('Output dirs', dest_path, dest_orig_path)

    # network builders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50',
        fc_dim=2048,
        weights='ckpt/upernet50/encoder_epoch_30.pth')
    net_decoder = ModelBuilder.build_decoder(
        arch='upernet',
        fc_dim=2048,
        num_class=150,
        weights='ckpt/upernet50/decoder_epoch_30.pth',
        use_softmax=True)
    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.eval()
    segmentation_module.to(device)

    # load and normalize video
    frames = read_frame_from_videos(args.input_path, 432, 240)
    frames = frames[::args.step]
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    img_original = torch.stack([torch.Tensor(np.array(frame)) for frame in frames])
    img_data = torch.stack([pil_to_tensor(frame) for frame in frames])
    print('Data shape (orig/data):', list(img_original.shape), list(img_data.shape))

    # split data into chunks
    n_chunks = img_data.shape[0] // args.bs
    img_original_chunks = torch.chunk(img_original, n_chunks)
    img_data_chunks = torch.chunk(img_data, n_chunks)
    print('Number of chunks', len(img_data_chunks))

    # loop each chunk
    k = 0
    for j, dat in enumerate(zip(img_original_chunks, img_data_chunks)):
        img_orig, img_chunk = dat
        frames_batch = {'img_data': img_chunk.to(device)}
        output_size = img_chunk.shape[2:]
        print(f' - {j} Chunk shape (orig/data):', list(img_orig.shape), list(img_chunk.shape))

        # run segmentation at highest resolution
        with torch.no_grad():
            scores = segmentation_module(frames_batch, segSize=output_size)

        # store frames
        for imgs in zip(scores, img_orig):
            score, orig = imgs
            mask = get_mask(score, args.labels)
            orig = orig.detach().numpy().copy()
            
            # calcualte masked frames
            mskorig = (orig.T * (1 - mask.T)).T

            # convert to pil images and store
            mask_img = PIL.Image.fromarray(np.uint8(mask))
            mask_path = osp.join(dest_path, f'img{k:04}.png')
            mask_img.save(mask_path)
            mskorig_img = PIL.Image.fromarray(np.uint8(mskorig))
            orig_path = osp.join(dest_orig_path, f'img{k:04}.png')
            mskorig_img.save(orig_path)
            k += 1
        
        
if __name__ == "__main__":   
     main()
