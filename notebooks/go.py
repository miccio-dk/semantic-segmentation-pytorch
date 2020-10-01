# System libs
import argparse
import os.path as osp
import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
import cv2
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode

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

def main():
    parser = argparse.ArgumentParser(description='run semantic segmentation on video.')
    #parser.add_argument('integers', help='an integer for the accumulator')input_path
    parser.add_argument('input_path', type=str, help='input video path')
    parser.add_argument('--dest_path', type=str, help='masks path', default=None)
    parser.add_argument('--dest_orig_path', type=str, help='orig frames path', default=None)
    parser.add_argument('--target_idx', type=int, help='orig frames path', default=12)
    args = parser.parse_args()

    target_idx = 12

    # load labels
    names = {}
    with open('data/object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]


    # Network Builders
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
    segmentation_module.cuda()


    step = 10
    frames = read_frame_from_videos(args.input_path, 432, 240)
    frames = frames[::step]
    print('# of frames:', len(frames))

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])
    img_original = numpy.array([numpy.array(frame) for frame in frames])
    img_data = torch.stack([pil_to_tensor(frame) for frame in frames])
    frames_batch = {'img_data': img_data.cuda()}
    output_size = img_data.shape[2:]
    print('Image size:', output_size)

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        scores = segmentation_module(frames_batch, segSize=output_size)
        
    # Get the predicted scores for each pixel
    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu().numpy()

    # store few frames
    dest_path = args.dest_path or (osp.splitext(args.input_path)[0] + '_masks')
    dest_orig_path = args.dest_orig_path or (osp.splitext(args.input_path)[0] + '_orig')
    print(dest_path, dest_orig_path)
    for i, imgs in enumerate(zip(pred, img_original)):
        mask, orig = imgs
        orig[mask == target_idx] = 0
        mask[mask != target_idx] = 0
        mask[mask == target_idx] = 255

        maskimg = PIL.Image.fromarray(numpy.uint8(mask))
        mask_path = osp.join(dest_path, f'img{i:04}.png')
        maskimg.save(mask_path)
        origimg = PIL.Image.fromarray(numpy.uint8(orig))
        orig_path = osp.join(dest_orig_path, f'img{i:04}.png')
        origimg.save(orig_path)
        
        
if __file__ == 'main':
    main()
