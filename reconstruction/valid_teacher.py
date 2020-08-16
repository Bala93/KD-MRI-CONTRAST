import pathlib
import sys
from collections import defaultdict
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import SliceData
#from models import UnetModel,conv_block
#from models import DnCn2
#from models import TeacherNet,StudentNet
from models import DCTeacherNet,DCStudentNet
import h5py
from tqdm import tqdm

def save_reconstructions(reconstructions, out_dir):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.
    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
    """
    out_dir.mkdir(exist_ok=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)


def create_data_loaders(args):

    data = SliceData(args.data_path,args.acceleration_factor,'validation')
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
    )

    return data_loader


def load_model(model_type,checkpoint_file):

    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']

    if model_type == 'teacher':
        model = DCTeacherNet(args).to(args.device)
    else:
        model = DCStudentNet(args).to(args.device)

    model.load_state_dict(checkpoint['model'])

    return model


def run_model(args, model,data_loader):

    model.eval()
    reconstructions_flair = defaultdict(list)
    reconstructions_t1 = defaultdict(list)

    with torch.no_grad():
        for (iter,data) in enumerate(tqdm(data_loader)):

            flair_img, flair_kspace, t1_img, t1_kspace, flair_target, t1_target, fnames, slices = data # Return kspace also we can ignore that for train and test 
            
            input = torch.stack([flair_img, t1_img], dim=1)
            input_kspace = torch.stack([flair_kspace, t1_kspace], dim=1)
            target = torch.stack([flair_target, t1_target], dim=1)
    
            input = input.float().to(args.device)
            input_kspace = input_kspace.to(args.device)
            target = target.float().to(args.device)
    
            #print (input.shape, input_kspace.shape)
    
            recons = model(input, input_kspace)[-1]
     
            recons = recons.to('cpu')

            for i in range(recons.shape[0]):
                reconstructions_flair[fnames[i]].append((slices[i].numpy(), recons[i][0].numpy()))
                reconstructions_t1[fnames[i]].append((slices[i].numpy(), recons[i][1].numpy()))

    #print (reconstructions_flair.items())

    reconstructions_flair = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions_flair.items()
    }

    reconstructions_t1 = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions_t1.items()
    }

    return reconstructions_flair, reconstructions_t1


def main(args):
    
    data_loader = create_data_loaders(args)
    model = load_model(args.model_type,args.checkpoint)

    reconstructions_flair,reconstructions_t1 = run_model(args, model, data_loader)
    save_reconstructions(reconstructions_flair, args.out_dir / 'flair')
    save_reconstructions(reconstructions_t1, args.out_dir / 't1')


def create_arg_parser():

    parser = argparse.ArgumentParser(description="Valid setup for MR recon U-Net")
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int, help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Which device to run on')
    parser.add_argument('--data-path',type=str,help='path to validation dataset')

    parser.add_argument('--acceleration_factor',type=str,help='acceleration factors')
    parser.add_argument('--dataset_type',type=str,help='cardiac,kirby')
    parser.add_argument('--usmask_path',type=str,help='undersampling mask path')
    #parser.add_argument('--data_consistency',action='store_true')
    parser.add_argument('--model_type',type=str,help='model type teacher student')
    
    return parser

if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])
    main(args)
