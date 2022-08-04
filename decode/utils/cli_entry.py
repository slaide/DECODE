import decode
from decode.utils.entry import Entry
import scipy.io
import argparse
import skimage
from pathlib import Path

# use like this: pyenv exec python -m decode.utils.cli_entry /home/patrick/code/test_decode/out/new_cell_background experiments/BY8853_with_dots/BY8853/EXP-22-BY8853/Run/Pos01/fluor515/img_000000000.tiff -o eval/results_for_one_image.mat

parser=argparse.ArgumentParser(description="apply DECODE to an image")
parser.add_argument("model",help="DECODE model (folder)")
parser.add_argument("image_file",help="image file (image.tif / image.tiff)")
parser.add_argument("-d","--device",default="cuda:0",choices=["cpu","cuda:0","cuda:1"],help="hardware decide to run localization on")
parser.add_argument("-o","--output_file",help="file that will contain all emitter data",required=True)

args=parser.parse_args()

entry=Entry(
    model_file=str(Path(args.model)/"model_2.pt"),
    param_file=str(Path(args.model)/"param_run_in.yaml"),
    device=args.device,
)

#minx,miny,maxx,maxy=(262,1140,1081,2040) # maybe user-definable ROI? output coordinates would need to be adjusted

image_adu=skimage.io.imread(args.image_file, plugin="tifffile", as_gray = True)
coords=entry.localize(image_adu)

# switch x and y axis to match matlabs coordinate format
coords.xyz_px[:,[0,1]]=coords.xyz_px[:,[1,0]]
coords.xyz_sig[:,[0,1]]=coords.xyz_sig[:,[1,0]]

scipy.io.savemat(args.output_file,{
    'coords_xyz_px':coords.xyz_px.cpu().numpy(),
    'brightness':coords.phot.cpu().numpy(),
    'probability':coords.prob.cpu().numpy(),
    'xyz_err':coords.xyz_sig.cpu().numpy(),
})