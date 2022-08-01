import decode
from decode.utils.entry import Entry
import scipy.io
import argparse

# use like this yenv exec python -m decode.utils.cli_entry /home/patrick/code/test_decode/out/new_cell_background/model_2.pt /home/patrick/code/test_decode/out/new_cell_background/param_run_in.yaml experiments/BY8853_with_dots/BY8853/EXP-22-BY8853/Run/Pos01/fluor515/img_000000000.tiff -o eval/results_for_one_image.mat

parser=argparse.ArgumentParser(description="apply DECODE to an image")
parser.add_argument("model_file",help="DECODE model file (file.pt)")
parser.add_argument("param_file",help="param file (params.yaml)")
parser.add_argument("image_file",help="image file (image.tif / image.tiff)")
parser.add_argument("-d","--device",default="cuda:0",choices=["cpu","cuda:0","cuda:1"],help="hardware decide to run localization on")
parser.add_argument("-o","--output_file",help="file that will contain all emitter data",required=True)

args=parser.parse_args()

entry=Entry(
    model_file=args.model_file,
    param_file=args.param_file,
)

image_adu=decode.utils.img_file_io.read_img(args.image_file)[1000:,:]
coords=entry.localize(image_adu)

import matplotlib.pyplot as plt
plt.imshow(image_adu)
plt.scatter(coords.xyz_px[:,1],coords.xyz_px[:,0])
plt.show()

scipy.io.savemat(args.output_file,{
    'coords_xyz_px':coords.xyz_px.cpu().numpy(),
    'brightness':coords.phot.cpu().numpy(),
    'probability':coords.prob.cpu().numpy(),
    'xyz_err':coords.xyz_sig.cpu().numpy(),
})