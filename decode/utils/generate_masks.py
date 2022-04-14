import sys
import decode
from decode.utils.gen_masks import generate_cell_masks
from pathlib import Path

param = decode.utils.param_io.load_params('konrads_params.yaml')

experiment_dir=sys.argv[1]
input(f"press enter if you want to generate masks for '{experiment_dir}' : ")
overwrite_data=input("overwrite previous masks? [yes/no] : ") in ("yes\n","yes","y","y\n")
print(f"{'' if overwrite_data else 'not '}overwriting previous data")

if str(experiment_dir)[0]!="/":
    experiment_dir=Path(".")/experiment_dir
else:
    experiment_dir=Path(experiment_dir)
    
assert experiment_dir.exists() and experiment_dir.is_dir()

generate_cell_masks(experiment_dir,threshold=0.9,overwrite_data=overwrite_data,
    fluo_roi=param.Simulation.fluo_roi,
    fluor_dir_name="fluor515",
    fluor_cropped_out_dir_name="fluor_cropped_515",
    transmat_file_name="transMatV_3D.mat",
    dir_dist_masks="dist_masks_515",
    dir_warped_dist_masks="warped_dist_masks_515")
param.Simulation.fluo_roi[1]=(param.Simulation.fluo_roi[1][0]-(2082//2),param.Simulation.fluo_roi[1][1]-(2082//2))
generate_cell_masks(experiment_dir,threshold=0.9,overwrite_data=overwrite_data,
    fluo_roi=param.Simulation.fluo_roi,
    fluor_dir_name="fluor580",
    fluor_cropped_out_dir_name="fluor_cropped_580",
    transmat_file_name="transMatC_3D.mat",
    dir_dist_masks="dist_masks_580",
    dir_warped_dist_masks="warped_dist_masks_580")