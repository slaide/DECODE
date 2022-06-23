import numpy
import matplotlib.pyplot as plt
import decode
import torch
from pathlib import Path
from tqdm import tqdm
import pickle # to cache results
from decode.neuralfitter.train.masked_simulation import setup_masked_simulation

# continue training from a previous snapshot like this:
#  pyenv exec python -m decode.neuralfitter.train.train -p konrads_params.yaml -l out

class Entry:
    def __init__(self,model_file,param_file):
        self.model_file=Path(model_file)
        assert self.model_file.exists() and self.model_file.is_file()

        self.param_file=Path(param_file)
        assert self.param_file.exists() and self.param_file.is_file()

        self.raw_params=decode.utils.read_params(self.param_file)

        sim_train, sim_test = setup_masked_simulation(self.raw_params)

        self.sim_train=sim_train
        self.sim_test=sim_test

        assert sim_train.em_sampler.intensity_dist_type=="discrete"
        #set scaling parameter manually
        self.raw_params.Simulation.intensity_mu_sig=sim_train.em_sampler._intensity_mu_sig()

        self.params = decode.utils.param_io.autoset_scaling(self.raw_params)

        self.camera = decode.simulation.camera.Photon2Camera.parse(self.params)

        self.device=torch.device(self.params.Hardware.device)

        model_archs_available = {
            'SigmaMUNet': decode.neuralfitter.models.SigmaMUNet,
            'DoubleMUnet': decode.neuralfitter.models.model_param.DoubleMUnet,
            'SimpleSMLMNet': decode.neuralfitter.models.model_param.SimpleSMLMNet,
        }

        model_arch = model_archs_available[self.params.HyperParameter.architecture].parse(self.params)
        self.model = decode.utils.model_io.LoadSaveModel(model_arch,input_file=self.model_file, output_file=None).load_init(device=self.device).to(self.device)

        self.post_processor = decode.neuralfitter.utils.processing.TransformSequence([
            decode.neuralfitter.scale_transform.InverseParamListRescale(
                phot_max=self.params.Scaling.phot_max,
                z_max=self.params.Scaling.z_max,
                bg_max=self.params.Scaling.bg_max),
            decode.neuralfitter.coord_transform.Offset2Coordinate.parse(self.params),
            decode.neuralfitter.post_processing.SpatialIntegration(
                raw_th=self.params.PostProcessingParam.raw_th,
                xy_unit='px',
                px_size=self.params.Camera.px_size)
        ])

        self.frame_proc = decode.neuralfitter.scale_transform.AmplitudeRescale.parse(self.params)

    def localize(self,fluor_image):
        a=self.params.Simulation.img_size[0]
        b=self.params.Simulation.img_size[1]

        total_num_snippets=((fluor_image.shape[0]//a)+1)*((fluor_image.shape[1]//b)+1)
        assert total_num_snippets<=self.params.HyperParameter.batch_size,f"total number of snippets required for prediction of a whole frame: {total_num_snippets=} {self.params.HyperParameter.batch_size=}"

        #apply model to input
        batch_size=self.params.HyperParameter.batch_size
        snippets=numpy.zeros((batch_size,1,a,b))

        snippet_index=0
        for i in range(0,fluor_image.shape[0],a): 
            for j in range(0,fluor_image.shape[1],b):
                from_ax0=(i,i+a)
                from_ax1=(j,j+b)

                fluor_image_snippet=fluor_image[from_ax0[0]:from_ax0[1],from_ax1[0]:from_ax1[1]]

                snippets[snippet_index,0,:fluor_image_snippet.shape[0],:fluor_image_snippet.shape[1]]=fluor_image_snippet
                
                snippet_index+=1 #i//96*(fluor_image.shape[1]//96 +1)+j//96

        model_input_snippets=self.frame_proc.forward(torch.from_numpy(snippets).to(self.device).float())
        with torch.no_grad():
            result=self.model.forward(model_input_snippets)

        #print(f"{self.params.PostProcessingParam.raw_th=} {result[0,0].max().item()=}")

        #plt.figure(figsize=(15,5))
        #for slice_i in range(10):
        #    plt.subplot(1,10,slice_i+1)
        #    plt.imshow(result[0,slice_i].cpu().numpy())
        #plt.colorbar()
        #plt.show()

        #post-process output
        em_out=self.post_processor.forward(result)
        #em_out = infer.forward(model_input_snippets) # combines model forward and post-processing into a single function call

        #check snippetization
        reverse_snippets=numpy.zeros(fluor_image.shape)

        for i in range(0,batch_size):
            ax0_count=fluor_image.shape[0]//a
            if (fluor_image.shape[0]%b)>0:
                ax0_count+=1

            ax1_count=fluor_image.shape[1]//b
            if (fluor_image.shape[1]%a)>0:
                ax1_count+=1

            ax0l=(i//ax1_count)*a # axis 0 lower bound
            ax0u=(1+i//ax1_count)*b # axis 0 upper bound
            #assert (ax0u-ax0l)==a==b

            ax1l=(i%ax1_count)*a # axis 1 lower bound
            ax1u=(1+i%ax1_count)*b # axis 1 upper bound
            #assert (ax1u-ax1l)==a==b
            
            #check snippetization
            (ax0r,ax1r)=reverse_snippets[ax0l:ax0u,ax1l:ax1u].shape
            reverse_snippets[ax0l:ax0u,ax1l:ax1u]=snippets[i,0,:ax0r,:ax1r]

            em_out.xyz_px[em_out.frame_ix==i,0]+=ax0l
            em_out.xyz_px[em_out.frame_ix==i,1]+=ax1l

        assert ((fluor_image-reverse_snippets)==0).all()

        em_out.frame_ix[:]=0
        em_out.xyz_px=em_out.xyz_px.detach().cpu().float()

        return em_out

entry=Entry(
    #model_file="/home/patrick/code/test_decode/out/real_emitter_brightness_distribution/model_2.pt",
    model_file="/home/patrick/code/test_decode/out/low_cell_background/model_2.pt",
    param_file="/home/patrick/code/test_decode/konrads_params.yaml",
)

# mean image value is ~210 for image#0, ~150 for #200, ~140 for #999
image_list=range(200,1000,1)
coords_list=[]
for image_index in tqdm(image_list):
    image_adu=decode.utils.img_file_io.read_img(f"/mnt/big_data/code/test_decode/experiments/.membrane_associated/fluor580/img_000000{image_index:03}.tiff",'u12','u12')
    coords=entry.localize(image_adu)

    coords_list.append(coords)

#plt.imshow(image_adu,cmap="jet")
coords_list_numpy=numpy.concatenate([coords.xyz_nm.cpu().numpy() for coords in coords_list])

def plot_thing(image_axis_0,image_axis_1,step_size):
    x_lim=coords_list_numpy[:,image_axis_0].min(),coords_list_numpy[:,image_axis_0].max()
    y_lim=coords_list_numpy[:,image_axis_1].min(),coords_list_numpy[:,image_axis_1].max()

    x_bins=numpy.arange(*x_lim,step_size)
    y_bins=numpy.arange(*y_lim,step_size)

    hist,_bins_x,_bins_y=numpy.histogram2d(coords_list_numpy[:,image_axis_0],coords_list_numpy[:,image_axis_1],bins=(x_bins,y_bins))

    plt.figure()
    plt.imshow(hist,extent=(y_lim[0],y_lim[1],x_lim[0],x_lim[1]),cmap="jet")
    plt.xlim(*y_lim)
    axis_names=["x (short axis)","y (long axis)","z"]
    plt.xlabel(axis_names[image_axis_1])
    plt.ylim(*x_lim)
    plt.ylabel(axis_names[image_axis_0])
    #plt.scatter(coords.xyz_px.numpy()[:,1],coords.xyz_px.numpy()[:,0],s=10,marker="x",c="black")
    plt.colorbar()
    plt.show()

plot_thing(0,1,50) # xy-plot
plot_thing(2,0,10) # zx-plot
