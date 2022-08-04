import numpy
import matplotlib.pyplot as plt
import decode
import torch
from pathlib import Path
from tqdm import tqdm
import pickle # to cache results
from decode.neuralfitter.train.masked_simulation import setup_masked_simulation
import scipy.io

# continue training from a previous snapshot like this:
#  pyenv exec python -m decode.neuralfitter.train.train -p konrads_params.yaml -l out

class Entry:
    def __init__(self,model_file,param_file,device):
        self.model_file=Path(model_file)
        assert self.model_file.exists() and self.model_file.is_file(), f"DECODE model file not found {model_file}"

        self.param_file=Path(param_file)
        assert self.param_file.exists() and self.param_file.is_file(), f"param file not found {param_file}"

        self.raw_params=decode.utils.read_params(self.param_file)

        self.sim_train, _sim_test = setup_masked_simulation(self.raw_params)

        self.params = self.raw_params if 1 else decode.utils.param_io.autoset_scaling(self.raw_params)

        self.camera = decode.simulation.camera.Photon2Camera.parse(self.params)

        self.device=torch.device(device)

        model_archs_available = {
            'SigmaMUNet': decode.neuralfitter.models.SigmaMUNet,
            #'DoubleMUnet': decode.neuralfitter.models.model_param.DoubleMUnet,
            #'SimpleSMLMNet': decode.neuralfitter.models.model_param.SimpleSMLMNet,
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

        dim0_list=list(range(0,fluor_image.shape[1],a))
        dim1_list=list(range(0,fluor_image.shape[0],b))

        total_num_snippets=len(dim0_list)*len(dim1_list)

        offsets0=[] #numpy.expand_dims(numpy.concatenate([[0],numpy.array([a]).repeat(len(dim1_list)-1)]).cumsum(),0).repeat(len(dim0_list),0).flatten()
        offsets1=[] #(numpy.array([b]).repeat(len(dim0_list)).cumsum()-b).repeat(len(dim1_list))

        #copy input image into snippets fit for model consumption
        batch_size=self.params.HyperParameter.batch_size
        snippets=numpy.zeros((total_num_snippets if total_num_snippets%batch_size==0 else total_num_snippets+(batch_size-total_num_snippets%batch_size),1,a,b),dtype=numpy.float32)

        snippet_index=0
        for i in dim0_list:
            for j in dim1_list:
                from_ax0=(i,i+a)
                from_ax1=(j,j+b)

                offsets1.append(i)
                offsets0.append(j)

                fluor_image_snippet=fluor_image[from_ax1[0]:from_ax1[1],from_ax0[0]:from_ax0[1]]
                # partial snippets are not discarded or shifted
                snippets[snippet_index,0,:fluor_image_snippet.shape[0],:fluor_image_snippet.shape[1]]=fluor_image_snippet
                
                snippet_index+=1

        offsets0=numpy.array(offsets0)
        offsets1=numpy.array(offsets1)

        # if size of image does not fit into the number of snippets contained in a single batch, apply decode to all batches, then combine results
        if total_num_snippets<batch_size:
            model_input_snippets=self.frame_proc.forward(torch.from_numpy(snippets).to(self.device).float())

            plt.imshow(model_input_snippets[0])
            plt.colorbar()
            plt.show()

            with torch.no_grad():
                result=self.model.forward(model_input_snippets)
                em_out=self.post_processor.forward(result)
        else:
            for i in range(0,snippets.shape[0],batch_size):
                model_input_snippets=self.frame_proc.forward(torch.from_numpy(snippets[i:i+batch_size]).to(self.device).float())

                with torch.no_grad():
                    partial_result=self.model.forward(model_input_snippets)
                    partial_em_out=self.post_processor.forward(partial_result)
                    partial_em_out.frame_ix[:]+=i

                if i==0:
                    em_out=partial_em_out
                else:
                    em_out=decode.generic.emitter.EmitterSet.cat([em_out,partial_em_out])

            em_out.xyz_px[:,0]+=offsets0[em_out.frame_ix]
            em_out.xyz_px[:,1]+=offsets1[em_out.frame_ix]

        em_out.frame_ix[:]=0
        em_out.xyz_px=em_out.xyz_px.detach().cpu().float()

        return em_out

