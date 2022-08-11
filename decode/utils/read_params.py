from pathlib import Path
from typing import Union

import decode
from decode.utils.types import RecursiveNamespace


def read_params(path:Union[str,Path]):
  """
  workaround for several bugs and inconveniences is DECODE
  """
  if isinstance(path,str):
    path=Path(path)

  assert path.exists()
  assert path.suffix in (".yaml",".yml")

  param=decode.utils.param_io.load_params(str(path))

  param.Simulation.fluo_roi=(param.Simulation.emitter_extent[0],param.Simulation.emitter_extent[1])
  param.Simulation.psf_extent_img=(
    (
      0,
      param.Simulation.fluo_roi[1][1] - param.Simulation.fluo_roi[1][0],
    ),
    (
      0,
      param.Simulation.fluo_roi[0][1] - param.Simulation.fluo_roi[0][0],
    )
  )
  param.Simulation.psf_extent=(
    (
      0,
      param.Simulation.img_size[0]
    ),(
      0,
      param.Simulation.img_size[1]
    )
  )

  param.TestSet=RecursiveNamespace()
  param.TestSet.test_size=512

  # this needs to be specified to allow setting img_size to something other than 40x40
  param.TestSet.img_size=param.Simulation.img_size
  param.TestSet.frame_extent=((0,param.Simulation.img_size[0]),(0,param.Simulation.img_size[1]))

  return param