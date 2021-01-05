# The purpose of this is to check the availability of the assets. This of course requires web connection of the
# machine that tests
from pathlib import Path

import requests
import yaml
import pytest

import  decode.utils.loader


@pytest.fixture(scope='module')
def gateway_public():
    url = 'https://raw.githubusercontent.com/TuragaLab/DECODE/master/gateway.yaml'
    r = requests.get(url, allow_redirects=True)

    return yaml.load(r.content, Loader=yaml.FullLoader)


@pytest.fixture(scope='module')
def gateway_host(path: Path = Path(__file__).parent.parent.parent / 'gateway.yaml'):
    assert path.is_file(), "Host gateway path is incorrect or does not exist."

    with path.open('r') as p:
        y = yaml.load(p, Loader=yaml.FullLoader)

    return y


@pytest.mark.web
@pytest.mark.slow
@pytest.mark.parametrize("gate_type", ['host', 'public'])  # unfortunately fixture in paramet. does not work yet
def test_examples(gate_type, gateway_host, gateway_public, tmpdir):

    gate = gateway_public if gate_type == 'public' else gateway_host
    examples = gate['examples']

    for k, v in examples.items():
        fpath = Path(tmpdir / (v['name'] + '.zip'))

        decode.utils.loader.load(fpath, url=v['url'], hash=v['hash'])
        if not decode.utils.loader.check_file(fpath, hash=v['hash']):
            raise FileNotFoundError(f"Load check of example package {k} with name {v['name']} failed. "
                                    f"File does not exists or hash does not match.")