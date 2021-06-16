import torch
import torch.utils.data
# from torch._six import container_abcs

import decode.generic


def smlm_collate(batch):
    """
    Collate for dataloader that allows for None return and EmitterSet.
    Otherwise defaults to default pytorch collate

    Args:
        batch
    """
    elem = batch[0]
    # ToDo: This is super ugly, however I don't know how to overcome this, because one must break out of recursion
    # BEGIN PARTLY INSERTION of default collate
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    # END INSERT
    elif elem is None:
        return None
    elif isinstance(elem, decode.generic.emitter.EmitterSet):
        return [em for em in batch]
    else:
        return torch.utils.data.dataloader.default_collate(batch)
