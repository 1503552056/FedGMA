import torch
def flatten_state_dict(state_dict, keys=None):
    vecs = []
    meta = []
    items = list(state_dict.items()) if keys is None else [(k, state_dict[k]) for k in keys]
    for name, tensor in items:
        t = tensor.detach().reshape(-1)
        vecs.append(t)
        meta.append((name, tuple(tensor.shape)))
    if len(vecs) == 0:
        return torch.zeros(0), []
    return torch.cat(vecs), meta
def apply_vector_to_state_dict(model, vec, meta):
    offset = 0
    for name, shape in meta:
        numel = 1
        for s in shape: numel *= s
        chunk = vec[offset: offset+numel].view(shape)
        offset += numel
        p = dict(model.named_parameters()).get(name, None)
        if p is not None:
            p.data.copy_(chunk)
