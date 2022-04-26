import torch


def get_cos_dist_matrix(features: torch.Tensor):
    norm_feat = torch.exp(features - torch.max(features))
    l2norm = torch.norm(norm_feat, dim=1, keepdim=True)
    unit_vec = norm_feat / l2norm
    cos_dist = torch.mm(unit_vec, unit_vec.transpose(0, 1))

    return torch.triu(cos_dist)
