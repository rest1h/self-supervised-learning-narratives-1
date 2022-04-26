import torch


def get_cos_dist_matrix(features: torch.Tensor, device: str):
    norm_feat = torch.exp(features - torch.max(features)).to(device)
    l2norm = torch.norm(norm_feat, dim=1, keepdim=True).to(device)
    unit_vec = (norm_feat / l2norm).to(device)
    cos_dist = torch.mm(unit_vec, unit_vec.transpose(0, 1)).to(device)

    return torch.triu(cos_dist).to(device)
