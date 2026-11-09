"""Models and freezing modes (protocol A1.1, sections 3-4).

Modes:
  scratch  random init, everything trained
  linear   ImageNet weights, only the new head is trained (backbone frozen)
  partial  ImageNet weights, first half of the backbone (by depth) frozen
  full     ImageNet weights, everything trained
"""
import timm
import torch.nn as nn

CHECKPOINTS = {"resnet50": "resnet50.a1_in1k", "deit_s": "deit_small_patch16_224.fb_in1k"}
MODES = ("scratch", "linear", "partial", "full")


def head_of(model):
    """The classification layer that timm re-created for 101 classes."""
    return model.get_classifier()


def frozen_prefixes(arch, mode):
    """Names (state_dict prefixes) of the parameters that are NOT trained."""
    if mode in ("scratch", "full"):
        return ()
    if mode == "linear":
        return ("",)                               # everything except the head (see apply_freeze)
    if arch == "resnet50":                          # 7 of 16 bottleneck blocks (3-4-6-3)
        return ("conv1.", "bn1.", "layer1.", "layer2.")
    if arch == "deit_s":                            # 6 of 12 transformer blocks
        return ("patch_embed.", "cls_token", "pos_embed") + tuple(f"blocks.{i}." for i in range(6))
    raise ValueError(arch)


def build(arch, mode, num_classes=101):
    model = timm.create_model(CHECKPOINTS[arch], pretrained=mode != "scratch", num_classes=num_classes)
    head = {id(p) for p in head_of(model).parameters()}
    prefixes = frozen_prefixes(arch, mode)
    for name, p in model.named_parameters():
        if id(p) not in head and any(name.startswith(pre) for pre in prefixes):
            p.requires_grad = False
    return model


def frozen_bn_to_eval(model):
    """Call after model.train(): BatchNorm layers whose weights are frozen must also keep
    their ImageNet running statistics (otherwise the frozen part still changes)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def param_groups(model, mode, lr_head, lr_backbone, wd):
    """Head and backbone get different learning rates (protocol table 'Hyperparameter').
    No weight decay on biases and norm parameters (1-D tensors)."""
    head = {id(p) for p in head_of(model).parameters()}
    groups = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        lr = lr_head if (id(p) in head or mode == "scratch") else lr_backbone
        decay = wd if p.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(p)
    return [{"params": ps, "lr": lr, "weight_decay": d} for (lr, d), ps in groups.items()]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
