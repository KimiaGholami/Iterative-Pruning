import torch
import torch.nn as nn
import logging
from transformers import AutoModelForCausalLM
import math
from torch.distributed.tensor import DTensor, Replicate, distribute_tensor, Shard
import torch.distributed as dist
from typing import Optional, List, Tuple, Literal, Union
import enum
from dataclasses import dataclass
import numpy as np
from torchao.optim.subclass_8bit import OptimState8bit
from torchao.optim.subclass_4bit import OptimState4bit

def _to_np_f32(x: torch.Tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        return x.float().cpu().numpy()
    return x.detach().cpu().to(torch.float32).numpy()

def get_llm(
    model_name:str, 
    seqlen:int=2048
)-> AutoModelForCausalLM:
    """
    Load the model from huggingface hub or local directory.
    The model should be a causal language model, such as Llama2, Gemma, etc.
    Args:
        Model_name: str, directly from huggingface hub, or the directory of the model.
        seqlen: int, the maximum sequence length for the model.
    Returns:
        model: AutoModelForCausalLM, the model loaded from huggingface hub.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True,
        device_map="cpu",
        trust_remote_code = True,
        use_flash_attention_2=False
    )
    assert seqlen<=model.config.max_position_embeddings, f"seqlen({seqlen}) should be less than or equal to model.config.max_position_embeddings({model.config.max_position_embeddings})"
    model.seqlen = seqlen
    return model

def _as_dense_a(a):
    """Convert importance matrix to dense tensor if it is DTensor."""
    if a is None:
        return None
    if isinstance(a, DTensor):
        return a.redistribute(placements=[Replicate()]).to_local()
    return a

def _proj_impl_dense(
    weight: torch.Tensor,
    a: Optional[torch.Tensor],
    sparsity: float,
    prune_n: int,
    prune_m: int,
    comparison_group: str,
) -> torch.Tensor:
    """
    Projection core for dense tensors.
    - Works entirely on local tensor (no communication).
    - Supports unstructured and n:m semi-structured sparsity.
    - comparison_group:
        'layer'  : global threshold (within this tensor)
        'column' : prune k smallest entries per column
        'row'    : prune k smallest entries per row
    """
    device = weight.device
    new_z = weight.detach().clone()

    if a is not None:
        if isinstance(a,OptimState8bit):
            a = a.dequantize()
        elif isinstance(a,OptimState4bit):
            a = a.dequantize()
        
            z_metric = a * (weight**2)
    else:
        # Standard projection: metric is |x|
        z_metric = weight.abs()

    if prune_n != 0 and prune_m != 0:
        # n:m semi-structured pruning (column-block based)
        z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
        cols = z_metric.shape[1]
        for ii in range(0, cols, prune_m):
            blk = z_metric[:, ii:ii + prune_m].float()
            if blk.numel() == 0:
                continue
            k = min(prune_n, blk.shape[1])
            if k <= 0:
                continue
            # Select k smallest entries per row in the block
            _, idxs = torch.topk(blk, k=k, dim=1, largest=False)
            z_mask.scatter_(1, ii + idxs, True)
        new_z[z_mask] = 0
        return new_z

    # ---- Unstructured sparsity ----
    if comparison_group == "layer":
        # Global threshold within this tensor
        k = int(new_z.numel() * sparsity)
        if k > 0:
            if a is None:
                z_metric = weight.detach().abs()
            else:
                z_metric = a * weight.detach().pow(2)
            flat_sorted = torch.sort(z_metric.flatten(), stable=True)[0]
            kth = flat_sorted[min(k - 1, flat_sorted.numel() - 1)]
            new_z[z_metric <= kth] = 0
        return new_z

    elif comparison_group == "column":
        # Prune per column: smallest k rows
        num_rows_to_prune_per_col = int(new_z.shape[0] * sparsity)
        if num_rows_to_prune_per_col > 0:
            z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
            _, idx = torch.topk(z_metric, k=num_rows_to_prune_per_col, dim=0, largest=False)
            z_mask.scatter_(0, idx, True)
            new_z[z_mask] = 0
        return new_z

    else:  # 'row'
        # Prune per row: smallest k columns
        num_cols_to_prune_per_row = int(new_z.shape[1] * sparsity)
        if num_cols_to_prune_per_row > 0:
            z_mask = torch.zeros_like(new_z, dtype=torch.bool, device=device)
            _, idx = torch.topk(z_metric, k=num_cols_to_prune_per_row, dim=1, largest=False)
            z_mask.scatter_(1, idx, True)
            new_z[z_mask] = 0
        return new_z

def projection(
    w: List[torch.Tensor],
    sparsity: float,
    prune_n: int = 0,
    prune_m: int = 0,
    importance_matrix: Optional[List[torch.Tensor]] = None,
    comparison_group: str = "layer",
) -> List[torch.Tensor]:
    """
    Distributed/DTensor-friendly projection.
    If weight is a DTensor, it is always replicated before projection.
    """
    assert comparison_group in ("layer", "column", "row")
    use_a = importance_matrix is not None
    if use_a:
        assert len(importance_matrix) == len(w)

    out: List[torch.Tensor] = []
    for i, weight in enumerate(w):
        a = importance_matrix[i] if use_a else None
        if isinstance(weight, DTensor):
            # Always replicate DTensor to a dense tensor for projection
            mesh = weight.device_mesh
            orig_places = weight.placements
            
            dense_w = _as_dense_a(weight)
            dense_a = _as_dense_a(a)
            new_dense = _proj_impl_dense(dense_w, dense_a, sparsity, prune_n, prune_m, comparison_group)
            
            # Redistribute the result back to the original sharding
            new_dt = distribute_tensor(new_dense, device_mesh=mesh, placements=orig_places)
            out.append(new_dt)
        else:
            # It's a regular dense tensor
            new_dense = _proj_impl_dense(weight, _as_dense_a(a), sparsity, prune_n, prune_m, comparison_group)
            out.append(new_dense)
            
    return out


def find_layers(
    module: nn.Module,
    layers: list = [nn.Linear],
    name: str = ''
) -> dict:
    """
    Recursively find the layers of a certain type in a module.
    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find. 
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_model_layers(model):
    """
    Returns the list of Transformer layers based on the model architecture.

    Args:
        model (nn.Module): A Hugging Face Transformer model object.

    Returns:
        nn.ModuleList: The list of Transformer layers in the model.

    Raises:
        ValueError: If the model architecture is unsupported.
    """

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama, Gemma, Mistral, etc.
        return model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT, etc.
        return model.model.decoder.layers
    else:
        raise ValueError("Unsupported model architecture: Cannot find layers.")

def check_sparsity(model, log_by_block: bool = False):
    """
    Calculates the sparsity (ratio of zero parameters) of the model's linear layers.

    Args:
        model (nn.Module): The model object to check sparsity for.
        log_by_block (bool): If True, logs the sparsity for each transformer block.

    Returns:
        float: The overall sparsity ratio of the linear layers in the model (0.0 to 1.0).
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_model_layers(model)

    count = 0
    total_params = 0
    
    if log_by_block:
        logging.info("Checking sparsity per block...")

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer) # Find linear layers within the layer

        sub_count = 0
        sub_params = 0
        for name in subset:
            # Check if the layer has a weight parameter
            if hasattr(subset[name], 'weight') and subset[name].weight is not None:
                W = subset[name].weight.data
                
                zeros = (W == 0).sum().item()
                total = W.numel()

                count += zeros
                total_params += total
                
                sub_count += zeros
                sub_params += total

        if log_by_block:
            layer_sparsity = float(sub_count) / sub_params if sub_params > 0 else 0.0
            logging.info(f"  - Block {i:02d} sparsity: {layer_sparsity:.4f}")

    model.config.use_cache = use_cache
    overall_sparsity = float(count) / total_params if total_params > 0 else 0.0
    
    if log_by_block:
        logging.info(f"Overall linear layer sparsity: {overall_sparsity:.4f}")
        
    return overall_sparsity

import logging as stdlogging

logger = stdlogging.getLogger(__name__)




Granularity = Literal["tensorwise", "rowwise"]


class ScalingType(enum.Enum):
    DYNAMIC = 0   # use current step's range (torchao: dynamic)
    NONE = 1      # do not update scales automatically


@dataclass
class FP8Config:
    fp8_dtype: torch.dtype = torch.float8_e4m3fn  # torchao: float8_e4m3fn/float8_e5m2
    scaling_type: ScalingType = ScalingType.DYNAMIC  # torchao: dynamic/none
    granularity: Granularity = "tensorwise"
    safety_margin: float = 1.05
    sync_scales: bool = True
    process_group: Optional[dist.ProcessGroup] = None

    def torch_dtype(self) -> torch.dtype:
        if self.fp8_dtype == torch.float8_e4m3fn:
            return torch.float8_e4m3fn
        elif self.fp8_dtype == torch.float8_e5m2:
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported fp8 dtype: {self.fp8_dtype}")


class FP8State:
    """
    FP8 storage wrapper for a *single* tensor (e.g., ADMM's z or u).

    - Storage: one FP8 tensor + FP32 scale(s) (tensorwise or rowwise).
    - Compute: Always upcast to FP32 on demand; caller performs math.
    - FSDP2-aware: (optional) synchronize scale across ranks (max).
    """

    def __init__(self, ref: torch.Tensor, cfg: Optional[FP8Config] = None):
        if cfg is None:
            cfg = FP8Config()
        self.cfg = cfg
        self.device = ref.device
        self.shape = tuple(ref.shape)
        self.ndim = ref.ndim
        self.fp8_dtype = cfg.torch_dtype()
        self._is_dtensor = isinstance(ref, DTensor)
        self._init_storage(ref)
        self._init_vmax()

    # ------------------------------
    # Constructors
    # ------------------------------
    @classmethod
    def from_like(cls, ref: torch.Tensor, **kwargs) -> "FP8State":
        """Create an empty FP8State with buffers shaped like `ref`."""
        return cls(ref, FP8Config(**kwargs))

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, x: torch.Tensor, **kwargs) -> "FP8State":
        """Create FP8State and immediately quantize `x` into storage."""
        st = cls(x, FP8Config(**kwargs))
        st.requant(x.to(torch.float32))
        return st

    # ------------------------------
    # Internal inits
    # ------------------------------
    def _init_storage(self, ref: torch.Tensor) -> None:
        if self._is_dtensor:
            self.data_fp8 = torch.zeros_like(ref, dtype=self.fp8_dtype)
        else:
            self.data_fp8 = torch.zeros(self.shape, dtype=self.fp8_dtype, device=self.device)
        if self.cfg.granularity == "tensorwise":
            self.scale = torch.ones((), dtype=torch.float32, device=self.device)
        elif self.cfg.granularity == "rowwise":
            assert self.ndim >= 2, "rowwise granularity requires tensor with at least 2 dims"
            rows = self.shape[-2]
            self.scale = torch.ones((rows,), dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Unsupported granularity: {self.cfg.granularity}")

    def _init_vmax(self) -> None:
        finfo = torch.finfo(self.fp8_dtype)
        self.vmax = torch.tensor(finfo.max, dtype=torch.float32, device=self.device)
        self.eps = torch.tensor(1e-12, dtype=torch.float32, device=self.device)

    # ------------------------------
    # Public API
    # ------------------------------
    @torch.no_grad()
    def dequant(self) -> torch.Tensor:
        """Return FP32 view of stored FP8 data using current scale(s)."""
        if self._is_dtensor:
            # FSDP2 path: dequantize on local shard, then reconstruct DTensor
            local_data = self.data_fp8.to_local().to(torch.float32)
            if self.cfg.granularity == "tensorwise":
                local_dequant = local_data * self.scale
            else:  # rowwise
                assert self.ndim >= 2
                shape = [1] * (self.ndim - 2) + [self.scale.numel(), 1]
                local_dequant = local_data * self.scale.view(*shape)
            return DTensor.from_local(
                local_dequant,
                device_mesh=self.data_fp8.device_mesh,
                placements=self.data_fp8.placements,
                run_check=False
            )
        else:
            # Standard path
            if self.cfg.granularity == "tensorwise":
                return self.data_fp8.to(torch.float32) * self.scale
            else:  # rowwise
                assert self.ndim >= 2
                shape = [1] * (self.ndim - 2) + [self.scale.numel(), 1]
                return self.data_fp8.to(torch.float32) * self.scale.view(*shape)

    @torch.no_grad()
    def get_fp32(self) -> torch.Tensor:
        """Alias for dequant() (kept for readability)."""
        return self.dequant()

    @torch.no_grad()
    def requant(self, x_new: torch.Tensor) -> None:
        """Quantize-and-store updated tensor using updated scale(s)."""
        self._update_scale_(x_new)
        if self.cfg.sync_scales and dist.is_available() and dist.is_initialized():
            self._sync_scale_()

        if self._is_dtensor:
            assert isinstance(x_new, DTensor), "Input must be a DTensor in FSDP2 mode"
            # FSDP2 path: quantize on local shard, then copy to local part of storage
            local_x = x_new.to_local()
            if self.cfg.granularity == "tensorwise":
                local_quant = (local_x / (self.scale + self.eps)).to(self.fp8_dtype)
            else:  # rowwise
                shape = [1] * (x_new.ndim - 2) + [self.scale.numel(), 1]
                local_quant = (local_x / (self.scale.view(*shape) + self.eps)).to(self.fp8_dtype)
            self.data_fp8.to_local().copy_(local_quant)
        else:
            # Standard path
            if self.cfg.granularity == "tensorwise":
                self.data_fp8.copy_((x_new / (self.scale + self.eps)).to(self.fp8_dtype))
            else:  # rowwise
                shape = [1] * (x_new.ndim - 2) + [self.scale.numel(), 1]
                self.data_fp8.copy_((x_new / (self.scale.view(*shape) + self.eps)).to(self.fp8_dtype))

    # ------------------------------
    # Scale updates & synchronization
    # ------------------------------
    @torch.no_grad()
    def _update_scale_(self, x: torch.Tensor) -> None:
        if self.cfg.scaling_type == ScalingType.NONE:
            return
        margin = self.cfg.safety_margin

        # For DTensors, compute scale on the local shard.
        # The subsequent _sync_scale_ will correctly synchronize it.
        x_local = x.to_local() if isinstance(x, DTensor) else x

        if self.cfg.granularity == "tensorwise":
            maxabs = x_local.abs().max().to(torch.float32)
            target = (maxabs / self.vmax) * margin
        else:
            *_, R, C = x_local.shape
            xr = x_local.reshape(-1, R, C).abs().amax(dim=(0, 2)).to(torch.float32)
            target = (xr / self.vmax) * margin
        new_scale = torch.clamp(target, min=1e-8)
        self.scale.copy_(new_scale)

    @torch.no_grad()
    def _sync_scale_(self) -> None:
        pg = self.cfg.process_group if self.cfg.process_group is not None else dist.group.WORLD
        dist.all_reduce(self.scale, op=dist.ReduceOp.MAX, group=pg)

    # ------------------------------
    # Checkpointing & stats
    # ------------------------------
    @torch.no_grad()
    def state_dict(self) -> dict:
        return {
            "cfg": {
                "fp8_dtype": self.cfg.fp8_dtype,
                "scaling_type": self.cfg.scaling_type.value,
                "granularity": self.cfg.granularity,
                "safety_margin": self.cfg.safety_margin,
                "sync_scales": self.cfg.sync_scales,
            },
            "data_fp8": self.data_fp8,
            "scale": self.scale,
        }

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        cfg = sd.get("cfg", {})
        self.cfg.fp8_dtype = cfg.get("fp8_dtype", self.cfg.fp8_dtype)
        self.fp8_dtype = FP8Config(fp8_dtype=self.cfg.fp8_dtype).torch_dtype()
        st = cfg.get("scaling_type", self.cfg.scaling_type.value)
        self.cfg.scaling_type = ScalingType(st)
        self.cfg.granularity = cfg.get("granularity", self.cfg.granularity)
        self.cfg.safety_margin = float(cfg.get("safety_margin", self.cfg.safety_margin))
        self.cfg.sync_scales = bool(cfg.get("sync_scales", self.cfg.sync_scales))

        self.data_fp8.copy_(sd["data_fp8"].to(self.fp8_dtype).to(self.device))
        self.scale.copy_(sd["scale"].to(torch.float32).to(self.device))

    @torch.no_grad()
    def saturation_ratio(self) -> float:
        x = self.dequant()
        if self.cfg.granularity == "tensorwise":
            sat = float((x.abs().amax() / (self.scale * self.vmax + self.eps)).clamp(max=1.0).item())
        else:
            *_, R, C = x.shape
            row = x.reshape(-1, R, C).abs().amax(dim=(0, 2))
            sat = float((row / (self.scale * self.vmax + self.eps)).clamp(max=1.0).mean().item())
        return sat