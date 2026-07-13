import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from absl import logging
import wandb
import sys
import inspect
import functools
from .utils import FP8State
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import shutil

import numpy as np
from packaging import version
from transformers import Trainer, TrainingArguments
from transformers.optimization import get_scheduler
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    ExportableState,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from accelerate.data_loader import skip_first_batches
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import deepspeed_init, is_deepspeed_available
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.trainer_pt_utils import find_batch_size, nested_detach, nested_numpify, distributed_concat, get_model_param_count, EvalLoopContainer, IterableDatasetShard
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, TrainOutput, denumpify_detensorize, has_length, speed_metrics
from transformers.utils import (
    is_apex_available, 
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_torch_xla_available,
    is_accelerate_available
)
from accelerate.utils import InitProcessGroupKwargs

from .optimizers import get_admm_optimizer
from .scheduler import PenaltyScheduler
from .utils import find_layers, projection

deepspeed_init = None
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

pl = None
nested_xla_mesh_reduce = None
if is_torch_tpu_available():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        from transformers.trainer_pt_utils import nested_xla_mesh_reduce
    except ImportError:
        pl = None
        nested_xla_mesh_reduce = None

distributed_concat = None
try:
    from transformers.trainer_pt_utils import distributed_concat
except ImportError:
    distributed_concat = None

if is_apex_available():
    from apex import amp

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration
# =====================================================================================

logger = logging

TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"



class ADMMTrainer(Trainer):
    """
    Trainer using the external ADMM optimizer from lib.utils.optimizers.
    Supports standard Causal LM loss and Reconstruction Error Minimization (REM) loss.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
            
        super().__init__(*args, **kwargs)
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        self.is_tp_enabled = getattr(self.accelerator.state, "torch_tp_plugin", None) is not None

    def allocate_nonuniform_sparsity(self):
        """
        Calculates and allocates non-uniform sparsity based on a config file.
        The formula `sparsity = base + k * (1M / layer_weights)` is applied per layer.
        This method is FSDP-compatible as it relies on parameter names.
        """
        config_file = getattr(self.args, 'nonuniform_sparsity_config_file', None)
        if not config_file:
            logger.warning("`nonuniform_sparsity_config_file` not provided. Skipping non-uniform sparsity allocation.")
            return

        if not os.path.exists(config_file):
            logger.error(f"Sparsity config file not found at {config_file}. Aborting non-uniform allocation.")
            return

        if self.is_world_process_zero():
            logger.info(f"Allocating non-uniform sparsity from config file: {config_file}")

        layer_levels = {}
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    if not line.strip() or ':' not in line: continue
                    name, level = line.strip().split(':', 1)
                    layer_levels[name.strip()] = int(level.strip())
        except Exception as e:
            logger.error(f"Error parsing sparsity config file {config_file}: {e}")
            return

        unwrapped_model = unwrap_model(self.model)
        base_sparsity = self.args.sparsity_ratio
        layer_sparsities = {}
        
        normalized_param_map = {}
        for name, param in unwrapped_model.named_parameters():
            normalized_name = name.replace('._checkpoint_wrapped_module', '')
            normalized_param_map[normalized_name] = param

        for module_name, level in layer_levels.items():
            param_name = f"{module_name}.weight"
            if param_name in normalized_param_map:
                param = normalized_param_map[param_name]
                layer_weights = math.prod(param.shape)
                if layer_weights > 0:
                    sparsity = base_sparsity + level * (1e6 / layer_weights)
                    layer_sparsities[module_name] = min(0.99, max(0.0, sparsity))
                else:
                    layer_sparsities[module_name] = base_sparsity
            else:
                 if self.is_world_process_zero():
                    logger.warning(f"Module '{module_name}' from config not found in model parameters. Skipping.")

        unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
        
        if self.is_world_process_zero():
            logger.info("Applying non-uniform sparsities to optimizer state...")

        param_to_group = {id(p): g for g in unwrapped_optimizer.param_groups for p in g['params']}

        updated_params = 0
        for name, param in unwrapped_model.named_parameters():
            normalized_name = name.replace('._checkpoint_wrapped_module', '')
            if normalized_name in self.admm_param_names:
                module_name = normalized_name.replace('.weight', '')
                new_sparsity = layer_sparsities.get(module_name, base_sparsity)
                
                param_group = param_to_group.get(id(param))
                if param_group is None:
                    if self.is_world_process_zero():
                        logger.warning(f"Could not find optimizer parameter group for {name}. Skipping state update.")
                    continue
                                
                unwrapped_optimizer._lazy_init_admm_state(param, param_group)
                
                if unwrapped_optimizer.state[param]['sparsity'] != new_sparsity:
                    unwrapped_optimizer.state[param]['sparsity'] = new_sparsity
                    updated_params += 1

        if self.is_world_process_zero():
            logger.info(f"Finished applying non-uniform sparsity to {updated_params} parameter groups.")
            if self.args.wandb:
                sparsity_table = wandb.Table(columns=["Layer Name", "Level", "Allocated Sparsity"])
                for name, sparsity in sorted(layer_sparsities.items()):
                    level = layer_levels.get(name, 'N/A')
                    sparsity_table.add_data(name, level, sparsity)
                    logger.info(f"  - Layer {name}: Level {level}, Sparsity {sparsity:.4f}")
                wandb.log({"non_uniform_sparsity_allocation": sparsity_table}, step=self.state.global_step)

    def create_accelerator_and_postprocess(self):
        grad_acc_kwargs = {}
        if is_accelerate_available("0.28.0") and self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        if "num_steps" in grad_acc_kwargs and self.args.gradient_accumulation_steps > 1:
            raise ValueError(
                "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
            )
        elif "num_steps" not in grad_acc_kwargs:
            grad_acc_kwargs["num_steps"] = self.args.gradient_accumulation_steps

        grad_acc_kwargs["sync_with_dataloader"] = False

        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_config = self.args.accelerator_config.to_dict()

        if is_accelerate_available("0.28.0"):
            dataloader_config = DataLoaderConfiguration(
                split_batches=accelerator_config.pop("split_batches"),
                dispatch_batches=accelerator_config.pop("dispatch_batches"),
                even_batches=accelerator_config.pop("even_batches"),
                use_seedable_sampler=accelerator_config.pop("use_seedable_sampler"),
            )
        non_blocking = accelerator_config.pop("non_blocking")
        if not is_accelerate_available("0.30.0"):
            if non_blocking:
                raise ImportError(
                    "`non_blocking` is only supported in accelerate v0.30.0 and above. Please upgrade accelerate to use this feature."
                )
        else:
            if non_blocking and not self.args.dataloader_pin_memory:
                logger.warning(
                    "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
                )
            dataloader_config.non_blocking = non_blocking

        accelerator_config.pop("gradient_accumulation_kwargs")

        args = {
            "deepspeed_plugin": self.args.deepspeed_plugin,
            "gradient_accumulation_plugin": gradient_accumulation_plugin,
        }
        if is_accelerate_available("0.28.0"):
            args["dataloader_config"] = dataloader_config
        else:
            args.update(accelerator_config)

        self.accelerator = Accelerator(**args)
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters.keys():
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get(
                "limit_all_gathers", fsdp_plugin.limit_all_gathers
            )
            fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get(
                "activation_checkpointing", fsdp_plugin.activation_checkpointing
            )
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            self.propagate_args_to_deepspeed()

        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )

    def create_optimizer(self):
        """
        Overrides the base method to create the ADMM optimizer.
        It now stores the names of ADMM parameters in `self.admm_param_names`
        to ensure robust identification in distributed environments.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        
        if self.optimizer is None:
            layers_to_prune = [nn.Linear]
            admm_layers = find_layers(opt_model, layers=layers_to_prune)
            
            self.admm_param_names = set()
            for name in admm_layers:
                if 'lm_head' in name:
                    continue
                if hasattr(admm_layers[name], 'weight') and admm_layers[name].weight.requires_grad:
                    full_param_name = f'{name}.weight'
                    self.admm_param_names.add(full_param_name)
            admm_param_list = []
            other_param_list = []
            for name,param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue

                if name in self.admm_param_names:
                    admm_param_list.append(param)
                else:
                    other_param_list.append(param)
            
            admm_param_group = {
                'params': admm_param_list,
                'name': 'weights',
                'admm': True,
            }
            other_param_group = {
                'params': other_param_list,
                'name': 'other_params',
                'admm': False,
            }
            param_groups = []
            if admm_param_group['params']: # Only add if there are params in this group
                param_groups.append(admm_param_group)
            if other_param_group['params']: # Only add if there are params in this group
                param_groups.append(other_param_group)

            if not any(pg.get('admm', False) for pg in param_groups) and self.args.rank == 0:
                logger.warning(f'No parameters were marked for ADMM. Pruning might not be effective.')

            if self.is_world_process_zero():
                logger.info(f'Found {len(admm_param_list)} / {len([p for p in opt_model.parameters()])} parameters for ADMM.')

            # Base optimizer for ADMM
            if self.args.base_optimizer_type in ['adam','adamw','adam8bit','adam4bit']:
                base_optimizer_kwargs = {
                    'betas': (self.args.admm_beta1, self.args.admm_beta2),
                    'weight_decay': self.args.weight_decay
                }
            else: ## sgd
                base_optimizer_kwargs = {}

            ADMM = get_admm_optimizer(self.args.base_optimizer_type)
            if self.is_world_process_zero():
                logger.info(f"Using base optimizer: {ADMM.__bases__[0].__name__}")
            self.optimizer = ADMM(
                param_groups,
                projection_fn=projection,
                lmda=self.args.admm_lmda,
                init_lmda=self.args.admm_init_lmda,
                final_lmda=self.args.admm_final_lmda,
                sparsity=self.args.sparsity_ratio,
                interval=self.args.admm_interval,
                lmda_schedule_mode=self.args.admm_lmda_schedule_mode,
                init_lambda_from_inv_resid=self.args.admm_init_lambda_from_inv_resid,
                total_steps=self.args.max_steps,
                lr=float(self.args.learning_rate),
                prune_n=self.args.prune_n,
                prune_m=self.args.prune_m,
                projection_mode=self.args.admm_projection_mode,
                projection_bias_correction=self.args.admm_projection_bias_correction,
                dual_dtype=self.args.admm_dual_dtype,
                split_dtype=self.args.admm_split_dtype,
                accelerator=self.accelerator,
                **base_optimizer_kwargs,
            )
            if self.is_world_process_zero():
                logger.info(f"Created {self.optimizer.__class__.__name__} optimizer targeting {len(admm_param_list)} Linear weight tensors.")
        return self.optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Override to create the ADMM optimizer and ensure the scheduler targets the base optimizer.
        If adaptive sparsity is enabled, it's calculated once before training starts.
        """
        self.create_optimizer()
        optimizer = self.optimizer

        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
        if self.is_world_process_zero():
            logger.info("ADMM optimizer and scheduler created (scheduler targets base optimizer).")

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        creates a learning rate scheduler, penalty scheduler, and sparsity scheduler if needed.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step.
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        prepared_inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        original_labels = None
        if has_labels:
            original_labels = nested_detach(tuple(prepared_inputs.get(name) for name in self.label_names))
            if len(original_labels) == 1:
                original_labels = original_labels[0]

        loss = None
        logits = None
        labels_for_output = original_labels 

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model,prepared_inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, prepared_inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**prepared_inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]
        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, original_labels)

    @torch.no_grad()
    def _evaluate_sparse_model(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval_sparse",
    ) -> EvalLoopOutput:
        """
        Evaluates the sparse model by temporarily applying final_projection,
        then restoring the dense weights. Handles FSDP2 sharding and stores
        dense weights on CPU for memory efficiency.
        """
        unwrapped_model = unwrap_model(self.model)
        unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
        
        dense_state = {}
        admm_param_ids = {id(p) for group in unwrapped_optimizer.param_groups 
                          if group.get('admm', False) for p in group['params']}
        
        if self.is_world_process_zero():
            logger.info("Saving dense weights to CPU for sparse evaluation...")
        
        for name, param in unwrapped_model.named_parameters():
            if id(param) in admm_param_ids:
                if self._is_dtensor(param):
                    dense_state[name] = param.to_local().detach().cpu().clone()
                else:
                    dense_state[name] = param.detach().cpu().clone()
        
        if self.is_world_process_zero():
            total_size_mb = sum(v.numel() * v.element_size() / (1024**2) for v in dense_state.values())
            logger.info(f"Dense weights saved on CPU: {total_size_mb:.2f} MB")
        
        try:
            unwrapped_optimizer.final_projection()

            if self.is_world_process_zero():
                logger.info(f"Sparse projection applied for {description}")
            
            if self.args.world_size > 1 and dist.is_initialized():
                dist.barrier()
            
            self._skip_residual_calculation = True
            
            sparse_output = self._evaluation_loop_impl(
                dataloader=dataloader,
                description=f"{description} (sparse)",
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
            
            self._skip_residual_calculation = False
            
            return sparse_output
            
        finally:
            if self.is_world_process_zero():
                logger.info("Restoring dense weights from CPU...")
            
            for name, param in unwrapped_model.named_parameters():
                if name in dense_state:
                    cpu_tensor = dense_state[name]
                    if self._is_dtensor(param):
                        local_param = param.to_local()
                        local_param.copy_(cpu_tensor.to(local_param.device))
                    else:
                        param.data.copy_(cpu_tensor.to(param.device))
            
            if self.args.world_size > 1 and dist.is_initialized():
                dist.barrier()
            
            dense_state.clear()
            torch.cuda.empty_cache()
            
            if self.is_world_process_zero():
                logger.info(f"Dense weights restored from CPU after sparse evaluation")

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Now includes sparse model evaluation alongside dense model evaluation.

        Works both with or without labels.
        """
        dense_output = self._evaluation_loop_impl(
            dataloader=dataloader,
            description=description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        
        sparse_output = None
        unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer

        if hasattr(unwrapped_optimizer, 'final_projection'):
            sparse_output = self._evaluate_sparse_model(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=f"{metric_key_prefix}_sparse",
            )
        
        if sparse_output is not None:
            dense_output.metrics.update(sparse_output.metrics)
        
        return dense_output

    def _evaluation_loop_impl(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Internal evaluation loop implementation (original evaluation_loop logic).
        Renamed to avoid conflict with the new evaluation_loop wrapper.
        Includes ADMM metrics (primal/dual residuals) calculation.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        if self.is_world_process_zero():
            logger.info(f"\n***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

        model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size

            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            if losses is not None:
                gathered_losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(gathered_losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function((labels))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    if args.include_inputs_for_metrics:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
                            compute_result=is_last_step,
                        )
                    else:
                        metrics = self.compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels),
                            compute_result=is_last_step,
                        )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        elif metrics is None:
            metrics = {}
        
        if not getattr(self, '_skip_residual_calculation', False):
            primal_residual = self.calculate_primal_residual(
                metric_key_prefix=metric_key_prefix
            )
            dual_residual = self.calculate_dual_residual(
                metric_key_prefix=metric_key_prefix
            )
            if self.is_world_process_zero():
                metrics.update(primal_residual)
                metrics.update(dual_residual)

        metrics = denumpify_detensorize(metrics)
        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def _is_dtensor(self, x):
        """Check if tensor is a DTensor (distributed tensor from FSDP)."""
        return hasattr(x, "to_local")

    def _loc(self, x):
        """Return local shard if DTensor, otherwise the tensor itself."""
        return x.to_local() if self._is_dtensor(x) else x
    
    def calculate_primal_residual(
        self,
        metric_key_prefix: str = "eval",
        eps: float = 1e-12,
    ) -> Dict[str, float]:
        """
        Computes primal and relative residuals in a distributed-friendly way.
        Iterates through per-parameter optimizer states to get current w and z.
        """
        with torch.no_grad():
            unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
            
            local_primal_residual_sq = torch.tensor(0.0, device=self.args.device)
            local_weight_norm_sq = torch.tensor(0.0, device=self.args.device)
            
            for group in unwrapped_optimizer.param_groups:
                if not group.get('admm', False): continue
                for param in group['params']:
                    if param in unwrapped_optimizer.state:
                        state = unwrapped_optimizer.state[param]
                        w = self._loc(param)
                        z = self._loc(state['split'].get_fp32() if isinstance(state['split'],FP8State) else state['split'])

                        local_primal_residual_sq += torch.sum((w - z) ** 2)
                        local_weight_norm_sq += torch.sum(w ** 2)
            
            if self.args.world_size > 1 and dist.is_initialized():
                dist.all_reduce(local_primal_residual_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_weight_norm_sq, op=dist.ReduceOp.SUM)

            primal_residual = torch.sqrt(local_primal_residual_sq).item()
            relative_residual = primal_residual / (torch.sqrt(local_weight_norm_sq).item() + eps)

        return {
            f"{metric_key_prefix}_primal_residual": primal_residual,
            f"{metric_key_prefix}_relative_residual": relative_residual,
        }

    def calculate_dual_residual(
        self,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Computes the dual residual in a distributed-friendly way using all_reduce.
        Iterates through per-parameter optimizer states to get current w, u, z, and lmda.
        """
        with torch.no_grad():
            unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
            
            local_dual_residual_sq = torch.tensor(0.0, device=self.args.device)
            local_scaled_dual_residual_sq = torch.tensor(0.0, device=self.args.device) # New tensor for scaled sum

            for group in unwrapped_optimizer.param_groups:
                if not group.get('admm', False): continue
                for param in group['params']:
                    if param in unwrapped_optimizer.state:
                        state = unwrapped_optimizer.state[param]
                        w = self._loc(param)
                        u = self._loc(state['dual'].get_fp32() if isinstance(state['dual'],FP8State) else state['dual'])
                        z = self._loc(state['split'].get_fp32() if isinstance(state['split'],FP8State) else state['split'])
                        p_sparsity = state.get('sparsity', unwrapped_optimizer.sparsity)
                        current_param_lmda = state['lmda'] # Get the per-parameter lmda
                        
                        z_new = projection([w + u], sparsity=p_sparsity, prune_n=unwrapped_optimizer.prune_n, prune_m=unwrapped_optimizer.prune_m, comparison_group="layer")[0]
                        
                        param_dual_residual_sq = torch.sum((z_new - z) ** 2)
                        local_dual_residual_sq += param_dual_residual_sq

                        local_scaled_dual_residual_sq += (current_param_lmda ** 2) * param_dual_residual_sq

            if self.args.world_size > 1 and dist.is_initialized():
                dist.all_reduce(local_dual_residual_sq, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_scaled_dual_residual_sq, op=dist.ReduceOp.SUM) # All-reduce the scaled sum

            dual_residual = torch.sqrt(local_dual_residual_sq).item()
            scaled_dual_residual = torch.sqrt(local_scaled_dual_residual_sq).item() # Take sqrt of the summed squares

        return {
            f"{metric_key_prefix}_dual_residual": dual_residual,
            f"{metric_key_prefix}_scaled_dual_residual": scaled_dual_residual,
        }
    
    # --- Training Methods ---

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes NTP loss.
        """
        if 'batch_idx' in inputs:
            inputs.pop("batch_idx")
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def get_tp_size(self) -> int:
        """Get the tensor parallel size from either the model or DeepSpeed config."""

        
        if (model_tp := getattr(self.model, "_tp_size", None)) is not None:
            return model_tp

        
        if self.is_deepspeed_enabled and (deepspeed_config := getattr(self.args, "hf_deepspeed_config", None)):
            return deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)

        
        return 1

    def get_total_train_batch_size(self, args) -> int:
        """Calculates total batch size (micro_batch * grad_accum * dp_world_size).

        Note: Only considers DP and TP (dp_world_size = world_size // tp_size)."""
        dp_world_size = args.world_size // self.get_tp_size()
        return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                
                if self.is_deepspeed_enabled:
                
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled or self.is_tp_enabled
        )
        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)
        
        model = self._wrap_model(self.model_wrapped)

        
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    if delay_optimizer_creation:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        
        if model is not self.model:
            self.model_wrapped = model

        
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        
        if self.args.admm_nonuniform_sparsity:
            self.allocate_nonuniform_sparsity()

        if self.is_world_process_zero():
            logger.info("***** Running ADMM Training *****")
            logger.info(f"  Num examples = {num_examples}")
            if args.max_steps > 0:
                logger.info(f"  ADMM Steps = {args.max_steps}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f'  Gradient Checkpointing = {args.gradient_checkpointing}')
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  ADMM Optimizer Args: lmda={args.admm_lmda}, interval={args.admm_interval}, sparsity={args.sparsity_ratio}, weight_decay={args.weight_decay}")
            logger.info(f'  ADMM Penalty Scheduler = {args.admm_lmda_schedule_mode}')

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        tr_loss = torch.tensor(0.0).to(args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )
                ### gradient normalization / clipping
                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            # This will handle FSDP, DDP, etc. correctly.
                            # It returns the total norm of the parameters before clipping.
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        # For DeepSpeed, get_global_grad_norm is the way to get the norm
                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm
                                    
                    unwrapped_optimizer = self.optimizer.optimizer if hasattr(self.optimizer, 'optimizer') else self.optimizer
                    is_dual_update_step = (unwrapped_optimizer.current_step + 1) % unwrapped_optimizer.interval == 0

                    if is_dual_update_step:
                        mask_metrics = unwrapped_optimizer.get_mask_metrics()
                        lmda_stats = unwrapped_optimizer.get_lmda_stats()
                        if self.is_world_process_zero():
                            logger.info(f'ADMM Step {self.state.global_step}: Mask Hamming (step): {mask_metrics["step_hamming"]:.4f}, '
                                        f'Mask Hamming (initial): {mask_metrics["initial_hamming"]:.4f}, Mask IoU (step): {mask_metrics["step_iou"]:.4f}, Mask IoU (initial): {mask_metrics["initial_iou"]:.4f}, Avg lmda: {lmda_stats["avg_lmda"]:.4e}')
                            wandb_metrics = {
                                'ADMM_mask_hamming_step': mask_metrics["step_hamming"],
                                'ADMM_mask_hamming_initial': mask_metrics["initial_hamming"],
                                'ADMM_mask_iou_step': mask_metrics["step_iou"],
                                'ADMM_mask_iou_initial': mask_metrics["initial_iou"],
                                'ADMM_avg_lmda': lmda_stats["avg_lmda"],
                                'ADMM_min_lmda': lmda_stats["min_lmda"],
                                'ADMM_max_lmda': lmda_stats["max_lmda"]
                            }  
                            self.log(wandb_metrics)

                    self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
    

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            # --- End Batch Loop ---
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm=None, model=model, trial=trial, epoch=epoch, ignore_keys_for_eval=ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
        # --- End Epoch Loop ---

        if self.is_world_process_zero():
            logger.info("\n\nADMM Training completed.\n\n")

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        unwrapped_optimizer.final_projection()

        if hasattr(unwrapped_optimizer, "save_flip_statistics"):
            unwrapped_optimizer.save_flip_statistics("flip_stats.pt")

        if self.is_world_process_zero():
            logger.info('final projection finished')

        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()


        # Final loss calculation and metrics

        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        # Add the primary training loss with a specific key
        metrics["train_cross_entropy_loss"] = train_loss

        self.is_in_train = False
        self._memory_tracker.stop_and_update_metrics(metrics)
        if self.is_world_process_zero():
            self.log(metrics) # Log final training metrics
        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        self._finish_current_push()

        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader, total_train_batch_size: int
    ):
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `epoch_based`
        - `len_dataloader`
        - `max_steps`
        """
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        )