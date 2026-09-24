"""DualSystem joint self-attention architecture.

True joint attention (MMDiT / FastWAM MoT style): at every transformer
layer, the video and action backbones each compute Q/K/V independently
through ``pre_attn_at_layer``; :class:`DualSystemMoTDriver` concatenates the
two modalities, runs a single mixed self-attention, splits the result,
and feeds each slice back through ``post_attn_at_layer``.

The driver lives on ``self._mot_driver`` as a plain Python object with
no parameters; ``forward`` delegates the per-layer loop to it.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from openwam.model.action_backbone.separate_action_dit import ActionDiT
from openwam.model.value_backbone import ValueBackbone
from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.registry import register_architecture
from openwam.model.architectures.utils.common import resolve_bridge_layers
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO, MUTUAL
from openwam.model.compile_options import (
    compile_enabled,
    section_enabled,
    self_attn_compile_cfg,
    torch_compile_kwargs,
)

logger = logging.getLogger(__name__)


@register_architecture(
    "dual_system_self_attn",
    status="supported",
    note="DualSystem joint self-attention: MoT-style mixed attention at every layer.",
    framework="dual_system",
    variant="joint_self_attn",
)
class DualSystemSelfAttnArchitecture(BaseWAMArchitecture):
    """DualSystem with true joint self-attention (FastWAM MoT pattern)."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._mot_driver: DualSystemMoTDriver | None = None
        self._mot_driver_kwargs: dict = {}
        self._compiled_mot_run_joint_loop: Optional[Callable[..., Tuple[object, object]]] = None
        if cfg is None:
            return
        if self.video_backbone is not None:
            cfg = dict(cfg) if isinstance(cfg, dict) else {k: v for k, v in cfg.items()}
            cfg.setdefault("num_dit_layers", self.video_backbone.num_layers)
            cfg.setdefault("video_dim", self.video_backbone.dim)
            cfg.setdefault("num_heads", self.video_backbone.num_heads)
            cfg.setdefault("attn_head_dim", self.video_backbone.head_dim)
        bl = resolve_bridge_layers(cfg)
        video_dim = self._resolve_video_dim(cfg)
        # Default the action-side raw context width to the loaded backbone's
        # text_dim (Wan T5-XXL=4096, Cosmos-Predict2.5=1024); explicit cfg/CLI
        # still wins. Falls back to 4096 when the backbone doesn't expose it.
        _vb_text_dim = getattr(self.video_backbone, "text_dim", None)
        text_dim = int(self._cfg_get(cfg, "text_dim", _vb_text_dim or 4096))
        self._init_proprio_context(cfg, text_dim=text_dim)

        # FastWAM-Joint compat: action residual hidden_dim may differ from
        # video_dim. The MoT driver only requires num_heads / attn_head_dim
        # parity (validated at DualSystemMoTDriver.__init__).
        action_dim_hidden = int(cfg.get("dim", 1024))
        num_heads = int(cfg.get("num_heads", 24))
        attn_head_dim = int(cfg.get("attn_head_dim", video_dim // num_heads))

        action_ffn_dim = int(cfg.get("ffn_dim", 4 * action_dim_hidden))
        self.action_backbone = ActionDiT(
            action_dim=int(cfg.get("action_dim", 20)),
            dim=action_dim_hidden,
            ffn_dim=action_ffn_dim,
            num_heads=num_heads,
            num_layers=len(bl),
            video_dim=video_dim,
            bridge_layers=bl,
            variant="joint_self_attn",
            attn_head_dim=attn_head_dim,
            text_dim=text_dim,
            shift_action=cfg.get("shift_action"),
        )

        value_cfg = cfg.get("value_backbone", {})
        value_enabled = bool(value_cfg.get("enabled", False)) if value_cfg is not None else False
        if value_enabled:
            # Three-stream visibility has one explicit contract.  Reusing a
            # two-stream asymmetric mode would make the video/action submatrix
            # ambiguous, so reject it rather than silently changing semantics.
            configured_mode = str(cfg.get("attention_mask_mode", MUTUAL))
            if configured_mode != MUTUAL:
                raise ValueError(
                    "value_backbone.enabled=true requires attention_mask_mode='mutual'; "
                    "the three-stream mask itself blocks value->action."
                )
            value_dim = int(value_cfg.get("value_dim", ValueBackbone.DEFAULT_VALUE_DIM))
            if value_dim <= 0:
                raise ValueError(f"value_backbone.value_dim must be > 0, got {value_dim}")
            configured_value_hidden = value_cfg.get("dim")
            if configured_value_hidden is not None and int(configured_value_hidden) != action_dim_hidden:
                raise ValueError("value_backbone.dim must match ActionDiT dim to copy its attention projections")
            configured_value_ffn = value_cfg.get("ffn_dim")
            if configured_value_ffn is not None and int(configured_value_ffn) != action_ffn_dim:
                raise ValueError("value_backbone.ffn_dim must match ActionDiT ffn_dim")
            value_num_heads = value_cfg.get("num_heads")
            value_attn_head_dim = value_cfg.get("attn_head_dim")
            if value_num_heads is not None and int(value_num_heads) != num_heads:
                raise ValueError("value_backbone.num_heads must match ActionDiT num_heads")
            if value_attn_head_dim is not None and int(value_attn_head_dim) != attn_head_dim:
                raise ValueError("value_backbone.attn_head_dim must match ActionDiT attn_head_dim")
            self.value_backbone = ValueBackbone(
                value_dim=value_dim,
                dim=action_dim_hidden,
                ffn_dim=action_ffn_dim,
                num_heads=num_heads,
                num_layers=len(bl),
                video_dim=video_dim,
                bridge_layers=bl,
                attn_head_dim=attn_head_dim,
                text_dim=text_dim,
                freq_dim=int(value_cfg.get("freq_dim", 256)),
                rope_base_length=int(value_cfg.get("rope_base_length", value_cfg.get("max_value_len", 57))),
                eps=float(value_cfg.get("eps", 1e-6)),
                shift_value=value_cfg.get("shift_value", ValueBackbone.DEFAULT_SHIFT_VALUE),
            )

        # MoT driver is built once both backbones are available. The video
        # backbone is normally constructed in ``BaseWAMArchitecture._init_video_backbone``
        # (already done by ``super().__init__``), so we can build the driver
        # right here. Tests that swap in a mock video backbone after init call
        # :meth:`build_mot_driver` directly.
        self._mot_driver_kwargs = {
            "mot_checkpoint_mixed_attn": bool(cfg.get("mot_checkpoint_mixed_attn", True)),
            "attention_mask_mode": str(cfg.get("attention_mask_mode", ACTION_SEES_VIDEO)),
            "video_attention_mask_mode": str(cfg.get("video_attention_mask_mode", "first_frame_causal")),
        }
        if self.video_backbone is not None:
            self.build_mot_driver()

    def build_mot_driver(self) -> DualSystemMoTDriver:
        """Construct the :class:`DualSystemMoTDriver` from the current backbones.

        Re-callable; raises if either backbone is missing. Tests that swap in
        a mock video backbone after ``__init__`` should call this method to
        wire up the driver afterwards.
        """
        if self.video_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: video_backbone is not "
                "set. Construct the architecture with a video_backbone config or attach "
                "one before calling this method."
            )
        if self.action_backbone is None:
            raise RuntimeError(
                "DualSystemSelfAttnArchitecture.build_mot_driver: action_backbone is not "
                "set. Architecture must be built from a non-None cfg."
            )

        self._mot_driver = DualSystemMoTDriver(
            self.video_backbone,
            self.action_backbone,
            value_backbone=self.value_backbone,
            **self._mot_driver_kwargs,
        )
        return self._mot_driver

    @property
    def mot_driver(self) -> DualSystemMoTDriver | None:
        """The MoT joint-attention driver (None if the architecture wasn't fully built)."""
        return self._mot_driver

    def apply_compile_optimizations(self, compile_cfg) -> None:
        """Compile the eval-time MoT joint loop when requested."""

        super().apply_compile_optimizations(compile_cfg)
        self._compiled_mot_run_joint_loop = None
        if not compile_enabled(compile_cfg, default=False, strict=True):
            return

        section = self_attn_compile_cfg(compile_cfg)
        if not section_enabled(section, default=True):
            logger.info("dual self-attn MoT compile disabled by config; running eager.")
            return
        if self.video_backbone is None or self.action_backbone is None:
            logger.warning("dual self-attn MoT compile requested before backbones are ready; running eager.")
            return

        # The existing compiled function has a two-state signature.  Value
        # checkpoints deliberately use the eager three-stream loop until a
        # dedicated stable graph is compiled.
        if self.value_backbone is not None:
            logger.info("three-stream value MoT runs eager; skipping two-stream torch.compile path.")
            return

        driver = self._mot_driver or self.build_mot_driver()

        def _run_joint_loop(vstate, astate):
            return driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=False,
                use_gradient_checkpointing_offload=False,
            )

        kwargs = torch_compile_kwargs(section, default_mode="reduce-overhead")
        try:
            self._compiled_mot_run_joint_loop = torch.compile(_run_joint_loop, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive setup fallback
            self._compiled_mot_run_joint_loop = None
            logger.warning("dual self-attn MoT torch.compile setup failed; running eager: %s", exc)
            return
        logger.info("Enabled dual self-attn MoT compile with torch.compile kwargs=%s", kwargs)

    def forward(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        *,
        noisy_values: Optional[Tensor] = None,
        value_timestep: Optional[Tensor] = None,
        proprio: Optional[Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **pipeline_inputs,
    ) -> Tuple[Tensor, Optional[Tensor]] | Tuple[Tensor, Optional[Tensor], Tensor]:
        vb = self.video_backbone
        ab = self.action_backbone
        if vb is None:
            raise RuntimeError(
                "video_backbone is None — pass pipe= to build_architecture or "
                "architecture.__init__ to enable forward()."
            )

        pipeline_inputs = self._append_proprio_context_token(dict(pipeline_inputs), proprio)
        action_context = pipeline_inputs.get("context")
        action_context_mask = pipeline_inputs.get("context_mask")
        if action_context is not None and action_context_mask is None and pipeline_inputs.get("seq_lens") is not None:
            seq_lens = pipeline_inputs["seq_lens"].to(device=action_context.device)
            positions = torch.arange(action_context.shape[1], device=action_context.device)
            action_context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
        # Opt every Wan backbone into 4D + clean-prefix-aligned t_mod (mirrors
        # TI2V's native ``seperated_timestep + fuse_vae_embedding_in_latents``
        # path; TI2V itself fires that path first so these kwargs are inert for
        # it). VACE and I2V do NOT emit ``first_frame_latents`` (VACE routes
        # its first-frame condition through ``vace_context``; I2V uses the
        # ``y`` channel), so for them ``zero_clean_prefix_t_mod`` is
        # structurally inert — kept on only so the joint MoT driver gets the
        # 4D ``t_mod`` it needs. ``setdefault`` so explicit callers can still
        # pass False.
        pipeline_inputs.setdefault("force_per_token_t_mod", True)
        pipeline_inputs.setdefault("zero_clean_prefix_t_mod", True)
        vstate = vb.prepare(
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            **pipeline_inputs,
        )

        if noisy_values is not None and self.value_backbone is None:
            raise RuntimeError("noisy_values were provided but value_backbone is not enabled")
        if noisy_values is not None and value_timestep is None:
            raise ValueError("value_timestep is required when noisy_values are provided")

        if noisy_values is None and (noisy_actions is None or ab is None):
            for block_id in range(vb.num_layers):
                vstate = vb.run_block(block_id, vstate)
            return vb.finalize(vstate), None

        driver = self._mot_driver
        if driver is None:
            driver = self.build_mot_driver()

        astate = None
        if noisy_actions is not None and ab is not None:
            astate = ab.prepare_state(
                noisy_actions,
                action_timestep,
                context=action_context,
                context_mask=action_context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        qstate = None
        if noisy_values is not None:
            qstate = self.value_backbone.prepare_value_state(
                noisy_values,
                value_timestep,
                context=action_context,
                context_mask=action_context_mask,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        if qstate is not None:
            vstate, qstate, astate = driver.run_joint_value_loop(
                vstate,
                qstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            return (
                vb.finalize(vstate),
                ab.extract_prediction(astate) if astate is not None else None,
                self.value_backbone.extract_value_prediction(qstate),
            )

        assert astate is not None
        compiled_loop = getattr(self, "_compiled_mot_run_joint_loop", None)
        if compiled_loop is not None and not use_gradient_checkpointing and not use_gradient_checkpointing_offload:
            try:
                vstate, astate = compiled_loop(vstate, astate)
            except Exception as exc:
                self._compiled_mot_run_joint_loop = None
                logger.warning("dual self-attn compiled MoT loop failed; falling back to eager: %s", exc)
                vstate, astate = driver.run_joint_loop(
                    vstate,
                    astate,
                    use_gradient_checkpointing=use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                )
        else:
            vstate, astate = driver.run_joint_loop(
                vstate,
                astate,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
        return vb.finalize(vstate), ab.extract_prediction(astate)

    def forward_joint(
        self,
        noisy_actions: Optional[Tensor],
        action_timestep: Optional[Tensor],
        noisy_values: Tensor,
        value_timestep: Tensor,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Value-enabled forward with an explicit stable three-output contract."""
        output = self.forward(
            noisy_actions,
            action_timestep,
            noisy_values=noisy_values,
            value_timestep=value_timestep,
            **kwargs,
        )
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("value-enabled joint forward did not return video/action/value predictions")
        return output


__all__ = ["DualSystemSelfAttnArchitecture"]
