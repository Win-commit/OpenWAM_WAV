"""Flow-matching scheduler owned by the value stream."""

from openwam.model.action_backbone.scheduler import ActionScheduler


class ValueScheduler(ActionScheduler):
    """Value-stream scheduler with independent state from the action stream.

    The underlying flow-matching equation is intentionally the same as the
    action scheduler, but this is a distinct object: training samples its
    timestep/noise independently and deployment can place it on its own
    shifted grid. Keep the requested WAV default visible here instead of
    relying on :meth:`ActionScheduler.set_timesteps`' incidental default.
    """

    DEFAULT_SHIFT = 5.0


__all__ = ["ValueScheduler"]
