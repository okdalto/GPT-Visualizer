from dataclasses import dataclass, field


@dataclass
class StageTimeline:
    stage_name: str
    start_time: float
    appear_duration: float
    compute_duration: float
    settle_duration: float
    # Per-group segment boundaries within compute t=[0,1].
    # e.g. [(0.0, 0.33), (0.33, 0.67), (0.67, 1.0)] for 3 equal groups.
    group_segments: list[tuple[float, float]] = field(default_factory=list)

    @property
    def end_time(self) -> float:
        return self.start_time + self.total_duration

    @property
    def total_duration(self) -> float:
        return self.appear_duration + self.compute_duration + self.settle_duration

    def get_phase(self, global_time: float) -> tuple:
        """Returns (phase_name, local_t) where phase_name is 'appear'/'compute'/'settle'
        and local_t is 0..1 within that phase. Returns ('inactive', 0) if outside."""
        if global_time < self.start_time:
            return ('inactive', 0.0)
        if global_time > self.end_time:
            return ('done', 1.0)

        local = global_time - self.start_time

        if local < self.appear_duration:
            return ('appear', local / max(self.appear_duration, 1e-6))
        local -= self.appear_duration

        if local < self.compute_duration:
            return ('compute', local / max(self.compute_duration, 1e-6))
        local -= self.compute_duration

        return ('settle', local / max(self.settle_duration, 1e-6))


# ── Per-stage, per-phase speed config ───────────────────────────────
# appear / settle: single float speed multiplier.
# compute: list of per-group speed multipliers (one per phase_group).
#          Higher = faster (2.0 → half duration), 0.5 → double.
#          Omit or use empty list for stages with no compute.
STAGE_SPEED = {
    'char_display':       {'appear': 2.0, 'settle': 1.5, 'compute': [2.5]},
    'input':              {'appear': 1.0, 'settle': 0.5},
    'qkv_projection':     {'appear': 1.0, 'settle': 1.0, 'compute': [
                               0.6,           # QKV matmul (parallel)
                           ]},
    'multi_head_attn':    {'appear': 1.0, 'settle': 1.0, 'compute': [
                               1.0,           # Q × K^T
                               1.0,           # softmax
                               0.7,           # Weights × V
                           ]},
    'concat_output_proj': {'appear': 1.0, 'settle': 1.0, 'compute': [
                               0.6,           # Concat × W_O
                           ]},
    'residual_ln1':       {'appear': 1.0, 'settle': 1.0, 'compute': [
                               1.0,           # Add
                               1.0,           # LayerNorm
                           ]},
    'ffn':                {'appear': 1.0, 'settle': 1.0, 'compute': [
                               0.25,           # × W1
                               1.0,           # ReLU
                               0.25,           # × W2
                           ]},
    'residual_ln2':       {'appear': 1.0, 'settle': 1.3, 'compute': [
                               1.0,           # Add
                               1.0,           # LayerNorm
                           ]},
    'output':             {'appear': 2.0, 'settle': 2.0},
    'block_1':            {'appear': 1.5, 'settle': 1.5, 'compute': [1.5]},
    'block_2':            {'appear': 1.5, 'settle': 1.5, 'compute': [1.5]},
    'block_3':            {'appear': 1.5, 'settle': 1.5, 'compute': [1.5]},
    'block_4':            {'appear': 1.5, 'settle': 1.5, 'compute': [1.5]},
    'output_projection':  {'appear': 1.0, 'settle': 1.0, 'compute': [0.4]},
    'token_probs':        {'appear': 1.0, 'settle': 2.0, 'compute': [1.0, 2.0, 2.0]},
}

# Speed overrides for logits_only abbreviated mode (arrow-key skip feel)
LOGITS_ONLY_SPEED = {
    'block_1':            {'appear': 6.0, 'settle': 3.0},
    'block_2':            {'appear': 6.0, 'settle': 3.0},
    'block_3':            {'appear': 6.0, 'settle': 3.0},
    'block_4':            {'appear': 6.0, 'settle': 3.0},
    'output_projection':  {'appear': 6.0, 'settle': 3.0},
    'token_probs':        {'appear': 6.0},
}


def _compute_group_segments(group_speeds: list[float]) -> tuple[float, list[tuple[float, float]]]:
    """Convert per-group speed multipliers into segment boundaries.

    Returns (total_weight, segments) where segments are (start_t, end_t)
    pairs in the 0-1 range. Each group's duration weight is 1/speed.
    """
    if not group_speeds:
        return 0.0, []
    weights = [1.0 / max(s, 0.01) for s in group_speeds]
    total = sum(weights)
    segments = []
    accum = 0.0
    for w in weights:
        start = accum / total
        accum += w
        end = accum / total
        segments.append((start, end))
    return total, segments


class AnimationTimeline:
    def __init__(self, has_lm_head: bool = False, has_char_display: bool = False,
                 final_display: bool = False, logits_only: bool = False):
        self.stages: list[StageTimeline] = []
        self.total_duration: float = 0.0
        self.return_duration: float = 1.3
        self.current_time: float = 0.0
        self.playing: bool = True
        self.speed: float = 1.0
        self.loop: bool = True
        self.completed: bool = False

        self._build_default_timeline(has_lm_head, has_char_display,
                                     final_display, logits_only)
        if logits_only:
            self.speed = 3.0

    def _build_default_timeline(self, has_lm_head: bool = False,
                                has_char_display: bool = False,
                                final_display: bool = False,
                                logits_only: bool = False):
        t = 0.0

        # Uniform base timing
        APPEAR = 2.0
        COMPUTE_PER_GROUP = 2.5
        SETTLE = 1.0

        # (stage_name, num_phase_groups)
        if final_display:
            configs = [('char_display', 1)]
        elif logits_only:
            configs = []
            if has_char_display:
                configs.append(('char_display', 1))
            configs.extend([
                ('input',             0),   # 0 duration (chars fly during char_display)
                ('block_1',           0),   # static flash (no compute animation)
                ('block_2',           0),
                ('block_3',           0),
                ('block_4',           0),
                ('output_projection', 0),   # logits matrix flash
                ('token_probs',       3),
            ])
        else:
            configs = []
            if has_char_display:
                configs.append(('char_display', 1))
            configs.extend([
                ('input',              0),
                ('qkv_projection',     1),
                ('multi_head_attn',    3),
                ('concat_output_proj', 1),
                ('residual_ln1',       2),
                ('ffn',                3),
                ('residual_ln2',       2),
                ('output',             0),
                ('block_2',            1),
                ('block_3',            1),
                ('block_4',            1),
            ])
            if has_lm_head:
                configs.extend([
                    ('output_projection', 1),
                    ('token_probs',       3),
                ])
        for name, groups in configs:
            if logits_only and name in LOGITS_ONLY_SPEED:
                cfg = LOGITS_ONLY_SPEED[name]
            else:
                cfg = STAGE_SPEED.get(name, {})
            appear = APPEAR / cfg.get('appear', 1.0)
            settle = SETTLE / cfg.get('settle', 1.0)

            # Input matrix appears during char_display compute; skip its own timing
            if name == 'input' and has_char_display:
                appear = 0.0
                settle = 0.0

            # Per-group compute speeds
            group_speeds = cfg.get('compute', [1.0] * groups)
            if len(group_speeds) < groups:
                group_speeds += [1.0] * (groups - len(group_speeds))
            group_speeds = group_speeds[:groups]

            total_weight, segments = _compute_group_segments(group_speeds)
            # total compute = COMPUTE_PER_GROUP * sum(1/speed_i)
            compute = COMPUTE_PER_GROUP * total_weight if groups > 0 else 0.0

            self.stages.append(StageTimeline(
                name, t, appear, compute, settle, segments))
            t += appear + compute + settle
        self.total_duration = t

    def update(self, dt: float):
        if self.playing:
            self.current_time += dt * self.speed
            loop_duration = self.total_duration + self.return_duration
            if self.current_time > loop_duration:
                if self.loop:
                    self.current_time %= loop_duration
                else:
                    self.current_time = loop_duration
                    self.completed = True
                    self.playing = False

    def get_stage(self, name: str) -> StageTimeline:
        for s in self.stages:
            if s.stage_name == name:
                return s
        return None

    def get_stage_phase(self, name: str) -> tuple:
        """Returns (phase_name, local_t) for a named stage."""
        stage = self.get_stage(name)
        if stage is None:
            return ('inactive', 0.0)
        return stage.get_phase(self.current_time)

    def get_current_stage_index(self) -> int:
        for i, s in enumerate(self.stages):
            if self.current_time < s.end_time:
                return i
        return len(self.stages) - 1

    def jump_to_stage(self, index: int):
        index = max(0, min(index, len(self.stages) - 1))
        self.current_time = self.stages[index].start_time
        self.playing = True

    def toggle_play(self):
        self.playing = not self.playing
