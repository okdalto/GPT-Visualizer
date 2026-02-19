from dataclasses import dataclass
import numpy as np
from core.camera import Camera
from core.renderer import InstancedBoxRenderer
from animation.timeline import AnimationTimeline
from visualization.operation_visual import (
    MatmulVisual, AddVisual, ActivationVisual, StaticMatrixVisual
)
from visualization.layout import (
    STAGE_Z, matrix_origin, side_by_side_x, stacked_y, SPACING,
    MATRIX_X_GAP, MHA_MATMUL_X_GAP, QKV_STACK_Y_GAP, MHA_HEAD_Y_GAP,
    MATMUL_Z_C, MATMUL_Z_B,
    RESIDUAL_ADD_Z,
    FFN_W1_Z, FFN_PRERELU_Z,
)
from visualization.colormap import (
    matrix_to_colors_sequential, matrix_to_colors_sequential_per_row,
    matrix_to_colors_sequential_per_row_fade, matrix_to_colors_sequential_fade,
)
from transformer.parameters import TransformerConfig


class Stage:
    """A single visualization stage with its visuals and animation.

    Supports phase_group for intra-stage sequencing: visuals in group 0
    animate first, then group 1, etc. Visuals in the same group run in parallel.
    """

    def __init__(self, name: str):
        self.name = name
        self.visuals = []
        self.labels = []  # list of (text, world_pos_3d)
        self.alpha = 0.0
        self.phase = 'inactive'
        self.phase_t = 0.0
        self.group_segments = []  # set from timeline: [(start_t, end_t), ...]

    def _get_num_groups(self):
        groups = set()
        for v in self.visuals:
            groups.add(getattr(v, 'phase_group', 0))
        return max(len(groups), 1)

    def _get_group_segment(self, g):
        """Return (seg_start, seg_end) in 0-1 range for phase group g."""
        if g < len(self.group_segments):
            return self.group_segments[g]
        # Fallback: equal segments
        n = self._get_num_groups()
        return (g / n, (g + 1) / n)

    def update_animation(self, phase: str, t: float):
        self.phase = phase
        self.phase_t = t

        if phase == 'inactive':
            self.alpha = 0.0
            for v in self.visuals:
                v.alpha = 0.0
                v.appear_t = 0.0
                v.t = 0.0
                v.depart_t = 0.0

        elif phase == 'appear':
            # During stage appear, only group 0 visuals fly in
            self.alpha = 1.0
            for v in self.visuals:
                g = getattr(v, 'phase_group', 0)
                if g == 0:
                    v.alpha = 1.0
                    v.appear_t = t
                else:
                    v.alpha = 0.0
                    v.appear_t = 0.0
                v.t = 0.0
                v.depart_t = 0.0

        elif phase == 'compute':
            # Stagger compute across phase groups (weighted segments)
            self.alpha = 1.0
            for v in self.visuals:
                g = getattr(v, 'phase_group', 0)
                seg_start, seg_end = self._get_group_segment(g)

                if t < seg_start:
                    # Group not started yet
                    v.alpha = 0.0 if g > 0 else 1.0
                    v.appear_t = 1.0 if g == 0 else 0.0
                    v.t = 0.0
                elif t < seg_end:
                    # Active segment
                    local_t = (t - seg_start) / max(seg_end - seg_start, 1e-9)
                    v.alpha = 1.0
                    if g == 0:
                        v.appear_t = 1.0
                        v.t = local_t
                    else:
                        # First 20% of segment: appear (fly in from prev sub-op)
                        # Remaining 80%: compute
                        if local_t < 0.20:
                            v.appear_t = local_t / 0.20
                            v.t = 0.0
                        else:
                            v.appear_t = 1.0
                            v.t = (local_t - 0.20) / 0.80
                else:
                    # Group done
                    v.alpha = 1.0
                    v.appear_t = 1.0
                    v.t = 1.0
                v.depart_t = 0.0

        elif phase in ('settle', 'done'):
            self.alpha = 1.0
            for v in self.visuals:
                v.alpha = 1.0
                v.appear_t = 1.0
                v.t = 1.0
                v.depart_t = 0.0

    def get_all_instance_data(self) -> np.ndarray:
        """Collect instance data from all visible visuals in this stage."""
        all_data = []
        for v in self.visuals:
            if v.alpha < 0.01:
                continue
            data = v.get_instance_data()
            if data.shape[0] > 0:
                all_data.append(data)
        if all_data:
            return np.vstack(all_data)
        return np.zeros((0, 10), dtype=np.float32)


@dataclass
class _PredCharReturn:
    """Data for the predicted-character-return animation (return phase)."""
    char: str                              # predicted character (e.g. 'e')
    start_pos: np.ndarray                  # output label position (flight start)
    end_pos: np.ndarray                    # empty slot in char_display (flight end)
    pct_text: str                          # percentage text that fades (e.g. '85%')
    existing_chars: list                   # [(char, pos)] — current chars at next-step spacing
    char_spacing: float                    # next step's char_spacing


# ── Fixed camera geometry (45° azimuth, 45° elevation) ────────────
_CAM_AZIMUTH = np.pi / 4
_CAM_ELEVATION = np.pi / 4
_cam_offset_dir = np.array([
    -np.sin(_CAM_AZIMUTH) * np.cos(_CAM_ELEVATION),
    np.sin(_CAM_ELEVATION),
    -np.cos(_CAM_AZIMUTH) * np.cos(_CAM_ELEVATION),
])
_cam_offset_dir /= np.linalg.norm(_cam_offset_dir)
_CAM_FWD = -_cam_offset_dir
_world_up = np.array([0.0, 1.0, 0.0])
_CAM_RIGHT = np.cross(_CAM_FWD, _world_up)
_CAM_RIGHT = _CAM_RIGHT / np.linalg.norm(_CAM_RIGHT)
_CAM_UP = np.cross(_CAM_RIGHT, _CAM_FWD)
_CAM_OFFSET_DIR = _cam_offset_dir
_CAM_DIST = 200.0
_CAM_MARGIN = 3.5


class Scene:
    def __init__(self, results: dict, config: TransformerConfig, shader,
                 aspect: float = 16.0 / 9.0,
                 input_labels: list[str] = None,
                 output_labels: list[str] = None,
                 renderer=None,
                 final_display: bool = False,
                 logits_only: bool = False,
                 prev_char_spacing: float = None):
        self.results = results
        self.config = config
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.aspect = aspect
        self.final_display = final_display
        self.logits_only = logits_only
        self._prev_char_spacing = prev_char_spacing
        self.has_lm_head = 'logits' in results
        active = [l for l in (input_labels or []) if l not in ('_', '<END>')]
        self.has_char_display = len(active) > 0
        self.timeline = AnimationTimeline(
            has_lm_head=self.has_lm_head,
            has_char_display=self.has_char_display,
            final_display=self.final_display,
            logits_only=self.logits_only,
        )
        if self.final_display:
            self.timeline.loop = False
            self.timeline.return_duration = 0.0
        self.camera = Camera()
        self.renderer = renderer or InstancedBoxRenderer(shader)
        self.shader = shader

        self._char_display_data = []   # [(char, start_pos, row_idx)]
        self._char_spacing = 2.5       # default, updated in _build_all_stages
        self._char_end_positions = {}  # row_idx -> end_pos
        self._predicted_char_return = None  # (char, start_pos, end_pos) for return anim
        self.stages: dict[str, Stage] = {}
        self._build_all_stages()
        # Copy per-group segment boundaries from timeline to stages
        for tl_stage in self.timeline.stages:
            if tl_stage.stage_name in self.stages:
                self.stages[tl_stage.stage_name].group_segments = tl_stage.group_segments
        self._wire_flow_connections()
        self._build_camera_path()
        self._build_labels()
        self._setup_predicted_char_return()

        # Smooth label fade state
        self._label_fade = {}   # (stage_name, idx) -> current alpha
        self._prev_label_time = 0.0

    def _build_all_stages(self):
        r = self.results
        c = self.config
        bs = 0.4
        gap = 0.1

        # =========================================================
        # Stage 0: Character Display (text mode only)
        # Shows input characters big, then flies them to input row positions
        # =========================================================
        if self.has_char_display:
            active_chars = [(i, lbl) for i, lbl in enumerate(self.input_labels)
                            if lbl not in ('_', '<END>')]
            n_active = len(active_chars)
            z_cd = STAGE_Z['char_display']
            # Wider spacing for few chars, tighter for many (prevents overlap)
            # Use previous scene's spacing if provided (seamless transition)
            if self._prev_char_spacing is not None:
                char_spacing = self._prev_char_spacing
            else:
                char_spacing = max(2.0, 4.5 - n_active * 0.15)
            self._char_spacing = char_spacing

            # Dummy visual (boxes hidden; only used so the stage has a visual)
            total_w = (n_active - 1) * char_spacing
            dummy = StaticMatrixVisual(
                np.zeros((2, 2), dtype=np.float32),
                matrix_origin(z_cd, SPACING / 2, 2 + SPACING / 2),
                box_size=bs, gap=gap,
            )
            dummy.phase_group = 0
            stage = Stage('char_display')
            stage.visuals.append(dummy)
            self.stages['char_display'] = stage

            # Store animated char data (start positions centered horizontally)
            for idx, (row_idx, char_label) in enumerate(active_chars):
                start_pos = np.array([
                    total_w / 2 - idx * char_spacing,
                    2,
                    z_cd,
                ], dtype=np.float32)
                self._char_display_data.append((char_label, start_pos, row_idx))

        if self.final_display:
            return

        if self.logits_only:
            self._build_logits_only_stages(r, c, bs, gap)
            return

        # =========================================================
        # Stage 1: Input Embeddings
        # =========================================================
        stage = Stage('input')
        z = STAGE_Z['input']
        w_inp = c.d_model * SPACING
        v = StaticMatrixVisual(r['input'], matrix_origin(z, -w_inp / 2, 2), bs, gap)
        v.phase_group = 0
        stage.visuals.append(v)
        self.stages['input'] = stage

        # =========================================================
        # Stage 2: QKV Projection - Three matmuls in parallel
        # Input × W_Q = Q, Input × W_K = K, Input × W_V = V
        # =========================================================
        stage = Stage('qkv_projection')
        z = STAGE_Z['qkv_projection']

        qkv_names = [
            ('Q', 'W_Q'),
            ('K', 'W_K'),
            ('V', 'W_V'),
        ]
        y_offsets = stacked_y([c.d_model * SPACING] * 3, gap=QKV_STACK_Y_GAP)

        for i, (out_name, w_name) in enumerate(qkv_names):
            w_mat = c.d_model * SPACING
            xs = side_by_side_x([w_mat, w_mat], gap=MATRIX_X_GAP)

            v = MatmulVisual(
                A=r['input'],
                B=r[w_name],
                C=r[out_name],
                origin_a=matrix_origin(z, xs[1], y_offsets[i]),
                origin_b=matrix_origin(z + MATMUL_Z_B, xs[1], y_offsets[i]),
                origin_c=matrix_origin(z + MATMUL_Z_C, xs[0], y_offsets[i]),
                box_size=bs, gap=gap,
            )
            v.phase_group = 0  # All 3 projections in parallel
            stage.visuals.append(v)
        self.stages['qkv_projection'] = stage

        # =========================================================
        # Stage 3: Multi-Head Attention
        # Per head: Scores = Q_h × K_h^T, Weights = softmax(Scores), Out = Weights × V_h
        # Heads run in parallel; sub-ops within a head are sequential (phase_group 0,1,2)
        # =========================================================
        stage = Stage('multi_head_attn')
        z = STAGE_Z['multi_head_attn']
        head_height = c.seq_len * SPACING
        y_offsets_heads = stacked_y([head_height] * c.num_heads, gap=MHA_HEAD_Y_GAP)

        # Compute color normalization ranges from full Q/K/V matrices
        # so head slices keep consistent colors with their parent
        q_vmax = max(abs(r['Q'].max()), abs(r['Q'].min()), 0.01)
        k_vmax = max(abs(r['K'].max()), abs(r['K'].min()), 0.01)
        v_vmax = max(abs(r['V'].max()), abs(r['V'].min()), 0.01)
        concat_vmax = max(abs(r['concat'].max()), abs(r['concat'].min()), 0.01)

        for h in range(c.num_heads):
            q_h = r['Q'].reshape(c.seq_len, c.num_heads, c.d_k)[:, h, :]
            k_h = r['K'].reshape(c.seq_len, c.num_heads, c.d_k)[:, h, :]
            v_h = r['V'].reshape(c.seq_len, c.num_heads, c.d_k)[:, h, :]
            scores_h = r['attention_scores'][h]
            weights_h = r['attention_weights'][h]
            out_h = r['head_outputs'][h]

            yo = y_offsets_heads[h]

            # Sub-op 1: Q_h × K_h^T = Scores (phase_group 0)
            w_qh = c.d_k * SPACING
            w_kt = c.seq_len * SPACING
            w_scores = c.seq_len * SPACING
            xs1 = side_by_side_x([w_scores, w_kt], gap=MHA_MATMUL_X_GAP)

            v = MatmulVisual(
                A=q_h, B=k_h.T, C=scores_h,
                origin_a=matrix_origin(z, xs1[1], yo),
                origin_b=matrix_origin(z + MATMUL_Z_B, xs1[1], yo),
                origin_c=matrix_origin(z + MATMUL_Z_C, xs1[0], yo),
                box_size=bs, gap=gap,
            )
            v.color_vmax_a = q_vmax  # Q_h uses full Q's range
            v.color_vmax_b = k_vmax  # K_h^T uses full K's range
            v.phase_group = 0
            v.fade_inputs_on_takeover = True  # Q_h/K_h^T fade gradually when softmax starts
            stage.visuals.append(v)

            # Sub-op 2: softmax(Scores) = Weights (phase_group 1)
            # In-place at Scores C position: just color transition + label change
            scores_c_origin = matrix_origin(z + MATMUL_Z_C, xs1[0], yo)

            v = ActivationVisual(
                pre=scores_h, post=weights_h,
                origin=scores_c_origin.copy(),
                box_size=bs, gap=gap,
            )
            v.phase_group = 1
            stage.visuals.append(v)

            # Sub-op 3: Weights × V_h = Out_h (phase_group 2)
            # Weights at Scores C position (softmax happened in-place)
            z_wt = z + MATMUL_Z_C
            w_wt = c.seq_len * SPACING
            w_out = c.d_k * SPACING
            xs3 = side_by_side_x([w_out, w_wt], gap=MHA_MATMUL_X_GAP)
            x_shift = xs1[0] - xs3[1]

            v = MatmulVisual(
                A=weights_h, B=v_h, C=out_h,
                origin_a=scores_c_origin.copy(),
                origin_b=matrix_origin(z_wt + MATMUL_Z_B, xs1[0], yo),
                origin_c=matrix_origin(z_wt + MATMUL_Z_C, xs3[0] + x_shift, yo),
                box_size=bs, gap=gap,
            )
            v.color_vmax_b = v_vmax  # V_h uses full V's range
            v.color_vmax_c = concat_vmax  # Out_h uses concat's range for seamless transition
            v.phase_group = 2
            stage.visuals.append(v)

        self.stages['multi_head_attn'] = stage

        # =========================================================
        # Stage 4: Concat + Output Projection
        # Concat(head_outputs) × W_O = attn_output
        # =========================================================
        stage = Stage('concat_output_proj')
        z = STAGE_Z['concat_output_proj']

        w_mat = c.d_model * SPACING
        xs = side_by_side_x([w_mat, w_mat], gap=MATRIX_X_GAP)

        v = MatmulVisual(
            A=r['concat'], B=r['W_O'], C=r['attn_output'],
            origin_a=matrix_origin(z, xs[1], 2),
            origin_b=matrix_origin(z + MATMUL_Z_B, xs[1], 2),
            origin_c=matrix_origin(z + MATMUL_Z_C, xs[0], 2),
            box_size=bs, gap=gap,
        )
        v.color_vmax_a = concat_vmax  # match MHA out_h color range
        v.phase_group = 0
        stage.visuals.append(v)
        self.stages['concat_output_proj'] = stage

        # =========================================================
        # Stage 5: Residual + LayerNorm 1
        # residual1 = input + attn_output, then LayerNorm
        # Sequential: Add (group 0) → LN (group 1)
        # All matrices centered, spaced only along Z
        # =========================================================
        w16 = c.d_model * SPACING
        self._build_residual_ln_stage(
            'residual_ln1', r['input'], r['attn_output'],
            r['residual1'], r['layernorm1'], w16, bs, gap)

        # =========================================================
        # Stage 6: FFN
        # Sequential: matmul (group 0) → ReLU (group 1) → matmul (group 2)
        # =========================================================
        stage = Stage('ffn')
        z = STAGE_Z['ffn']

        # Center each matrix at x=0, space along Z axis to avoid overlap
        w_model = c.d_model * SPACING
        w_hidden = c.d_ff * SPACING

        # Matmul 1: LN1 × W1 = pre-relu
        z_prerelu = z + FFN_PRERELU_Z
        v = MatmulVisual(
            A=r['layernorm1'], B=r['W1'], C=r['ffn_pre_relu'],
            origin_a=matrix_origin(z, -w_model / 2, 2),
            origin_b=matrix_origin(z + FFN_W1_Z, -w_hidden / 2, 2),
            origin_c=matrix_origin(z_prerelu, -w_hidden / 2, 2),
            box_size=bs, gap=gap,
        )
        v.phase_group = 0
        v.fade_inputs_on_takeover = True  # LN1/W1 fade gradually when ReLU starts
        stage.visuals.append(v)

        # ReLU: applies in-place at Matmul 1's C position
        v = ActivationVisual(
            pre=r['ffn_pre_relu'], post=r['ffn_hidden'],
            origin=matrix_origin(z_prerelu, -w_hidden / 2, 2),
            box_size=bs, gap=gap,
        )
        v.phase_group = 1
        stage.visuals.append(v)

        # Matmul 2: hidden × W2 = ffn_output (starts at ReLU position)
        z_ffn2 = z_prerelu
        v = MatmulVisual(
            A=r['ffn_hidden'], B=r['W2'], C=r['ffn_output'],
            origin_a=matrix_origin(z_ffn2, -w_hidden / 2, 2),
            origin_b=matrix_origin(z_ffn2 + MATMUL_Z_B, -w_hidden / 2, 2),
            origin_c=matrix_origin(z_ffn2 + FFN_PRERELU_Z, -w_model / 2, 2),
            box_size=bs, gap=gap,
        )
        v.phase_group = 2
        stage.visuals.append(v)
        self.stages['ffn'] = stage

        # =========================================================
        # Stage 7: Residual + LayerNorm 2
        # Sequential: Add (group 0) → LN (group 1)
        # All matrices centered, spaced only along Z
        # =========================================================
        self._build_residual_ln_stage(
            'residual_ln2', r['layernorm1'], r['ffn_output'],
            r['residual2'], r['layernorm2'], w16, bs, gap)

        # =========================================================
        # Stage 8: Block 1 Output
        # =========================================================
        block0_output = r['block_0']['output'] if 'block_0' in r else r['output']
        stage = Stage('output')
        z = STAGE_Z['output']
        w_out = c.d_model * SPACING
        v = StaticMatrixVisual(block0_output, matrix_origin(z, -w_out / 2, 2), bs, gap)
        v.phase_group = 0
        stage.visuals.append(v)
        self.stages['output'] = stage

        # =========================================================
        # Stages 9-11: Blocks 2-4 (simplified: in-place transition)
        # =========================================================
        for blk_idx, blk_name in enumerate(['block_2', 'block_3', 'block_4'], start=1):
            if f'block_{blk_idx}' not in r:
                break
            pre = r[f'block_{blk_idx - 1}']['output']
            post = r[f'block_{blk_idx}']['output']
            z_blk = STAGE_Z[blk_name]
            stage = Stage(blk_name)
            v = ActivationVisual(pre, post, matrix_origin(z_blk, -w_out / 2, 2), bs, gap)
            v.phase_group = 0
            stage.visuals.append(v)
            self.stages[blk_name] = stage

        # =========================================================
        # Output Projection (LM head only)
        # Uses final block's output
        # =========================================================
        final_output = r['output']
        if 'logits' in r:
            stage = Stage('output_projection')
            # Place A at final block's Z
            z_out = STAGE_Z.get('block_4', STAGE_Z['output'])
            if 'block_3' not in r:
                z_out = STAGE_Z['output']
            vocab_size = r['logits'].shape[1]
            w_vocab = vocab_size * SPACING
            w_model = c.d_model * SPACING
            c_x = -w_model / 2 - MATRIX_X_GAP - w_vocab

            v = MatmulVisual(
                A=final_output, B=r['W_out'], C=r['logits'],
                origin_a=matrix_origin(z_out, -w_model / 2, 2),
                origin_b=matrix_origin(z_out + MATMUL_Z_B, -w_model / 2, 2),
                origin_c=matrix_origin(z_out + MATMUL_Z_C, c_x, 2),
                box_size=bs, gap=gap,
            )
            v.phase_group = 0
            v.is_stage_output = True
            stage.visuals.append(v)
            self.stages['output_projection'] = stage

            # =========================================================
            # Stage 10: Token Probabilities (LM head only)
            # Group 0: logits → probs (softmax)
            # Group 1: probs → row_probs (fade non-prediction rows)
            # Group 2: row_probs → token_selected (argmax within row)
            # =========================================================
            stage = Stage('token_probs')
            z_tp = STAGE_Z['token_probs']
            logits_origin = matrix_origin(z_tp, -w_vocab / 2, 2)

            # Derive intermediate matrices from pred_pos
            pred_pos = int(r.get('pred_pos', 0))
            row_probs = np.zeros_like(r['probs'])
            row_probs[pred_pos] = r['probs'][pred_pos]

            token_selected = np.zeros_like(r['probs'])
            best_j = int(np.argmax(r['probs'][pred_pos]))
            token_selected[pred_pos, best_j] = 1.0

            # Group 0: softmax (logits → probs, full 8×28)
            v = ActivationVisual(
                pre=r['logits'], post=r['probs'],
                origin=logits_origin.copy(),
                box_size=bs, gap=gap,
            )
            v.phase_group = 0
            v.color_fn_post = lambda m, a: matrix_to_colors_sequential_per_row(m, a)
            stage.visuals.append(v)

            # Group 1: row selection (fade out non-prediction rows)
            v = ActivationVisual(
                pre=r['probs'], post=row_probs,
                origin=logits_origin.copy(),
                box_size=bs, gap=gap,
            )
            v.phase_group = 1
            v.color_fn_pre = lambda m, a: matrix_to_colors_sequential_per_row(m, a)
            v.color_fn_post = lambda m, a: matrix_to_colors_sequential_per_row_fade(m, a)
            stage.visuals.append(v)

            # Group 2: argmax (select highest probability token)
            v = ActivationVisual(
                pre=row_probs, post=token_selected,
                origin=logits_origin.copy(),
                box_size=bs, gap=gap,
            )
            v.phase_group = 2
            v.color_fn_pre = lambda m, a: matrix_to_colors_sequential_per_row_fade(m, a)
            v.color_fn_post = lambda m, a: matrix_to_colors_sequential_fade(m, a)
            stage.visuals.append(v)
            self.stages['token_probs'] = stage

    def _build_residual_ln_stage(self, name, add_a, add_b, add_c, ln_post, w, bs, gap):
        """Build a Residual + LayerNorm stage (Add group 0 → LN group 1)."""
        stage = Stage(name)
        z = STAGE_Z[name]
        xs = side_by_side_x([w, w], gap=MATRIX_X_GAP)

        v = AddVisual(
            A=add_a, B=add_b, C=add_c,
            origin_a=matrix_origin(z, xs[1], 2),
            origin_b=matrix_origin(z, xs[0], 2),
            origin_c=matrix_origin(z + RESIDUAL_ADD_Z, -w / 2, 2),
            box_size=bs, gap=gap,
        )
        v.phase_group = 0
        stage.visuals.append(v)

        v = ActivationVisual(
            pre=add_c, post=ln_post,
            origin=matrix_origin(z + RESIDUAL_ADD_Z, -w / 2, 2),
            box_size=bs, gap=gap,
        )
        v.phase_group = 1
        stage.visuals.append(v)
        self.stages[name] = stage

    def _build_logits_only_stages(self, r, c, bs, gap):
        """Build abbreviated pipeline for fast generation steps.

        Shows static snapshots of each block's output, like pressing the
        right-arrow key rapidly: input → block 1–4 → logits → token_probs.
        No compute animation for blocks — just appear and settle.
        """
        if 'logits' not in r:
            return

        w_out = c.d_model * SPACING

        # ── Input (0-duration stage, needed for char fly destination) ──
        stage = Stage('input')
        z_inp = STAGE_Z['input']
        v_inp = StaticMatrixVisual(
            r['input'], matrix_origin(z_inp, -w_out / 2, 2), bs, gap)
        v_inp.phase_group = 0
        stage.visuals.append(v_inp)
        self.stages['input'] = stage

        # ── Blocks 1–4: static snapshots of each block's output ──
        block_outputs = []
        for blk_idx in range(c.num_blocks):
            key = f'block_{blk_idx}'
            if key in r and 'output' in r[key]:
                block_outputs.append(r[key]['output'])
        # Fallback if no per-block data
        if not block_outputs:
            block_outputs = [r['output']] * 4

        for i, blk_name in enumerate(['block_1', 'block_2', 'block_3', 'block_4']):
            if i >= len(block_outputs):
                break
            stage = Stage(blk_name)
            z_blk = STAGE_Z[blk_name]
            v = StaticMatrixVisual(
                block_outputs[i],
                matrix_origin(z_blk, -w_out / 2, 2), bs, gap)
            v.phase_group = 0
            stage.visuals.append(v)
            self.stages[blk_name] = stage

        # ── Output Projection: static logits matrix ──
        vocab_size = r['logits'].shape[1]
        w_vocab = vocab_size * SPACING
        stage = Stage('output_projection')
        z_out = STAGE_Z['output_projection']
        v_out = StaticMatrixVisual(
            r['logits'],
            matrix_origin(z_out + MATMUL_Z_C, -w_vocab / 2, 2), bs, gap)
        v_out.phase_group = 0
        stage.visuals.append(v_out)
        self.stages['output_projection'] = stage

        # ── Token Probabilities (softmax → row selection → argmax) ──
        stage = Stage('token_probs')
        z_tp = STAGE_Z['token_probs']
        logits_origin = matrix_origin(z_tp, -w_vocab / 2, 2)

        pred_pos = int(r.get('pred_pos', 0))
        row_probs = np.zeros_like(r['probs'])
        row_probs[pred_pos] = r['probs'][pred_pos]

        token_selected = np.zeros_like(r['probs'])
        best_j = int(np.argmax(r['probs'][pred_pos]))
        token_selected[pred_pos, best_j] = 1.0

        v = ActivationVisual(
            pre=r['logits'], post=r['probs'],
            origin=logits_origin.copy(), box_size=bs, gap=gap,
        )
        v.phase_group = 0
        v.color_fn_post = lambda m, a: matrix_to_colors_sequential_per_row(m, a)
        stage.visuals.append(v)

        v = ActivationVisual(
            pre=r['probs'], post=row_probs,
            origin=logits_origin.copy(), box_size=bs, gap=gap,
        )
        v.phase_group = 1
        v.color_fn_pre = lambda m, a: matrix_to_colors_sequential_per_row(m, a)
        v.color_fn_post = lambda m, a: matrix_to_colors_sequential_per_row_fade(m, a)
        stage.visuals.append(v)

        v = ActivationVisual(
            pre=row_probs, post=token_selected,
            origin=logits_origin.copy(), box_size=bs, gap=gap,
        )
        v.phase_group = 2
        v.color_fn_pre = lambda m, a: matrix_to_colors_sequential_per_row_fade(m, a)
        v.color_fn_post = lambda m, a: matrix_to_colors_sequential_fade(m, a)
        stage.visuals.append(v)
        self.stages['token_probs'] = stage

        # ── Wire flow connections ──
        # Block 1 flies from input position
        self.stages['block_1'].visuals[0].from_origin = v_inp.origin.copy()

        # Chain blocks: each flies from previous block's position
        prev_v = self.stages['block_1'].visuals[0]
        for blk_name in ('block_2', 'block_3', 'block_4'):
            if blk_name not in self.stages:
                break
            bv = self.stages[blk_name].visuals[0]
            bv.from_origin = prev_v.origin.copy()
            prev_v = bv

        # Output projection flies from last block
        self.stages['output_projection'].visuals[0].from_origin = prev_v.origin.copy()

        # Token probs: group 0 flies from output_projection, then in-place transitions
        tp = self.stages['token_probs'].visuals
        tp[0].from_origin = v_out.origin.copy()
        tp[1].from_origin = tp[0].origin.copy()
        tp[2].from_origin = tp[1].origin.copy()

    def _wire_flow_connections(self):
        """Wire up from_origin on each visual so data flies from the previous stage/sub-op."""
        s = self.stages
        if 'input' not in s:
            return
        # logits_only wiring is handled inside _build_logits_only_stages
        if self.logits_only:
            return

        # --- Input stage: fade in at position ---
        input_v = s['input'].visuals[0]
        input_v.is_stage_output = True
        input_origin = input_v.origin

        # --- QKV Projection: A (input matrix) flies from input stage ---
        for idx, v in enumerate(s['qkv_projection'].visuals):
            v.from_origin_a = input_origin.copy()
            # Q (0), K (1): used immediately in MHA sub-op 1 → seamless hide
            # V (2): used later in sub-op 3 → fade gradually
            if idx < 2:
                v.is_stage_output = True

        # --- MHA: per-head split + sequential sub-ops ---
        qkv_Q = s['qkv_projection'].visuals[0]  # Q projection result
        qkv_K = s['qkv_projection'].visuals[1]  # K projection result
        qkv_V = s['qkv_projection'].visuals[2]  # V projection result
        num_heads = self.config.num_heads
        d_k = self.config.d_k

        for h in range(num_heads):
            base = h * 3
            mha = s['multi_head_attn'].visuals

            # Sub-op 1 (Q_h × K_h^T): Q_h comes from head h's SLICE of Q
            q_slice_origin = qkv_Q.origin_c.copy()
            q_slice_origin[0] -= h * d_k * SPACING
            mha[base].from_origin_a = q_slice_origin

            # K_h^T (B of sub-op 1): transpose changes shape (8×4 → 4×8),
            # animate the transpose by flying each element from [c,r] → [r,c]
            k_slice_origin = qkv_K.origin_c.copy()
            k_slice_origin[0] -= h * d_k * SPACING
            mha[base].from_origin_b = k_slice_origin
            mha[base].transpose_fly_b = True

            # Sub-op 2 (softmax): data from Scores result position
            mha[base + 1].from_origin = mha[base].origin_c.copy()

            # Sub-op 3 (Weights × V_h): Weights stay at softmax position (no fly-in)
            mha[base + 2].from_origin_a = mha[base + 1].origin.copy()
            mha[base + 2].is_stage_output = True  # Head output goes to concat

            # V_h (B of sub-op 3): flies from V output, head h's slice
            # Fade in gradually as it flies (not seamless pop-in)
            v_slice_origin = qkv_V.origin_c.copy()
            v_slice_origin[0] -= h * d_k * SPACING
            mha[base + 2].from_origin_b = v_slice_origin
            mha[base + 2].seamless_b = False

        # --- Concat × W_O: head outputs merge into concat via slice-aware fly-in ---
        concat_v = s['concat_output_proj'].visuals[0]
        concat_v.is_stage_output = True
        # Build from_origin_a_slices: each head's output → its column range in concat
        head_slices = []
        for h in range(num_heads):
            head_out_visual = s['multi_head_attn'].visuals[h * 3 + 2]  # Head h's output matmul
            col_start = h * d_k
            col_end = (h + 1) * d_k
            head_slices.append((col_start, col_end, head_out_visual.origin_c.copy()))
        concat_v.from_origin_a_slices = head_slices

        # --- Residual + LN1 ---
        res_ln1 = s['residual_ln1'].visuals
        # A = input (skip connection from way back) → fade in, not seamless
        res_ln1[0].from_origin_a = input_origin.copy()
        res_ln1[0].seamless_a = False
        # B = attn_output from concat stage (direct)
        res_ln1[0].from_origin_b = concat_v.origin_c.copy()
        # LN: data from Add result
        res_ln1[1].from_origin = res_ln1[0].origin_c.copy()
        res_ln1[1].is_stage_output = True  # LN1 output goes to FFN

        # --- FFN: sequential sub-ops ---
        ffn = s['ffn'].visuals
        ln1_origin = res_ln1[1].origin  # LN1 output position
        # Matmul 1: A from LN1 output
        ffn[0].from_origin_a = ln1_origin.copy()
        # ReLU: from matmul 1 result
        ffn[1].from_origin = ffn[0].origin_c.copy()
        # Matmul 2: A from ReLU result
        ffn[2].from_origin_a = ffn[1].origin.copy()
        ffn[2].is_stage_output = True  # FFN output goes to residual2

        # --- Residual + LN2 ---
        res_ln2 = s['residual_ln2'].visuals
        # A = LN1 output (skip connection) → fade in, not seamless
        res_ln2[0].from_origin_a = ln1_origin.copy()
        res_ln2[0].seamless_a = False
        # B = FFN output
        res_ln2[0].from_origin_b = ffn[2].origin_c.copy()
        # LN: data from Add result
        res_ln2[1].from_origin = res_ln2[0].origin_c.copy()
        res_ln2[1].is_stage_output = True  # LN2 output goes to output stage

        # --- Output: from LN2 result ---
        output_v = s['output'].visuals[0]
        output_v.from_origin = res_ln2[1].origin.copy()

        # --- Blocks 2-4: chain from previous block ---
        prev_v = output_v
        for blk_name in ['block_2', 'block_3', 'block_4']:
            if blk_name not in s:
                break
            blk_v = s[blk_name].visuals[0]
            blk_v.from_origin = prev_v.origin.copy()
            prev_v.is_stage_output = True
            prev_v = blk_v

        # --- Output Projection: A from last block ---
        if 'output_projection' in s:
            out_proj_v = s['output_projection'].visuals[0]
            out_proj_v.from_origin_a = prev_v.origin.copy()
            prev_v.is_stage_output = True

            # --- Token Probs: group 0 from output_projection C ---
            tp = s['token_probs'].visuals
            tp[0].from_origin = out_proj_v.origin_c.copy()
            # group 1 from group 0's position (in-place)
            tp[1].from_origin = tp[0].origin.copy()
            # group 2 from group 1's position (in-place)
            tp[2].from_origin = tp[1].origin.copy()

    def _get_visual_extents(self, v):
        """Return (origin, rows, cols, spacing, box_size) tuples for a visual."""
        if isinstance(v, MatmulVisual):
            m, k = v.A.shape
            k_b, n_b = v.B.shape
            m_c, n_c = v.C.shape
            return [
                (v.origin_a, m, k, v.sp, v.bs),
                (v.origin_b, k_b, n_b, v.sp_b, v.bs_b),
                (v.origin_c, m_c, n_c, v.sp_c, v.bs_c),
            ]
        elif isinstance(v, AddVisual):
            rows, cols = v.A.shape
            return [
                (v.origin_a, rows, cols, v.sp, v.bs),
                (v.origin_b, rows, cols, v.sp, v.bs),
                (v.origin_c, rows, cols, v.sp, v.bs),
            ]
        elif isinstance(v, ActivationVisual):
            rows, cols = v.pre.shape
            return [(v.origin, rows, cols, v.sp, v.bs)]
        elif isinstance(v, StaticMatrixVisual):
            rows, cols = v.matrix.shape
            return [(v.origin, rows, cols, v.sp, v.bs)]
        return []

    def _compute_stage_view_bounds(self, stage_name, cam_right, cam_up,
                                   group=None):
        """Compute tight bounding box in view space (cam_right / cam_up).

        Projects each visual's actual corner points into view space,
        giving tighter bounds than projecting world-AABB corners.
        If group is specified, only include visuals of that phase_group.
        Returns (vx_min, vx_max, vy_min, vy_max).
        """
        stage = self.stages[stage_name]
        vx_min = np.inf
        vx_max = -np.inf
        vy_min = np.inf
        vy_max = -np.inf

        for v in stage.visuals:
            if group is not None and getattr(v, 'phase_group', 0) != group:
                continue
            for origin, rows, cols, sp, bs in self._get_visual_extents(v):
                x0 = origin[0] - (cols - 1) * sp - bs / 2
                x1 = origin[0] + bs / 2
                y0 = origin[1] - (rows - 1) * sp - bs / 2
                y1 = origin[1] + bs / 2
                z0 = origin[2] - bs / 2
                z1 = origin[2] + bs / 2
                for wx in (x0, x1):
                    for wy in (y0, y1):
                        for wz in (z0, z1):
                            pt = np.array([wx, wy, wz])
                            vx = np.dot(pt, cam_right)
                            vy = np.dot(pt, cam_up)
                            vx_min = min(vx_min, vx)
                            vx_max = max(vx_max, vx)
                            vy_min = min(vy_min, vy)
                            vy_max = max(vy_max, vy)

        return vx_min, vx_max, vy_min, vy_max

    def _frame_char_positions(self, positions):
        """Compute camera framing for a set of world positions (char_display style).

        Returns (position, target, ortho_size) — the same format used by
        Camera.add_waypoint.
        """
        positions = np.asarray(positions, dtype=np.float64)
        center = positions.mean(axis=0)

        vcx = float(np.dot(center, _CAM_RIGHT))
        vcy = float(np.dot(center, _CAM_UP))
        fwd_c = float(np.dot(center, _CAM_FWD))
        target = vcx * _CAM_RIGHT + vcy * _CAM_UP + fwd_c * _CAM_FWD
        position = target + _CAM_OFFSET_DIR * _CAM_DIST

        vx_vals = [float(np.dot(p, _CAM_RIGHT)) for p in positions]
        half_vx = (max(vx_vals) - min(vx_vals)) / 2.0 + 4.0 if len(vx_vals) > 1 else 4.0
        half_vy = 4.0
        ortho_size = max(half_vy + _CAM_MARGIN, half_vx / self.aspect + _CAM_MARGIN)
        return position, target, ortho_size

    def _build_camera_path(self):
        """Set up camera waypoints with ortho framing from 45° upper-left.

        Uses view-space bounding boxes with a fixed additive margin so
        every stage has identical margins regardless of content size.
        """

        def compute_framing(stage_name, group=None):
            vx_min, vx_max, vy_min, vy_max = \
                self._compute_stage_view_bounds(stage_name, _CAM_RIGHT, _CAM_UP,
                                                group=group)

            vcx = (vx_min + vx_max) / 2.0
            vcy = (vy_min + vy_max) / 2.0
            stage = self.stages[stage_name]
            fwd_sum, fwd_cnt = 0.0, 0
            for v in stage.visuals:
                if group is not None and getattr(v, 'phase_group', 0) != group:
                    continue
                for origin, *_ in self._get_visual_extents(v):
                    fwd_sum += np.dot(origin, _CAM_FWD)
                    fwd_cnt += 1
            fwd_center = fwd_sum / max(fwd_cnt, 1)

            target = vcx * _CAM_RIGHT + vcy * _CAM_UP + fwd_center * _CAM_FWD
            position = target + _CAM_OFFSET_DIR * _CAM_DIST

            half_w = (vx_max - vx_min) / 2.0 + _CAM_MARGIN
            half_h = (vy_max - vy_min) / 2.0 + _CAM_MARGIN
            ortho_size = max(half_h, half_w / self.aspect)
            return position, target, ortho_size

        def compute_char_display_framing():
            positions = [d[1] for d in self._char_display_data]
            return self._frame_char_positions(positions)

        for tl_stage in self.timeline.stages:
            sname = tl_stage.stage_name
            stage = self.stages[sname]
            num_groups = stage._get_num_groups()

            # Skip zero-duration stages (e.g. 'input' when char_display handles it)
            if tl_stage.total_duration <= 0:
                continue

            if num_groups <= 1:
                # Single group: one framing for the whole stage
                if sname == 'char_display' and self._char_display_data:
                    pos, tgt, osz = compute_char_display_framing()
                    t_arrive = tl_stage.start_time + tl_stage.appear_duration * 0.5
                    self.camera.add_waypoint(t_arrive, pos, tgt, osz)
                    # During compute, transition camera to input framing
                    # so input matrix is visible as chars fly toward it
                    if 'input' in self.stages:
                        t_compute_start = tl_stage.start_time + tl_stage.appear_duration
                        inp_pos, inp_tgt, inp_osz = compute_framing('input')
                        self.camera.add_waypoint(t_compute_start + 0.3, pos, tgt, osz)
                        self.camera.add_waypoint(tl_stage.end_time, inp_pos, inp_tgt, inp_osz)
                    else:
                        self.camera.add_waypoint(tl_stage.end_time, pos, tgt, osz)
                else:
                    pos, tgt, osz = compute_framing(sname)
                    t_arrive = tl_stage.start_time + tl_stage.appear_duration * 0.5
                    self.camera.add_waypoint(t_arrive, pos, tgt, osz)
                    self.camera.add_waypoint(tl_stage.end_time, pos, tgt, osz)
            else:
                # Multi-group: camera reframes per operation
                t_compute = tl_stage.start_time + tl_stage.appear_duration
                compute_dur = tl_stage.compute_duration

                # Appear phase → frame group 0
                pos, tgt, osz = compute_framing(sname, group=0)
                t_arrive = tl_stage.start_time + tl_stage.appear_duration * 0.5
                self.camera.add_waypoint(t_arrive, pos, tgt, osz)

                for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                    seg_t = t_compute + seg_s * compute_dur
                    seg_t_end = t_compute + seg_e * compute_dur
                    seg_dur = (seg_e - seg_s) * compute_dur

                    pos, tgt, osz = compute_framing(sname, group=gi)
                    # Transition in the first 30% of the segment (during appear)
                    t_arrive = seg_t + min(seg_dur * 0.3, 0.8)
                    self.camera.add_waypoint(t_arrive, pos, tgt, osz)
                    self.camera.add_waypoint(seg_t_end, pos, tgt, osz)

                # Settle: hold last group's framing
                self.camera.add_waypoint(tl_stage.end_time, pos, tgt, osz)

        first_name = self.timeline.stages[0].stage_name
        if first_name == 'char_display' and self._char_display_data:
            first_pos, first_tgt, first_osz = compute_char_display_framing()
        else:
            first_pos, first_tgt, first_osz = compute_framing(first_name)
        loop_end = self.timeline.total_duration + self.timeline.return_duration
        self.camera.add_waypoint(loop_end, first_pos, first_tgt, first_osz)

    def _matrix_label_pos(self, origin, rows, cols, sp):
        """Position above the center-top of a matrix."""
        x = origin[0] - (cols - 1) * sp / 2
        y = origin[1] + 1.2
        z = origin[2]
        return np.array([x, y, z], dtype=np.float32)

    def _build_labels(self):
        """Create labels for every matrix in every stage.

        Each label is (text, world_pos, visual, phase, no_bg) where phase is:
          'appear'  – fades in with visual.appear_t  (inputs that fly in)
          'compute' – fades in with visual.t          (results that emerge)
        no_bg=True disables the dark background quad behind the label.
        """
        if self.final_display:
            return
        if self.logits_only:
            self._build_logits_only_labels()
            return
        c = self.config

        def lbl(stage_name, text, origin, rows, cols, sp, vis, phase='appear'):
            pos = self._matrix_label_pos(origin, rows, cols, sp)
            self.stages[stage_name].labels.append((text, pos, vis, phase, False))

        # --- Input ---
        v = self.stages['input'].visuals[0]
        r, cl = v.matrix.shape
        lbl('input', 'X', v.origin, r, cl, v.sp, v)

        # Per-row character labels (to the right of the matrix on screen)
        # Also compute end positions for char_display fly animation
        if self.input_labels:
            for row_idx, char_label in enumerate(self.input_labels):
                pos = np.array([
                    v.origin[0] - cl * v.sp - 0.8,
                    v.origin[1] - row_idx * v.sp,
                    v.origin[2]
                ], dtype=np.float32)
                self._char_end_positions[row_idx] = pos.copy()
                self.stages['input'].labels.append(
                    (char_label, pos, v, 'appear', True, 'left'))

        # --- QKV Projection: A flies in (appear), B+C emerge (compute) ---
        qkv_label_names = [('W_Q', 'Q'), ('W_K', 'K'), ('W_V', 'V')]
        for i, (w_name, out_name) in enumerate(qkv_label_names):
            v = self.stages['qkv_projection'].visuals[i]
            ra, ca = v.A.shape
            lbl('qkv_projection', 'X', v.origin_a, ra, ca, v.sp, v, 'appear')
            rb, cb = v.B.shape
            lbl('qkv_projection', w_name, v.origin_b, rb, cb, v.sp_b, v, 'compute')
            rc, cc = v.C.shape
            lbl('qkv_projection', out_name, v.origin_c, rc, cc, v.sp_c, v, 'compute')

        # --- Multi-Head Attention: per head 3 sub-ops ---
        for h in range(c.num_heads):
            base = h * 3
            sn = 'multi_head_attn'

            # Sub-op 1: Q_h x K_h^T = Scores
            v0 = self.stages[sn].visuals[base]
            ra, ca = v0.A.shape
            lbl(sn, 'Q', v0.origin_a, ra, ca, v0.sp, v0, 'appear')
            rb, cb = v0.B.shape
            lbl(sn, 'K^T', v0.origin_b, rb, cb, v0.sp_b, v0, 'compute')
            rc, cc = v0.C.shape
            lbl(sn, 'Scores', v0.origin_c, rc, cc, v0.sp_c, v0, 'compute')

            # Sub-op 2: softmax (flies in from scores)
            v1 = self.stages[sn].visuals[base + 1]
            r1, c1 = v1.pre.shape
            lbl(sn, 'Softmax', v1.origin, r1, c1, v1.sp, v1, 'appear')

            # Sub-op 3: Weights x V_h = Head Out
            # (Weights label omitted: same position as Softmax above)
            v2 = self.stages[sn].visuals[base + 2]
            rb, cb = v2.B.shape
            lbl(sn, 'V', v2.origin_b, rb, cb, v2.sp_b, v2, 'compute')
            rc, cc = v2.C.shape
            lbl(sn, 'Head Out', v2.origin_c, rc, cc, v2.sp_c, v2, 'compute')

        # --- Concat + Output Projection ---
        v = self.stages['concat_output_proj'].visuals[0]
        ra, ca = v.A.shape
        lbl('concat_output_proj', 'Concat', v.origin_a, ra, ca, v.sp, v, 'appear')
        rb, cb = v.B.shape
        lbl('concat_output_proj', 'W_O', v.origin_b, rb, cb, v.sp_b, v, 'compute')
        rc, cc = v.C.shape
        lbl('concat_output_proj', 'Attn Out', v.origin_c, rc, cc, v.sp_c, v, 'compute')

        # --- Residual + LayerNorm 1: Add (A,B fly in, C emerges), then LN ---
        v_add = self.stages['residual_ln1'].visuals[0]
        ra, ca = v_add.A.shape
        lbl('residual_ln1', 'X', v_add.origin_a, ra, ca, v_add.sp, v_add, 'appear')
        lbl('residual_ln1', 'Attn Out', v_add.origin_b, ra, ca, v_add.sp, v_add, 'appear')
        rc, cc = v_add.C.shape
        lbl('residual_ln1', 'Add', v_add.origin_c, rc, cc, v_add.sp, v_add, 'compute')
        v_ln = self.stages['residual_ln1'].visuals[1]
        rl, cll = v_ln.post.shape
        lbl('residual_ln1', 'LayerNorm', v_ln.origin, rl, cll, v_ln.sp, v_ln, 'appear')

        # --- FFN ---
        v0 = self.stages['ffn'].visuals[0]  # matmul1
        ra, ca = v0.A.shape
        lbl('ffn', 'LN1', v0.origin_a, ra, ca, v0.sp, v0, 'appear')
        rb, cb = v0.B.shape
        lbl('ffn', 'W1', v0.origin_b, rb, cb, v0.sp_b, v0, 'compute')
        # C overlaps with ReLU activation position -> skip C label

        v1 = self.stages['ffn'].visuals[1]  # relu
        r1, c1 = v1.post.shape
        lbl('ffn', 'ReLU', v1.origin, r1, c1, v1.sp, v1, 'appear')

        v2 = self.stages['ffn'].visuals[2]  # matmul2
        # (Hidden label omitted: same position as ReLU above)
        rb, cb = v2.B.shape
        lbl('ffn', 'W2', v2.origin_b, rb, cb, v2.sp_b, v2, 'compute')
        rc, cc = v2.C.shape
        lbl('ffn', 'FFN Out', v2.origin_c, rc, cc, v2.sp_c, v2, 'compute')

        # --- Residual + LayerNorm 2 ---
        v_add = self.stages['residual_ln2'].visuals[0]
        ra, ca = v_add.A.shape
        lbl('residual_ln2', 'LN1', v_add.origin_a, ra, ca, v_add.sp, v_add, 'appear')
        lbl('residual_ln2', 'FFN Out', v_add.origin_b, ra, ca, v_add.sp, v_add, 'appear')
        rc, cc = v_add.C.shape
        lbl('residual_ln2', 'Add', v_add.origin_c, rc, cc, v_add.sp, v_add, 'compute')
        v_ln = self.stages['residual_ln2'].visuals[1]
        rl, cll = v_ln.post.shape
        lbl('residual_ln2', 'LayerNorm', v_ln.origin, rl, cll, v_ln.sp, v_ln, 'appear')

        # --- Output (Block 1) ---
        v = self.stages['output'].visuals[0]
        r, cl = v.matrix.shape
        lbl('output', 'Block 1', v.origin, r, cl, v.sp, v)

        # --- Blocks 2-4 ---
        for blk_name, blk_label in [('block_2', 'Block 2'), ('block_3', 'Block 3'), ('block_4', 'Block 4')]:
            if blk_name in self.stages:
                v = self.stages[blk_name].visuals[0]
                r, cl = v.post.shape
                lbl(blk_name, blk_label, v.origin, r, cl, v.sp, v)

        # --- Output Projection ---
        if 'output_projection' in self.stages:
            v = self.stages['output_projection'].visuals[0]
            ra, ca = v.A.shape
            lbl('output_projection', 'Output', v.origin_a, ra, ca, v.sp, v, 'appear')
            rb, cb = v.B.shape
            lbl('output_projection', 'Embed\u1d40', v.origin_b, rb, cb, v.sp_b, v, 'compute')
            rc, cc = v.C.shape
            lbl('output_projection', 'Logits', v.origin_c, rc, cc, v.sp_c, v, 'compute')

        # --- Token Probabilities ---
        if 'token_probs' in self.stages:
            from transformer.vocab import ID_TO_CHAR
            tp_visuals = self.stages['token_probs'].visuals
            v0 = tp_visuals[0]  # group 0: softmax
            v1 = tp_visuals[1]  # group 1: row selection
            v2 = tp_visuals[2]  # group 2: argmax
            r0, c0 = v0.pre.shape

            # Main section labels — raised to make room for column headers
            x_center = v0.origin[0] - (c0 - 1) * v0.sp / 2
            main_y = v0.origin[1] + 2.2
            softmax_pos = np.array([x_center, main_y, v0.origin[2]], dtype=np.float32)
            self.stages['token_probs'].labels.append(
                ('Softmax', softmax_pos, v0, 'appear', False))
            argmax_pos = np.array([x_center, main_y, v2.origin[2]], dtype=np.float32)
            self.stages['token_probs'].labels.append(
                ('argmax', argmax_pos, v2, 'appear', False))

            # Per-column token labels (a, b, ..., z, ., -)
            col_y = v0.origin[1] + 0.8
            for col_idx in range(c0):
                tok = ID_TO_CHAR.get(col_idx, '?')
                if tok == '<END>':
                    tok = '.'
                elif tok == '<PAD>':
                    tok = '-'
                col_pos = np.array([
                    v0.origin[0] - col_idx * v0.sp,
                    col_y,
                    v0.origin[2]
                ], dtype=np.float32)
                self.stages['token_probs'].labels.append(
                    (tok, col_pos, v0, 'appear', True))

        # Per-row prediction labels (to the right of the matrix on screen)
        # Attach to token_probs group 2 if it exists, otherwise to output
        if self.output_labels:
            if 'token_probs' in self.stages:
                target_stage = 'token_probs'
                target_v = self.stages['token_probs'].visuals[2]
                target_origin = target_v.origin
                target_sp = target_v.sp
                target_cols = target_v.pre.shape[1]
            else:
                target_stage = 'output'
                target_v = self.stages['output'].visuals[0]
                target_origin = target_v.origin
                target_sp = target_v.sp
                target_cols = target_v.matrix.shape[1]
            for row_idx, pred_label in enumerate(self.output_labels):
                pos = np.array([
                    target_origin[0] - target_cols * target_sp - 0.8,
                    target_origin[1] - row_idx * target_sp,
                    target_origin[2]
                ], dtype=np.float32)
                self.stages[target_stage].labels.append(
                    (pred_label, pos, target_v, 'appear', False, 'left'))

    def _build_logits_only_labels(self):
        """Build labels for logits_only abbreviated mode."""
        c = self.config

        def lbl(stage_name, text, origin, rows, cols, sp, vis, phase='appear'):
            pos = self._matrix_label_pos(origin, rows, cols, sp)
            self.stages[stage_name].labels.append((text, pos, vis, phase, False))

        # Input: per-row character labels (also sets char fly destinations)
        if 'input' in self.stages and self.input_labels:
            v = self.stages['input'].visuals[0]
            r_inp, cl = v.matrix.shape
            for row_idx, char_label in enumerate(self.input_labels):
                pos = np.array([
                    v.origin[0] - cl * v.sp - 0.8,
                    v.origin[1] - row_idx * v.sp,
                    v.origin[2]
                ], dtype=np.float32)
                self._char_end_positions[row_idx] = pos.copy()
                self.stages['input'].labels.append(
                    (char_label, pos, v, 'appear', True, 'left'))

        # Block stages: label each block
        for blk_name in ('block_1', 'block_2', 'block_3', 'block_4'):
            if blk_name not in self.stages:
                continue
            v = self.stages[blk_name].visuals[0]
            rows, cols = v.matrix.shape
            lbl(blk_name, blk_name.replace('_', ' ').title(),
                v.origin, rows, cols, v.sp, v, 'appear')

        # Output projection label
        if 'output_projection' in self.stages:
            v = self.stages['output_projection'].visuals[0]
            rows, cols = v.matrix.shape
            lbl('output_projection', 'Logits',
                v.origin, rows, cols, v.sp, v, 'appear')

        # Token Probabilities
        if 'token_probs' in self.stages:
            from transformer.vocab import ID_TO_CHAR
            tp_visuals = self.stages['token_probs'].visuals
            v0 = tp_visuals[0]
            v2 = tp_visuals[2]
            r0, c0 = v0.pre.shape

            x_center = v0.origin[0] - (c0 - 1) * v0.sp / 2
            main_y = v0.origin[1] + 2.2
            softmax_pos = np.array([x_center, main_y, v0.origin[2]], dtype=np.float32)
            self.stages['token_probs'].labels.append(
                ('Softmax', softmax_pos, v0, 'appear', False))
            argmax_pos = np.array([x_center, main_y, v2.origin[2]], dtype=np.float32)
            self.stages['token_probs'].labels.append(
                ('argmax', argmax_pos, v2, 'appear', False))

            col_y = v0.origin[1] + 0.8
            for col_idx in range(c0):
                tok = ID_TO_CHAR.get(col_idx, '?')
                if tok == '<PAD>':
                    tok = '-'
                col_pos = np.array([
                    v0.origin[0] - col_idx * v0.sp,
                    col_y,
                    v0.origin[2]
                ], dtype=np.float32)
                self.stages['token_probs'].labels.append(
                    (tok, col_pos, v0, 'appear', True))

        # Per-row prediction labels (to the right of the matrix on screen)
        if self.output_labels and 'token_probs' in self.stages:
            target_v = self.stages['token_probs'].visuals[2]
            target_origin = target_v.origin
            target_sp = target_v.sp
            target_cols = target_v.pre.shape[1]
            for row_idx, pred_label in enumerate(self.output_labels):
                pos = np.array([
                    target_origin[0] - target_cols * target_sp - 0.8,
                    target_origin[1] - row_idx * target_sp,
                    target_origin[2]
                ], dtype=np.float32)
                self.stages['token_probs'].labels.append(
                    (pred_label, pos, target_v, 'appear', False, 'left'))

    def _setup_predicted_char_return(self):
        """Set up the predicted character return animation.

        During the return phase, the existing big characters re-appear at
        their positions (using the NEXT step's spacing so the new character
        fits), and the predicted character flies from the output label to
        the empty last slot — seamlessly completing the sequence.
        """
        r = self.results
        if 'predicted_ids' not in r or 'pred_pos' not in r:
            return
        if 'token_probs' not in self.stages:
            return
        if self.final_display:
            return

        pred_pos = int(r['pred_pos'])
        pred_id = int(r['predicted_ids'][pred_pos])

        from transformer.vocab import ID_TO_CHAR
        pred_char = ID_TO_CHAR.get(pred_id, '?')
        if pred_char == '<PAD>':
            return

        # Start position: the output label on the right side of token_probs
        tp_v = self.stages['token_probs'].visuals[-1]  # argmax visual
        cols = tp_v.pre.shape[1]
        start_pos = np.array([
            tp_v.origin[0] - cols * tp_v.sp - 0.8,
            tp_v.origin[1] - pred_pos * tp_v.sp,
            tp_v.origin[2],
        ], dtype=np.float32)

        # Compute NEXT step's char_display layout (one more character)
        next_n_active = pred_pos + 2  # current active chars + predicted
        next_char_spacing = max(2.0, 4.5 - next_n_active * 0.15)
        next_total_w = (next_n_active - 1) * next_char_spacing
        z_cd = STAGE_Z.get('char_display', STAGE_Z['input'] - 2.0)

        # Existing big characters at their new positions
        existing_chars = []
        for idx, (char, _, _) in enumerate(self._char_display_data):
            pos = np.array([
                next_total_w / 2 - idx * next_char_spacing,
                2, z_cd,
            ], dtype=np.float32)
            existing_chars.append((char, pos))

        # Flying char's end position: the empty last slot
        end_pos = np.array([
            next_total_w / 2 - (next_n_active - 1) * next_char_spacing,
            2, z_cd,
        ], dtype=np.float32)

        # Extract percentage from the output label (e.g. "e 85%" → "85%")
        pct_text = ''
        if self.output_labels and pred_pos < len(self.output_labels):
            label_text = self.output_labels[pred_pos]
            parts = label_text.split(' ', 1)
            if len(parts) > 1:
                pct_text = parts[1]

        self._predicted_char_return = _PredCharReturn(
            char=pred_char,
            start_pos=start_pos,
            end_pos=end_pos,
            pct_text=pct_text,
            existing_chars=existing_chars,
            char_spacing=next_char_spacing,
        )

        # Update camera return waypoint to frame the NEXT step's char layout
        all_positions = [pos for _, pos in existing_chars] + [end_pos]
        position, target, ortho_size = self._frame_char_positions(all_positions)

        if self.camera.waypoints:
            last_wp = self.camera.waypoints[-1]
            self.camera.waypoints[-1] = type(last_wp)(
                time=last_wp.time,
                position=np.array(position, dtype=np.float64),
                target=np.array(target, dtype=np.float64),
                ortho_size=ortho_size,
            )

    def render_labels(self, text_renderer, fb_w, fb_h):
        """Render labels in world space using the camera's view/projection.

        Labels are placed on the XY plane at their 3D positions and
        transformed by the camera like all other geometry, so they
        rotate with the scene and are naturally depth-tested.
        """
        view = self.camera.get_view_matrix()
        aspect = fb_w / max(fb_h, 1)
        proj = self.camera.get_projection_matrix(aspect)

        # Scale labels with camera zoom so they stay readable when zoomed out
        char_height = 0.75 * (self.camera.ortho_size / 10.0)
        # Smaller height for input row chars (right of matrix & char_display landing)
        input_char_height = 0.55 * (self.camera.ortho_size / 10.0)

        # Animated char_display labels (custom rendering with position/size animation)
        self._render_char_display(text_renderer, view, proj, input_char_height)

        # In logits_only mode, keep input characters visible so predicted char
        # can "land" among them
        self._render_persistent_input_chars(text_renderer, view, proj, input_char_height)

        # Predicted character flying back to input during return phase
        self._render_predicted_char_return(text_renderer, view, proj, input_char_height)

        for stage_name, stage in self.stages.items():
            if stage.alpha < 0.01:
                continue
            for idx, label in enumerate(stage.labels):
                text, world_pos, visual, phase, no_bg = label[:5]
                align = label[5] if len(label) > 5 else 'center'
                label_alpha = self._label_fade.get((stage_name, idx), 0.0)
                if label_alpha < 0.01:
                    continue

                bg = (0, 0, 0, 0) if no_bg else (0.0, 0.0, 0.0, 0.92)
                # Input row char labels use smaller height
                h = input_char_height if (stage_name == 'input' and align == 'left') else char_height
                text_renderer.render_text_3d(
                    text, world_pos, view, proj,
                    char_height=h,
                    color=(1.0, 1.0, 1.0, label_alpha * 0.9),
                    bg_color=bg,
                    align=align)

    def _render_char_display(self, text_renderer, view, proj, normal_char_height):
        """Render animated character labels for char_display stage."""
        if not self._char_display_data or 'char_display' not in self.stages:
            return
        stage = self.stages['char_display']
        if stage.alpha < 0.01:
            return
        v = stage.visuals[0]

        big_height = min(4.0 * (self.camera.ortho_size / 10.0),
                         self._char_spacing * 1.4)

        for char, start_pos, row_idx in self._char_display_data:
            end_pos = self._char_end_positions.get(row_idx)

            # Character metrics for alignment correction
            info = text_renderer.char_info.get(char, text_renderer.char_info.get(' '))
            advance_px = info[4] if info else 0

            if v.t > 0.0 and end_pos is not None:
                # Compute phase: fly from start to end, shrink
                t = min(v.t, 1.0)
                t_smooth = t * t * (3.0 - 2.0 * t)  # smoothstep
                pos = start_pos * (1.0 - t_smooth) + end_pos * t_smooth
                h = big_height * (1.0 - t_smooth) + normal_char_height * t_smooth
                alpha = stage.alpha
                # Y: transition from vertically centered (big) to bottom-aligned (small)
                center_offset_y = h * 0.5 * (1.0 - t_smooth)
                # X: transition from center-aligned (big) to left-aligned (matching input labels)
                scale_h = h / max(text_renderer.cell_h, 1)
                center_offset_x = advance_px * scale_h / 2 * (1.0 - t_smooth)
                # Fade out at end for clean handoff to input row labels
                if t > 0.9:
                    alpha *= max(0.0, 1.0 - (t - 0.9) / 0.1)
            else:
                # Appear phase or final display (stay big at start)
                pos = start_pos
                h = big_height
                if self.logits_only or self.final_display or v.t > 0.0:
                    alpha = stage.alpha  # chars already visible from prev scene
                else:
                    alpha = stage.alpha * min(v.appear_t * 2.0, 1.0)
                center_offset_y = h * 0.5
                scale_h = h / max(text_renderer.cell_h, 1)
                center_offset_x = advance_px * scale_h / 2

            if alpha < 0.01:
                continue
            # Text renders upward from origin; offset down to center vertically
            # X offset transitions from centering (big) to left-aligned (matching input labels)
            render_pos = pos.copy()
            render_pos[1] -= center_offset_y
            render_pos[0] += center_offset_x
            text_renderer.render_text_3d(
                char, render_pos, view, proj,
                char_height=h,
                color=(1.0, 1.0, 1.0, alpha * 0.9),
                bg_color=(0, 0, 0, 0),
                align='left')

    def _render_persistent_input_chars(self, text_renderer, view, proj, char_height):
        """In logits_only mode, keep input character labels visible throughout.

        Normally, input labels disappear when the input stage fades during
        block_1's appear. This method renders them independently.
        Hidden during the return phase (big chars shown by
        _render_predicted_char_return instead).
        """
        if not self.logits_only or not self.input_labels:
            return
        if not self._char_end_positions:
            return

        # Only activate once char_display is past compute (settle or done)
        if self.has_char_display:
            cd_phase, _ = self.timeline.get_stage_phase('char_display')
            if cd_phase not in ('settle', 'done'):
                return
        else:
            inp_phase, _ = self.timeline.get_stage_phase('input')
            if inp_phase == 'inactive':
                return

        tl = self.timeline
        in_return = (tl.current_time > tl.total_duration and tl.return_duration > 0)

        # During return phase, big chars are shown by _render_predicted_char_return;
        # hide these small labels to avoid visual clutter
        if in_return and self._predicted_char_return is not None:
            return

        # During return phase, fade at the very end
        alpha = 1.0
        if in_return:
            return_t = (tl.current_time - tl.total_duration) / tl.return_duration
            if return_t > 0.95:
                alpha = max(0.0, 1.0 - (return_t - 0.95) / 0.05)

        if alpha < 0.01:
            return

        for row_idx, char_label in enumerate(self.input_labels):
            if not char_label or char_label in ('<END>',):
                continue
            pos = self._char_end_positions.get(row_idx)
            if pos is None:
                continue
            text_renderer.render_text_3d(
                char_label, pos, view, proj,
                char_height=char_height,
                color=(1.0, 1.0, 1.0, alpha * 0.9),
                bg_color=(0, 0, 0, 0),
                align='left')

    def _render_predicted_char_return(self, text_renderer, view, proj, char_height):
        """Render the char_display with the predicted character flying in.

        During the return phase:
        1. Existing big characters appear fully opaque at their positions
           (using the NEXT step's spacing so the new char fits)
        2. The last slot is empty — the predicted character flies from the
           output label into that slot, growing to big-char size
        3. The percentage text fades at the original output label position
        """
        if self._predicted_char_return is None:
            return
        tl = self.timeline
        if tl.current_time <= tl.total_duration or tl.return_duration <= 0:
            return

        return_t = (tl.current_time - tl.total_duration) / tl.return_duration
        if return_t > 1.0:
            return

        pcr = self._predicted_char_return

        big_height = min(4.0 * (self.camera.ortho_size / 10.0),
                         pcr.char_spacing * 1.4)

        # Helper: compute centering offsets matching _render_char_display
        def _center_offsets(char, h):
            info = text_renderer.char_info.get(char, text_renderer.char_info.get(' '))
            advance_px = info[4] if info else 0
            scale_h = h / max(text_renderer.cell_h, 1)
            return advance_px * scale_h / 2, h * 0.5

        # Existing big characters (match _render_char_display appear-phase alignment)
        for char, char_pos in pcr.existing_chars:
            ox, oy = _center_offsets(char, big_height)
            rp = char_pos.copy()
            rp[0] += ox
            rp[1] -= oy
            text_renderer.render_text_3d(
                char, rp, view, proj,
                char_height=big_height,
                color=(1.0, 1.0, 1.0, 0.9),
                bg_color=(0, 0, 0, 0),
                align='left')

        # Flying character → empty last slot
        fly_t = min(return_t / 0.8, 1.0)
        fly_smooth = fly_t * fly_t * (3.0 - 2.0 * fly_t)  # smoothstep
        pos = pcr.start_pos * (1.0 - fly_smooth) + pcr.end_pos * fly_smooth
        h = char_height * (1.0 - fly_smooth) + big_height * fly_smooth

        # Blend alignment corrections: 0 at start (output label) → full at end
        ox, oy = _center_offsets(pcr.char, h)
        rp = pos.copy()
        rp[0] += ox * fly_smooth
        rp[1] -= oy * fly_smooth
        text_renderer.render_text_3d(
            pcr.char, rp, view, proj,
            char_height=h,
            color=(1.0, 1.0, 1.0, 0.9),
            bg_color=(0, 0, 0, 0),
            align='left')

        # Percentage text fades at original position, offset by "e " prefix width
        if pcr.pct_text:
            pct_alpha = max(0.0, 1.0 - return_t * 4.0)
            if pct_alpha > 0.01:
                prefix_px = text_renderer._measure_text(pcr.char + ' ')
                scale = char_height / max(text_renderer.cell_h, 1)
                pct_pos = pcr.start_pos.copy()
                pct_pos[0] -= prefix_px * scale
                text_renderer.render_text_3d(
                    pcr.pct_text, pct_pos, view, proj,
                    char_height=char_height,
                    color=(1.0, 1.0, 1.0, pct_alpha * 0.7),
                    bg_color=(0, 0, 0, 0),
                    align='left')

    def update(self, dt: float):
        self.timeline.update(dt)
        self.camera.set_time(self.timeline.current_time)
        self.camera.update(dt)
        self._update_animations()
        self._update_label_fades()

    def _update_label_fades(self):
        """Smoothly interpolate label alpha independently from animation state."""
        dt = self.timeline.current_time - self._prev_label_time
        if dt < 0 or dt > 1.0:
            dt = 1.0 / 30.0
        self._prev_label_time = self.timeline.current_time

        FADE_IN = 0.45
        FADE_OUT = 0.35

        for stage_name, stage in self.stages.items():
            for idx, label in enumerate(stage.labels):
                text, world_pos, visual, phase, no_bg = label[:5]
                key = (stage_name, idx)

                # Target: should this label be visible?
                if visual.alpha < 0.01:
                    target = 0.0
                elif phase == 'appear':
                    target = 1.0 if visual.appear_t > 0.05 else 0.0
                else:
                    visible = visual.appear_t > 0.8 and visual.t > 0.02
                    target = 1.0 if visible else 0.0
                    target *= getattr(visual, '_label_output_fade', 1.0)

                target *= min(visual.alpha, 1.0)

                # Suppress input row labels during char_display fly animation;
                # crossfade: chars fade out → labels fade in at the end of compute
                if stage_name == 'input' and self.has_char_display:
                    align = label[5] if len(label) > 5 else 'center'
                    if align == 'left':
                        cd_phase, cd_t = self.timeline.get_stage_phase('char_display')
                        if cd_phase == 'appear' or (cd_phase == 'compute' and cd_t < 0.9):
                            target = 0.0
                        elif cd_phase == 'compute':
                            # Last 10% of compute: crossfade with char_display chars
                            self._label_fade[key] = (cd_t - 0.9) / 0.1
                            continue
                        elif cd_phase == 'settle':
                            # Chars are gone, labels fully visible
                            self._label_fade[key] = 1.0
                            continue

                # Linear interpolation toward target
                current = self._label_fade.get(key, 0.0)
                if target > current:
                    current = min(target, current + dt / FADE_IN)
                else:
                    current = max(target, current - dt / FADE_OUT)
                self._label_fade[key] = current

    def _update_animations(self):
        """Update visual properties based on timeline state.
        Also fades previous stages so the flow effect is clear.

        For output visuals (is_stage_output=True):
          - MatmulVisual/AddVisual: output_alpha_mult=0 hides C instantly,
            while A/B fade gradually via alpha
          - ActivationVisual/StaticMatrixVisual: alpha=0 hides entirely
        Non-output visuals fade gradually via alpha.
        """
        current_idx = self.timeline.get_current_stage_index()
        stage_names = list(self.stages.keys())

        for i, (stage_name, stage) in enumerate(self.stages.items()):
            phase, t = self.timeline.get_stage_phase(stage_name)

            # When has_char_display, input has zero timeline duration;
            # show it during char_display's compute phase instead
            if stage_name == 'input' and self.has_char_display:
                cd_phase, cd_t = self.timeline.get_stage_phase('char_display')
                if cd_phase == 'compute':
                    phase, t = 'appear', cd_t
                elif cd_phase == 'settle':
                    phase, t = 'settle', 1.0
                # else: use timeline's phase ('done' etc.) for normal fade

            stage.update_animation(phase, t)

            # Reset per-frame visual state
            for v in stage.visuals:
                v.output_alpha_mult = 1.0
                v._label_output_fade = 1.0

            # In-place takeover: hide visuals of earlier groups when a later
            # group has started (e.g. softmax replaces scores, LN replaces add)
            if phase in ('compute', 'settle', 'done'):
                num_groups = stage._get_num_groups()
                if num_groups > 1:
                    if phase == 'compute':
                        max_started = 0
                        for v in stage.visuals:
                            g = getattr(v, 'phase_group', 0)
                            seg_start, _ = stage._get_group_segment(g)
                            if t >= seg_start:
                                max_started = max(max_started, g)
                    else:
                        max_started = num_groups - 1
                    if max_started > 0:
                        for v in stage.visuals:
                            g = getattr(v, 'phase_group', 0)
                            if g < max_started:
                                v.output_alpha_mult = 0.0
                                if isinstance(v, ActivationVisual):
                                    # Keep alpha > 0 so label persists;
                                    # visual boxes hidden via output_alpha_mult=0
                                    v._label_output_fade = 1.0
                                elif isinstance(v, StaticMatrixVisual):
                                    # StaticMatrixVisual has no separate A/B/C —
                                    # output_alpha_mult=0 hides it completely.
                                    # Hide via alpha instead (immediately).
                                    v.output_alpha_mult = 1.0
                                    v.alpha = 0.0
                                    v._label_output_fade = 0.0
                                elif phase == 'compute':
                                    next_start, next_end = stage._get_group_segment(g + 1)
                                    next_seg_len = next_end - next_start
                                    next_local = (t - next_start) / max(next_seg_len, 1e-6)
                                    if v.fade_inputs_on_takeover:
                                        # Q_h/K_h^T: fade A/B over first 20%
                                        # C stays hidden (output_alpha_mult=0)
                                        v.alpha = max(0.0, 1.0 - min(next_local / 0.20, 1.0))
                                    else:
                                        v.alpha = 0.0
                                    v._label_output_fade = max(
                                        0.0, 1.0 - min(next_local / 0.20, 1.0))
                                else:
                                    v.alpha = 0.0
                                    v._label_output_fade = 0.0

            # Fade done stages: previous stage fades during next active stage's appear
            if phase == 'done' and i < current_idx:
                # Find next stage with nonzero duration (skip zero-duration stages)
                fade = 0.0
                for ni in range(i + 1, len(stage_names)):
                    ns_name = stage_names[ni]
                    ns_tl = self.timeline.get_stage(ns_name)
                    if ns_tl and ns_tl.total_duration > 0:
                        ns_phase, ns_t = self.timeline.get_stage_phase(ns_name)
                        if ns_phase == 'appear':
                            fade = max(0.0, 1.0 - ns_t * 4.0)
                        break
                stage.alpha = fade
                for v in stage.visuals:
                    if v.is_stage_output:
                        v.output_alpha_mult = 0.0
                    if isinstance(v, StaticMatrixVisual):
                        # StaticMatrixVisual has no A/B/C split — hide
                        # immediately so it doesn't overlap with the
                        # next stage's visual flying from the same spot.
                        v.alpha = 0.0
                    else:
                        # min() preserves alpha=0 from in-place takeover
                        v.alpha = min(v.alpha, fade)

        # Return-to-start: fade all stages out
        if (self.timeline.current_time > self.timeline.total_duration
                and self.timeline.return_duration > 0):
            return_t = ((self.timeline.current_time - self.timeline.total_duration)
                        / self.timeline.return_duration)
            fade = max(0.0, 1.0 - return_t * 2.0)
            for sname, stage in self.stages.items():
                # When predicted char flies, token_probs fades quickly
                # so the output label doesn't duplicate the flying char
                if (self._predicted_char_return is not None
                        and sname == 'token_probs'):
                    sf = max(0.0, 1.0 - return_t * 5.0)
                else:
                    sf = fade
                stage.alpha *= sf
                for v in stage.visuals:
                    v.alpha *= sf

        # Hide char_display boxes (labels-only stage, rendered by _render_char_display)
        if 'char_display' in self.stages:
            for v in self.stages['char_display'].visuals:
                v.output_alpha_mult = 0.0

    def render(self, aspect: float):
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(aspect)

        self.shader.use()
        self.shader.set_mat4("u_view", view)
        self.shader.set_mat4("u_projection", proj)
        self.shader.set_vec3("u_light_dir", np.array([0.3, -0.8, -0.5], dtype=np.float32))
        self.shader.set_vec3("u_camera_pos", self.camera.position)

        for stage in self.stages.values():
            if stage.alpha < 0.01:
                continue

            data = stage.get_all_instance_data()
            count = data.shape[0]
            if count > 0:
                max_inst = self.renderer.MAX_INSTANCES
                for offset in range(0, count, max_inst):
                    chunk = data[offset:offset + max_inst]
                    self.renderer.draw(chunk, chunk.shape[0])
