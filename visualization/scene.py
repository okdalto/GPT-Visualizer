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
                    local_t = (t - seg_start) / (seg_end - seg_start)
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


class Scene:
    def __init__(self, results: dict, config: TransformerConfig, shader,
                 aspect: float = 16.0 / 9.0):
        self.results = results
        self.config = config
        self.aspect = aspect
        self.timeline = AnimationTimeline()
        self.camera = Camera()
        self.renderer = InstancedBoxRenderer(shader)
        self.shader = shader

        self.stages: dict[str, Stage] = {}
        self._build_all_stages()
        # Copy per-group segment boundaries from timeline to stages
        for tl_stage in self.timeline.stages:
            if tl_stage.stage_name in self.stages:
                self.stages[tl_stage.stage_name].group_segments = tl_stage.group_segments
        self._wire_flow_connections()
        self._build_camera_path()
        self._build_labels()

        # Smooth label fade state
        self._label_fade = {}   # (stage_name, idx) -> current alpha
        self._prev_label_time = 0.0

    def _build_all_stages(self):
        r = self.results
        c = self.config
        bs = 0.4
        gap = 0.1

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
        # Stage 8: Output
        # =========================================================
        stage = Stage('output')
        z = STAGE_Z['output']
        w_out = c.d_model * SPACING
        v = StaticMatrixVisual(r['output'], matrix_origin(z, -w_out / 2, 2), bs, gap)
        v.phase_group = 0
        stage.visuals.append(v)
        self.stages['output'] = stage

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

    def _wire_flow_connections(self):
        """Wire up from_origin on each visual so data flies from the previous stage/sub-op."""
        s = self.stages

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
            q_slice_origin[0] += h * d_k * SPACING
            mha[base].from_origin_a = q_slice_origin

            # K_h^T (B of sub-op 1): transpose changes shape (8×4 → 4×8),
            # animate the transpose by flying each element from [c,r] → [r,c]
            k_slice_origin = qkv_K.origin_c.copy()
            k_slice_origin[0] += h * d_k * SPACING
            mha[base].from_origin_b = k_slice_origin
            mha[base].transpose_fly_b = True

            # Sub-op 2 (softmax): data from Scores result position
            mha[base + 1].from_origin = mha[base].origin_c.copy()

            # Sub-op 3 (Weights × V_h): Weights stay at softmax position (no fly-in)
            mha[base + 2].from_origin_a = mha[base + 1].origin.copy()
            mha[base + 2].is_stage_output = True  # Head output goes to concat

            # V_h (B of sub-op 3): flies from V output, head h's slice
            v_slice_origin = qkv_V.origin_c.copy()
            v_slice_origin[0] += h * d_k * SPACING
            mha[base + 2].from_origin_b = v_slice_origin

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

    def _compute_stage_view_bounds(self, stage_name, cam_right, cam_up):
        """Compute tight bounding box in view space (cam_right / cam_up).

        Projects each visual's actual corner points into view space,
        giving tighter bounds than projecting world-AABB corners.
        Returns (vx_min, vx_max, vy_min, vy_max).
        """
        stage = self.stages[stage_name]
        vx_min = np.inf
        vx_max = -np.inf
        vy_min = np.inf
        vy_max = -np.inf

        for v in stage.visuals:
            for origin, rows, cols, sp, bs in self._get_visual_extents(v):
                x0 = origin[0] - bs / 2
                x1 = origin[0] + (cols - 1) * sp + bs / 2
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

    def _build_camera_path(self):
        """Set up camera waypoints with ortho framing from 45° upper-left.

        Uses view-space bounding boxes with a fixed additive margin so
        every stage has identical margins regardless of content size.
        """
        azimuth = np.pi / 4
        elevation = np.pi / 4

        cam_offset_dir = np.array([
            -np.sin(azimuth) * np.cos(elevation),
            np.sin(elevation),
            -np.cos(azimuth) * np.cos(elevation),
        ])
        cam_offset_dir /= np.linalg.norm(cam_offset_dir)

        cam_fwd = -cam_offset_dir
        world_up = np.array([0.0, 1.0, 0.0])
        cam_right = np.cross(cam_fwd, world_up)
        cam_right /= np.linalg.norm(cam_right)
        cam_up = np.cross(cam_right, cam_fwd)

        cam_dist = 200.0
        assumed_aspect = self.aspect
        margin = 2.0  # fixed view-space margin (same for every stage)

        def compute_framing(stage_name):
            vx_min, vx_max, vy_min, vy_max = \
                self._compute_stage_view_bounds(stage_name, cam_right, cam_up)

            # View-space center → world-space target
            vcx = (vx_min + vx_max) / 2.0
            vcy = (vy_min + vy_max) / 2.0
            # Depth component: use average forward position of visuals
            stage = self.stages[stage_name]
            fwd_sum, fwd_cnt = 0.0, 0
            for v in stage.visuals:
                for origin, *_ in self._get_visual_extents(v):
                    fwd_sum += np.dot(origin, cam_fwd)
                    fwd_cnt += 1
            fwd_center = fwd_sum / max(fwd_cnt, 1)

            target = vcx * cam_right + vcy * cam_up + fwd_center * cam_fwd
            position = target + cam_offset_dir * cam_dist

            # Additive margin: same absolute amount for all stages
            half_w = (vx_max - vx_min) / 2.0 + margin
            half_h = (vy_max - vy_min) / 2.0 + margin
            ortho_size = max(half_h, half_w / assumed_aspect)
            return position, target, ortho_size

        for tl_stage in self.timeline.stages:
            pos, tgt, osz = compute_framing(tl_stage.stage_name)

            t_arrive = tl_stage.start_time + tl_stage.appear_duration * 0.5
            self.camera.add_waypoint(t_arrive, pos, tgt, osz)
            self.camera.add_waypoint(tl_stage.end_time, pos, tgt, osz)

        first_pos, first_tgt, first_osz = compute_framing(
            self.timeline.stages[0].stage_name)
        loop_end = self.timeline.total_duration + self.timeline.return_duration
        self.camera.add_waypoint(loop_end, first_pos, first_tgt, first_osz)

    def _matrix_label_pos(self, origin, rows, cols, sp):
        """Position above the center-top of a matrix."""
        x = origin[0] + (cols - 1) * sp / 2
        y = origin[1] + 1.2
        z = origin[2]
        return np.array([x, y, z], dtype=np.float32)

    def _build_labels(self):
        """Create labels for every matrix in every stage.

        Each label is (text, world_pos, visual, phase) where phase is:
          'appear'  – fades in with visual.appear_t  (inputs that fly in)
          'compute' – fades in with visual.t          (results that emerge)
        """
        c = self.config

        def lbl(stage_name, text, origin, rows, cols, sp, vis, phase='appear'):
            pos = self._matrix_label_pos(origin, rows, cols, sp)
            self.stages[stage_name].labels.append((text, pos, vis, phase))

        # --- Input ---
        v = self.stages['input'].visuals[0]
        r, cl = v.matrix.shape
        lbl('input', 'X', v.origin, r, cl, v.sp, v)

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

        # --- Output ---
        v = self.stages['output'].visuals[0]
        r, cl = v.matrix.shape
        lbl('output', 'Output', v.origin, r, cl, v.sp, v)

    def render_labels(self, text_renderer, fb_w, fb_h):
        """Render labels in world space using the camera's view/projection.

        Labels are placed on the XY plane at their 3D positions and
        transformed by the camera like all other geometry, so they
        rotate with the scene and are naturally depth-tested.
        """
        view = self.camera.get_view_matrix()
        aspect = fb_w / max(fb_h, 1)
        proj = self.camera.get_projection_matrix(aspect)

        for stage_name, stage in self.stages.items():
            if stage.alpha < 0.01:
                continue
            for idx, (text, world_pos, visual, phase) in enumerate(stage.labels):
                label_alpha = self._label_fade.get((stage_name, idx), 0.0)
                if label_alpha < 0.01:
                    continue

                text_renderer.render_text_3d(
                    text, world_pos, view, proj,
                    char_height=0.75,
                    color=(1.0, 1.0, 1.0, label_alpha * 0.9))

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
            for idx, (text, world_pos, visual, phase) in enumerate(stage.labels):
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
                                # Smooth label crossfade based on next group's
                                # appear progress (first 20% of its segment)
                                if phase == 'compute':
                                    next_start, next_end = stage._get_group_segment(g + 1)
                                    next_seg_len = next_end - next_start
                                    next_local = (t - next_start) / max(next_seg_len, 1e-6)
                                    v._label_output_fade = max(
                                        0.0, 1.0 - min(next_local / 0.20, 1.0))
                                else:
                                    v._label_output_fade = 0.0

            # Fade done stages: previous stage fades during next stage's appear
            if phase == 'done' and i < current_idx:
                distance = current_idx - i
                if distance == 1:
                    next_name = stage_names[min(i + 1, len(stage_names) - 1)]
                    next_phase, next_t = self.timeline.get_stage_phase(next_name)
                    if next_phase == 'appear':
                        fade = max(0.0, 1.0 - next_t)
                    else:
                        fade = 0.0
                else:
                    fade = 0.0
                stage.alpha = fade
                for v in stage.visuals:
                    if v.is_stage_output:
                        v.output_alpha_mult = 0.0
                        v.alpha = fade
                    else:
                        v.alpha = fade

        # Return-to-start: fade all stages out
        if self.timeline.current_time > self.timeline.total_duration:
            return_t = ((self.timeline.current_time - self.timeline.total_duration)
                        / self.timeline.return_duration)
            fade = max(0.0, 1.0 - return_t * 2.0)
            for stage in self.stages.values():
                stage.alpha *= fade
                for v in stage.visuals:
                    v.alpha *= fade

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
