import ctypes
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from OpenGL import GL as gl
from core.shader import ShaderProgram


STAGE_DISPLAY_NAMES = {
    'char_display':       'Input Characters',
    'input':              'Input Embeddings',
    'qkv_projection':     'QKV Projection',
    'multi_head_attn':    'Multi-Head Attention',
    'concat_output_proj': 'Concat + Output Projection',
    'residual_ln1':       'Residual + LayerNorm 1',
    'ffn':                'Feed-Forward Network',
    'residual_ln2':       'Residual + LayerNorm 2',
    'output':             'Block 1 Output',
    'block_1':            'Block 1',
    'block_2':            'Block 2',
    'block_3':            'Block 3',
    'block_4':            'Block 4',
    'output_projection':  'Vocabulary Projection',
    'token_probs':        'Token Probabilities',
}

MAX_CHARS = 256


class TextRenderer:
    def __init__(self, vert_path: str, frag_path: str, font_size: int = 32):
        self.shader = ShaderProgram(vert_path, frag_path)
        self.font_size = font_size
        self._build_font_atlas(font_size)
        self._setup_gl()

    def _build_font_atlas(self, font_size: int):
        font = ImageFont.load_default(size=font_size)

        # Use font metrics for cell height (ascent + descent covers all glyphs)
        chars = [chr(c) for c in range(32, 127)]
        ascent, descent = font.getmetrics()

        max_w = 0
        for ch in chars:
            bbox = font.getbbox(ch)
            max_w = max(max_w, bbox[2] - bbox[0])

        pad = 4
        self.cell_w = max_w + pad
        self.cell_h = ascent + descent + pad
        cols = math.ceil(math.sqrt(len(chars)))
        rows = math.ceil(len(chars) / cols)

        atlas_w = cols * self.cell_w
        atlas_h = rows * self.cell_h
        img = Image.new('L', (atlas_w, atlas_h), 0)
        draw = ImageDraw.Draw(img)

        self.char_info = {}  # char -> (u0, v0, u1, v1, advance_px)
        for idx, ch in enumerate(chars):
            col = idx % cols
            row = idx // cols
            x = col * self.cell_w + pad // 2
            y = row * self.cell_h + pad // 2
            # anchor='la' positions at left-ascender for consistent baseline
            draw.text((x, y), ch, fill=255, font=font, anchor='la')

            bbox = font.getbbox(ch)
            advance = bbox[2] - bbox[0] + 1
            # render_width: pixels from draw position to just past ink right edge.
            # Covers the full glyph even when bbox[0] > 0 (left bearing).
            render_width = bbox[2] + 2

            u0 = x / atlas_w
            v0 = y / atlas_h
            u1 = (x + self.cell_w - pad) / atlas_w
            v1 = (y + self.cell_h - pad) / atlas_h
            self.char_info[ch] = (u0, v0, u1, v1, advance, render_width)

        # Reserve a solid-white 1x1 pixel at top-left corner for background quads
        img.putpixel((0, 0), 255)
        self.solid_uv = (0.5 / atlas_w, 0.5 / atlas_h)

        # Flip vertically for OpenGL (origin at bottom-left)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        atlas_data = np.ascontiguousarray(np.array(img, dtype=np.uint8))

        # After vertical flip, the solid pixel moves from top-left to bottom-left
        self.solid_uv = (0.5 / atlas_w, 1.0 - 0.5 / atlas_h)

        # Upload to GPU - must set alignment to 1 for single-channel textures
        self.atlas_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_R8,
            atlas_w, atlas_h, 0,
            gl.GL_RED, gl.GL_UNSIGNED_BYTE, atlas_data
        )
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)  # restore default
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.atlas_w = atlas_w
        self.atlas_h = atlas_h

    def _setup_gl(self):
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)

        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            MAX_CHARS * 6 * 5 * 4,  # 6 verts * 5 floats * 4 bytes
            None, gl.GL_DYNAMIC_DRAW
        )

        stride = 5 * 4  # 20 bytes: vec3 position + vec2 texcoord
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))

        gl.glBindVertexArray(0)

    def destroy(self):
        gl.glDeleteBuffers(1, [self.vbo])
        gl.glDeleteVertexArrays(1, [self.vao])
        gl.glDeleteTextures(1, [self.atlas_texture])
        self.shader.destroy()

    def _ortho_2d(self, width: float, height: float) -> np.ndarray:
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = 2.0 / width
        m[1, 1] = 2.0 / height
        m[0, 3] = -1.0
        m[1, 3] = -1.0
        m[3, 3] = 1.0
        return m

    def _build_quads(self, text: str, x: float, y: float):
        """Build 2D screen-space quads (z=0) for overlay text."""
        verts = []
        cx = x
        for ch in text:
            info = self.char_info.get(ch)
            if info is None:
                info = self.char_info.get(' ')
                if info is None:
                    continue
            u0, v0_atlas, u1, v1_atlas = info[0], info[1], info[2], info[3]
            advance = info[4]

            # Flip V coords since atlas was flipped
            v0 = 1.0 - v1_atlas
            v1 = 1.0 - v0_atlas

            x0 = cx
            y0 = y
            x1 = cx + self.cell_w - 2
            y1 = y + self.cell_h - 2

            # Two triangles per character quad (x, y, z=0, u, v)
            verts.extend([
                x0, y0, 0.0, u0, v0,
                x1, y0, 0.0, u1, v0,
                x1, y1, 0.0, u1, v1,
                x0, y0, 0.0, u0, v0,
                x1, y1, 0.0, u1, v1,
                x0, y1, 0.0, u0, v1,
            ])
            cx += advance

        return np.array(verts, dtype=np.float32), len(text) * 6

    def _build_quads_3d(self, text: str, origin, char_height: float = 0.5,
                        align: str = 'center'):
        """Build 3D world-space quads on the XY plane at origin.

        Characters progress in -X direction (high X → low X) because the
        camera views from azimuth π/4 where world +X points camera-left.
        U coordinates are swapped to correct the horizontal mirroring.

        align='center': text centered at origin (default)
        align='left':   text left-edge (screen-left = high X) at origin
        """
        scale = char_height / max(self.cell_h, 1)

        # Measure total width for centering
        total_px_width = 0.0
        for ch in text:
            info = self.char_info.get(ch, self.char_info.get(' '))
            if info:
                total_px_width += info[4]
        total_world_width = total_px_width * scale

        verts = []
        # Start at high X (camera-left) and advance toward low X (camera-right)
        if align == 'left':
            cx = origin[0]  # anchor at screen-left (high X)
        else:
            cx = origin[0] + total_world_width / 2
        y_base = origin[1]
        z = origin[2]

        for ch in text:
            info = self.char_info.get(ch)
            if info is None:
                info = self.char_info.get(' ')
                if info is None:
                    continue
            u0, v0_atlas, u1, v1_atlas = info[0], info[1], info[2], info[3]
            advance = info[4]
            render_width = info[5]

            v0 = 1.0 - v1_atlas
            v1 = 1.0 - v0_atlas

            # Quad width uses render_width (covers full glyph including bearing)
            char_w = render_width * scale
            u1_glyph = u0 + render_width / self.atlas_w

            x0 = cx - char_w  # low X = camera-right = glyph right
            x1 = cx            # high X = camera-left = glyph left
            y0 = y_base
            y1 = y_base + (self.cell_h - 2) * scale

            # Swap u0/u1: x1 (camera-left) gets u0 (glyph-left),
            #              x0 (camera-right) gets u1 (glyph-right)
            verts.extend([
                x0, y0, z, u1_glyph, v0,
                x1, y0, z, u0, v0,
                x1, y1, z, u0, v1,
                x0, y0, z, u1_glyph, v0,
                x1, y1, z, u0, v1,
                x0, y1, z, u1_glyph, v1,
            ])
            cx -= advance * scale

        return np.array(verts, dtype=np.float32), len(text) * 6

    def _build_bg_quad_3d(self, origin, text_world_width, char_height,
                          padding: float = 0.15, align: str = 'center'):
        """Build a semi-transparent background quad behind 3D text."""
        su, sv = self.solid_uv
        h = char_height + padding * 0.6

        # Background extends slightly beyond text bounds
        # X reversed: high X = camera-left, low X = camera-right
        if align == 'left':
            x0 = origin[0] - text_world_width - padding
            x1 = origin[0] + padding
        else:
            hw = text_world_width / 2 + padding
            x0 = origin[0] - hw
            x1 = origin[0] + hw
        y0 = origin[1] - padding * 0.3
        y1 = origin[1] + h
        z = origin[2]

        verts = np.array([
            x0, y0, z, su, sv,
            x1, y0, z, su, sv,
            x1, y1, z, su, sv,
            x0, y0, z, su, sv,
            x1, y1, z, su, sv,
            x0, y1, z, su, sv,
        ], dtype=np.float32)
        return verts

    def render_text(self, text: str, x: float, y: float,
                    fb_width: int, fb_height: int,
                    color=(1.0, 1.0, 1.0, 1.0),
                    depth=None):
        """Render text at screen position (2D overlay).

        Args:
            depth: If provided (0..1), enable depth testing and write this
                   depth value so labels can be occluded by 3D geometry.
                   If None, render as a 2D overlay (no depth test).
        """
        if not text:
            return

        vertex_data, num_verts = self._build_quads(text, x, y)
        if num_verts == 0:
            return

        proj = self._ortho_2d(fb_width, fb_height)
        view = np.eye(4, dtype=np.float32)

        prev_depth_test = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        if depth is not None:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthMask(gl.GL_FALSE)  # read-only: don't write to depth buffer
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.shader.use()
        self.shader.set_mat4("u_projection", proj)
        self.shader.set_mat4("u_view", view)
        self.shader.set_int("u_font_atlas", 0)
        self.shader.set_vec4("u_text_color", np.array(color, dtype=np.float32))
        self.shader.set_float("u_depth", depth if depth is not None else -1.0)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, num_verts)
        gl.glBindVertexArray(0)

        if depth is not None:
            gl.glDepthMask(gl.GL_TRUE)  # restore depth writing
        if prev_depth_test:
            gl.glEnable(gl.GL_DEPTH_TEST)
        else:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def render_text_3d(self, text: str, origin, view, proj,
                       char_height: float = 0.5,
                       color=(1.0, 1.0, 1.0, 1.0),
                       bg_color=(0.0, 0.0, 0.0, 0.92),
                       align: str = 'center'):
        """Render text in world space on the XY plane at origin.

        Text is transformed by view and projection matrices like 3D geometry,
        so it rotates with the camera and is naturally depth-tested.
        Draws a semi-transparent background quad behind the text.

        align='center': text centered at origin (default)
        align='left':   text left-edge (screen-left = high X) at origin
        """
        if not text:
            return

        vertex_data, num_verts = self._build_quads_3d(text, origin, char_height,
                                                       align=align)
        if num_verts == 0:
            return

        # Compute text width for background sizing
        scale = char_height / max(self.cell_h, 1)
        total_px_width = 0.0
        for ch in text:
            info = self.char_info.get(ch, self.char_info.get(' '))
            if info:
                total_px_width += info[4]
        text_world_width = total_px_width * scale

        # Push background slightly behind text (further from camera).
        # Extract camera forward from view matrix row 2; offset along it.
        # In ortho projection this changes depth only, not screen position.
        fwd = -np.array([view[2, 0], view[2, 1], view[2, 2]], dtype=np.float32)
        bg_origin = np.array(origin, dtype=np.float32) + fwd * 0.15
        bg_data = self._build_bg_quad_3d(bg_origin, text_world_width, char_height,
                                         align=align)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)  # read-only: don't write to depth buffer
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.shader.use()
        self.shader.set_mat4("u_projection", proj)
        self.shader.set_mat4("u_view", view)
        self.shader.set_int("u_font_atlas", 0)
        self.shader.set_float("u_depth", -1.0)  # use natural depth from 3D transform

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        # Draw background quad first
        bg_alpha = bg_color[3] * color[3]  # modulate by label fade alpha
        self.shader.set_vec4("u_text_color",
                             np.array([bg_color[0], bg_color[1], bg_color[2], bg_alpha],
                                      dtype=np.float32))
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, bg_data.nbytes, bg_data)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # Draw text on top
        self.shader.set_vec4("u_text_color", np.array(color, dtype=np.float32))
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, vertex_data.nbytes, vertex_data)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, num_verts)

        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)

    def _measure_text(self, text: str) -> float:
        width = 0.0
        for ch in text:
            info = self.char_info.get(ch, self.char_info.get(' '))
            if info:
                width += info[4]
        return width

    def render_stage_name(self, stage_name: str, fb_width: int, fb_height: int):
        display_name = STAGE_DISPLAY_NAMES.get(stage_name, stage_name)
        text_width = self._measure_text(display_name)

        x = fb_width - text_width - 26
        y = 26

        # Semi-transparent dark background
        bg_x = x - 10
        bg_y = y - 6
        bg_w = text_width + 20
        bg_h = self.cell_h + 4
        self._render_bg_quad(bg_x, bg_y, bg_w, bg_h, fb_width, fb_height,
                             color=(0.0, 0.0, 0.0, 0.6))

        # White text
        self.render_text(display_name, x, y, fb_width, fb_height,
                         color=(1.0, 1.0, 1.0, 0.95))

    def _render_bg_quad(self, x: float, y: float, w: float, h: float,
                        fb_width: int, fb_height: int,
                        color=(0.0, 0.0, 0.0, 0.6)):
        # Use the solid-white pixel UV so the quad is fully opaque in the atlas
        su, sv = self.solid_uv

        verts = np.array([
            x, y, 0.0, su, sv,
            x + w, y, 0.0, su, sv,
            x + w, y + h, 0.0, su, sv,
            x, y, 0.0, su, sv,
            x + w, y + h, 0.0, su, sv,
            x, y + h, 0.0, su, sv,
        ], dtype=np.float32)

        proj = self._ortho_2d(fb_width, fb_height)
        view = np.eye(4, dtype=np.float32)

        prev_depth = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_DEPTH_TEST)

        self.shader.use()
        self.shader.set_mat4("u_projection", proj)
        self.shader.set_mat4("u_view", view)
        self.shader.set_int("u_font_atlas", 0)
        self.shader.set_vec4("u_text_color", np.array(color, dtype=np.float32))

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.atlas_texture)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, verts.nbytes, verts)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
        gl.glBindVertexArray(0)

        if prev_depth:
            gl.glEnable(gl.GL_DEPTH_TEST)
