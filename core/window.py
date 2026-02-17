import glfw
from OpenGL import GL as gl


class AppWindow:
    def __init__(self, width=1600, height=900, title="GPT Visualizer"):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.SAMPLES, 4)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        gl.glViewport(0, 0, fb_w, fb_h)

        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)

        self._key_callbacks = []
        self._mouse_callbacks = []
        self._scroll_callbacks = []
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)

        self.mouse_pressed = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

    def _framebuffer_size_callback(self, window, width, height):
        gl.glViewport(0, 0, width, height)

    def _key_callback(self, window, key, scancode, action, mods):
        for cb in self._key_callbacks:
            cb(key, scancode, action, mods)

    def _cursor_pos_callback(self, window, xpos, ypos):
        dx = xpos - self.last_mouse_x
        dy = ypos - self.last_mouse_y
        self.last_mouse_x = xpos
        self.last_mouse_y = ypos
        if self.mouse_pressed:
            for cb in self._mouse_callbacks:
                cb(dx, dy)

    def _mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = (action == glfw.PRESS)

    def _scroll_callback(self, window, xoffset, yoffset):
        for cb in self._scroll_callbacks:
            cb(xoffset, yoffset)

    def add_key_callback(self, cb):
        self._key_callbacks.append(cb)

    def add_mouse_callback(self, cb):
        self._mouse_callbacks.append(cb)

    def add_scroll_callback(self, cb):
        self._scroll_callbacks.append(cb)

    def get_framebuffer_size(self):
        return glfw.get_framebuffer_size(self.window)

    def should_close(self):
        return glfw.window_should_close(self.window)

    def swap_and_poll(self):
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def terminate(self):
        glfw.terminate()
