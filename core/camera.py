import numpy as np
from dataclasses import dataclass
from animation.easing import smoothstep


@dataclass
class CameraWaypoint:
    time: float
    position: np.ndarray
    target: np.ndarray
    ortho_size: float = 10.0


class Camera:
    def __init__(self):
        self.position = np.array([0.0, 5.0, -10.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.near = 0.1
        self.far = 1000.0
        self.ortho_size = 10.0

        self.waypoints: list[CameraWaypoint] = []
        self.current_time = 0.0

        self.user_override = False
        self.override_yaw = 0.0
        self.override_pitch = 0.0
        self.override_zoom = 0.0
        self.override_timeout = 0.0

    def add_waypoint(self, time: float, position, target, ortho_size: float = 10.0):
        self.waypoints.append(CameraWaypoint(
            time=time,
            position=np.array(position, dtype=np.float64),
            target=np.array(target, dtype=np.float64),
            ortho_size=ortho_size,
        ))
        self.waypoints.sort(key=lambda w: w.time)

    def set_time(self, t: float):
        self.current_time = t
        self._update_from_path()

    def update(self, dt: float):
        if self.user_override:
            self.override_timeout -= dt
            if self.override_timeout <= 0:
                self.user_override = False
                self.override_yaw = 0.0
                self.override_pitch = 0.0
                self.override_zoom = 0.0

    def handle_mouse(self, dx: float, dy: float):
        self.user_override = True
        self.override_timeout = 3.0
        self.override_yaw += dx * 0.003
        self.override_pitch += dy * 0.003
        self.override_pitch = np.clip(self.override_pitch, -1.2, 1.2)

    def handle_scroll(self, xoff: float, yoff: float):
        self.user_override = True
        self.override_timeout = 3.0
        self.override_zoom += yoff * 0.1

    def _update_from_path(self):
        if len(self.waypoints) < 2:
            if self.waypoints:
                self.position = self.waypoints[0].position.astype(np.float32).copy()
                self.target = self.waypoints[0].target.astype(np.float32).copy()
                self.ortho_size = self.waypoints[0].ortho_size
            return

        t = self.current_time

        if t <= self.waypoints[0].time:
            self.position = self.waypoints[0].position.astype(np.float32).copy()
            self.target = self.waypoints[0].target.astype(np.float32).copy()
            self.ortho_size = self.waypoints[0].ortho_size
            return
        if t >= self.waypoints[-1].time:
            self.position = self.waypoints[-1].position.astype(np.float32).copy()
            self.target = self.waypoints[-1].target.astype(np.float32).copy()
            self.ortho_size = self.waypoints[-1].ortho_size
            return

        # Find segment and interpolate with smoothstep (no overshoot)
        for i in range(len(self.waypoints) - 1):
            if self.waypoints[i].time <= t <= self.waypoints[i + 1].time:
                dt_seg = self.waypoints[i + 1].time - self.waypoints[i].time
                if dt_seg < 1e-9:
                    self.position = self.waypoints[i + 1].position.astype(np.float32).copy()
                    self.target = self.waypoints[i + 1].target.astype(np.float32).copy()
                    self.ortho_size = self.waypoints[i + 1].ortho_size
                    break
                seg_t = (t - self.waypoints[i].time) / dt_seg
                s = smoothstep(0.0, 1.0, seg_t)

                p0 = self.waypoints[i].position
                p1 = self.waypoints[i + 1].position
                t0 = self.waypoints[i].target
                t1 = self.waypoints[i + 1].target
                o0 = self.waypoints[i].ortho_size
                o1 = self.waypoints[i + 1].ortho_size

                self.position = (p0 * (1.0 - s) + p1 * s).astype(np.float32)
                self.target = (t0 * (1.0 - s) + t1 * s).astype(np.float32)
                self.ortho_size = o0 * (1.0 - s) + o1 * s
                break

        # Apply user override
        if self.user_override:
            direction = self.position - self.target
            dist = float(np.linalg.norm(direction))

            base_yaw = float(np.arctan2(direction[0], direction[2]))
            d_norm = float(np.linalg.norm(direction))
            base_pitch = float(np.arcsin(np.clip(direction[1] / max(d_norm, 1e-6), -1, 1)))
            yaw = base_yaw + self.override_yaw
            pitch = np.clip(base_pitch + self.override_pitch, -1.2, 1.2)

            self.position = self.target + np.array([
                np.sin(yaw) * np.cos(pitch) * dist,
                np.sin(pitch) * dist,
                np.cos(yaw) * np.cos(pitch) * dist,
            ], dtype=np.float32)

            # Ortho zoom: scroll up zooms in (smaller ortho_size)
            self.ortho_size *= 2.0 ** (-self.override_zoom)

    def get_view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        half_h = max(0.1, self.ortho_size)
        half_w = half_h * aspect
        return ortho(half_w, half_h, self.near, self.far)


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        # Camera looking straight up/down, use alternative up
        up2 = np.array([1.0, 0.0, 0.0])
        s = np.cross(f, up2)
        s_norm = np.linalg.norm(s)
    s = s / s_norm
    u = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, 0] = s[0];  m[0, 1] = s[1];  m[0, 2] = s[2]
    m[1, 0] = u[0];  m[1, 1] = u[1];  m[1, 2] = u[2]
    m[2, 0] = -f[0]; m[2, 1] = -f[1]; m[2, 2] = -f[2]
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def ortho(half_w, half_h, near, far):
    """Symmetric orthographic projection matrix."""
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = 1.0 / half_w
    m[1, 1] = 1.0 / half_h
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m
