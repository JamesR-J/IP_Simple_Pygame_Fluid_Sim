import numpy as np

from pygame import gfxdraw
from math import atan2, cos, sin, inf, sqrt


def clamp(val, low=None, high=None):
    # Return value no lower than low and no higher than high
    minimum = -inf if low is None else low
    maximum = inf if high is None else high
    return max(min(val, maximum), minimum)


def dist(x1, y1, x2, y2, less_than=None, greater_than=None):
    # Use less_than or greater_than avoid using costly square root function
    if less_than is not None:
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) < less_than ** 2
    if greater_than is not None:
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) > greater_than ** 2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# # def overlapping_particle(x, y, r):
# #        # Returns an overlapping particle from particles or None if not touching any
# #        for particle in particles:
# #            if dist(x, y, particle.x, particle.y, less_than=r + particle.r):
# #                return particle


def draw_circle(surface, x, y, radius, color):
    # Draw anti-aliased circle
    gfxdraw.aacircle(surface, int(x), int(y), radius, color)
    gfxdraw.filled_circle(surface, int(x), int(y), radius, color)


def particle_bounce_velocities(p1, p2):
    # Change particle velocities to make them bounce
    # Based off github.com/xnx/collision/blob/master/collision.py lines 148-156
    from parameters import COLLISION_DAMPING, DO_DAMPING
    m1, m2 = p1.r ** 2, p2.r ** 2
    big_m = m1 + m2
    r1, r2 = np.array((p1.x, p1.y)), np.array((p2.x, p2.y))
    d = np.linalg.norm(r1 - r2) ** 2
    v1, v2 = np.array((p1.vx, p1.vy)), np.array((p2.vx, p2.vy))
    u1 = v1 - 2 * m2 / big_m * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
    u2 = v2 - 2 * m1 / big_m * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
    p1.vx, p1.vy = u1 * (COLLISION_DAMPING if DO_DAMPING else 1)
    p2.vx, p2.vy = u2 * (COLLISION_DAMPING if DO_DAMPING else 1)


def particle_bounce_positions(p1, p2):
    # Push particles away so they don't overlap
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = atan2(dy, dx)
    center_x = p1.x + 0.5 * dx
    center_y = p1.y + 0.5 * dy
    radius = (p1.r + p2.r) / 2
    p1.x = center_x - (cos(angle) * radius)
    p1.y = center_y - (sin(angle) * radius)
    p2.x = center_x + (cos(angle) * radius)
    p2.y = center_y + (sin(angle) * radius)