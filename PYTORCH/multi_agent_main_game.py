import pygame
import itertools
import os
import time
import parameters
import gym

from math import pi, acos
from random import randint, random, uniform
from itertools import combinations
from scipy import interpolate
from PIL import Image
from parameters import max_particle_speed
from gym import spaces
from utils import *
from scipy.spatial.distance import cdist
from display_size import displayHeight, displayWidth

import pickle


class Particle:
    def __init__(self, r, angle=None, vel_mult=1, x=None, y=None):
        a = 2 * pi * random() if angle is None else angle  # Angles are clockwise from 3 o'clock
        self.vx, self.vy = (1, 0)  # (cos(a) * vel_mult, sin(a) * vel_mult)
        self.dir = pygame.math.Vector2((self.vx, self.vy)).normalize()
        self.color = (85, 239, 196)
        self.r = r  # Radius
        self.x, self.y = x, y  # for testing self.x, self.y = 300, 400
        if self.x is None or self.y is None:
            self.calculate_pos()

        self.force_x = 0  # used for correcting the particle
        self.force_y = 0  # used for correcting the particle

        self.max_vel = max_particle_speed

    def calculate_pos(self):
        # Find a position where we won't overlap with any other particles
        while True:
            self.x, self.y = randint(self.r, displayWidth - self.r), randint(self.r, displayHeight - self.r)
            # if overlapping_particle(self.x, self.y, self.r) is None:  # If valid spot
            # break
            break

    def damping(self, amount):
        # Slow down the particle gradually
        self.vx *= amount
        self.vy *= amount

    def gravity(self, amount):
        self.vx += amount

    def wall_collisions(self, damping_amount):
        # Velocity
        collided = False
        if not self.r < self.x < displayWidth - self.r:
            self.vx *= damping_amount
            collided = True
        if not self.r < self.y < displayHeight - self.r:
            self.vy *= damping_amount
            collided = True

        # Position
        if collided:
            # Get out of wall
            self.x = clamp(self.x, low=self.r, high=displayWidth - self.r)
            self.y = clamp(self.y, low=self.r, high=displayHeight - self.r)

    def thickness_lines(self, x, y, x1, x2, y1, y2, width, damping_amount):
        r = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        delta_x = width * (y1 - y2) / r
        delta_y = width * (x2 - x1) / r
        x3 = x1 + delta_x
        y3 = y1 + delta_y
        x4 = x2 + delta_x
        y4 = y2 + delta_y

        x5 = x1 - delta_x
        y5 = y1 - delta_y
        x6 = x2 - delta_x
        y6 = y2 - delta_y

        self.line_collision(x, y, x3, x4, y3, y4, damping_amount)
        self.line_collision(x, y, x5, x6, y5, y6, damping_amount)

    def line_collision(self, x, y, x1, x2, y1, y2, damping_amount):
        collided = False
        # radius = parameters.NEW_GENERATION_RADIUS
        # self.color = (85, 239, 196)
        x1 -= x
        y1 -= y
        x2 -= x
        y2 -= y

        normal_new = (-(y2 - y1), (x2 - x1))

        a = (x2 - x1) ** 2 + (y2 - y1) ** 2
        b = 2 * (x1 * (x2 - x1) + y1 * (y2 - y1))
        c = x1 ** 2 + y1 ** 2 - self.r ** 2
        disc = b ** 2 - 4 * a * c
        if disc <= 0:
            self.color = (85, 239, 196)
        else:
            sqrtdisc = sqrt(disc)
            t1 = (-b + sqrtdisc) / (2 * a)
            t2 = (-b - sqrtdisc) / (2 * a)
            if (0 < t1 < 1) or (0 < t2 < 1):
                # self.color = (255, 255, 255)

                new_direc = self.dir.reflect(normal_new).normalize()

                self.vx = new_direc[0]  # * damping_amount
                self.vy = new_direc[1]  # * damping_amount

    def move(self, speed):
        self.dir = pygame.math.Vector2((self.vx, self.vy)).normalize()

        self.correction_force()

        if self.vx > self.max_vel:
            self.vx = self.max_vel
        elif self.vx < -self.max_vel:
            self.vx = -self.max_vel
        elif self.vy > self.max_vel:
            self.vy = self.max_vel
        elif self.vy < -self.max_vel:
            self.vy = -self.max_vel

        self.x += self.vx * speed
        self.y += self.vy * speed
        if self.x > displayWidth - self.r:  # is the BC that allows particles to move through like a pipe, resets to x=0
            self.x = self.r

    def draw(self, surface):
        draw_circle(surface, self.x, self.y, self.r, self.color)

    def apply_force_towards(self, x, y, strength=1):
        # May be used in later implementations with attractions
        dx = x - self.x
        dy = y - self.y
        self.vx += dx * 0.00001 * strength
        self.vy += dy * 0.00001 * strength

    def correction_force(self):
        vx = self.vx
        vy = self.vy

        # self.vx += (self.force_x * (vx ** 2)) #/ 250
        # self.vy += (self.force_y * (vy ** 2)) #/ 250

        self.vx += self.force_x / 60
        self.vy += self.force_y / 60


class Grid:
    def __init__(self):
        self.resolution = 20
        self.width = displayWidth
        self.height = displayHeight

    def draw_arrow(self, surface, x, y, direc_x, direc_y, width=2):
        x1 = x
        y1 = y
        x2 = (x1 + (direc_x * 20))  # *20
        y2 = (y1 + (direc_y * 20))
        pygame.draw.line(surface, (255, 255, 255), (x1, y1), (x2, y2), width).normalize()

    def draw(self, surface):
        for x in range(0, self.width, self.resolution):
            for y in range(0, self.height, self.resolution):
                rect = pygame.Rect(x, y, self.resolution, self.resolution)
                pygame.draw.rect(surface, (255, 255, 255), rect, 1)


# class Wall:     # unused code to create walls - it is littered throughout script for future work if needed
#
#     def __init__(self, x1, y1, x2, y2, width):
#         self.x1 = x1
#         self.y1 = y1
#         self.x2 = x2
#         self.y2 = y2
#         self.width = width
#
#     def draw(self, surface):
#         pygame.draw.line(surface, (50, 120, 100), (self.x1, self.y1), (self.x2, self.y2), self.width)


class Game(gym.Env):

    def __init__(self, words):
        self.render_images = False
        if words == 'Render':
            self.render_images = True

        self.width = displayWidth
        self.height = displayHeight

        if self.render_images:
            pygame.init()
            pygame.display.set_caption("Simple Wall Fluid Sim")

            self.screen = pygame.display.set_mode((self.width, self.height))
            self.myfont = pygame.font.SysFont("monospace", 16)
            self.myfont2 = pygame.font.SysFont("monospace", 24)

        self.clock = pygame.time.Clock()
        self.ticks = 1  # sets fps
        self.exit = False

        self.grid = Grid()

        self.walls = []
        self.line_width = 5

        # self.set_walls()
        # self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

        self.particles = []
        self.new_generation()

        self.press = False

        self.wall_list = []

        self.gravity = True
        self.flow = True

        self.r = parameters.NEW_GENERATION_RADIUS  # particle radius

        self.final_mesh = 40  # dictates the resolution of divergence calculations

        self.draw_particles = True

        self.draw_flow_field = False
        self.fps = 0

        self.div_norm = 0

        self.k = 5  # number of knn to use for state space

        self.par = np.array([0, 0])

        self.max_vel = max_particle_speed  # relates to the particle class

        self.last_div = 0

        self.n = parameters.NEW_GENERATION_NUM_PARTICLES

        self.particle_vel = [self.max_vel, self.max_vel]
        self.particle_position_low = [0, 0]
        self.particle_position_high = [1, 1]
        self.agent_angle_low = 2 * self.k * [0]
        self.agent_angle_high = 2 * self.k * [1]
        self.agent_vel = 2 * self.k * [self.max_vel]

        # adjusting for different state spaces
        # low = np.array((self.particle_vel + self.agent_angle_low + self.agent_vel), dtype=np.float32) low =
        # np.array((self.agent_angle_low + self.agent_vel), dtype=np.float32) low = np.array((self.particle_vel +
        # self.particle_position_low + self.agent_angle_low + self.agent_vel), dtype=np.float32)
        low = np.array(self.particle_vel, dtype=np.float32)

        # high = np.array((self.particle_vel + self.agent_angle_high + self.agent_vel), dtype=np.float32) high =
        # np.array((self.agent_angle_high + self.agent_vel), dtype=np.float32) high = np.array((self.particle_vel +
        # self.particle_position_high + self.agent_angle_high + self.agent_vel), dtype=np.float32)
        high = np.array(self.particle_vel, dtype=np.float32)

        self.observation_space = []
        self.action_space = []
        for agent in self.particles:
            self.observation_space.append(spaces.Box(low=-low, high=high, dtype=np.float32))
            self.action_space.append(
                spaces.Box(np.array([-1, -1]).astype(np.float32), np.array([1, 1]).astype(np.float32)))

        # self.state_space = 2 + 4 * self.k  # spaces.Box(low=-1, high=1, shape=[15,], dtype=np.float32)
        self.state_space = 2
        # self.state_space = 4 * self.k  # spaces.Box(low=-1, high=1, shape=[15,], dtype=np.float32)
        self.num_agents = parameters.NEW_GENERATION_NUM_PARTICLES

        self.area = [0] * self.num_agents
        self.time_taken = 0
        self.time = time.time()

        self.display_num = 0

    def reset(self):
        self.grid = Grid()

        self.walls = []
        self.line_width = 5  # 2.5

        # self.set_walls()
        # self.mouse_x, self.mouse_y = pygame.mouse.get_pos()

        self.particles = []
        self.new_generation()

        self.press = False

        self.wall_list = []

        self.div_norm = 0

        self.area = [0] * self.num_agents
        self.time_taken = 0
        self.time = time.time()

        # self.flow = True

        self.par = np.array([0, 0])

        self.gravity = True

        # if self.gravity:
        parameters.GRAVITY = 0.025  # if parameters.GRAVITY == 0 else 0  # originally 0.025
        parameters.DO_DAMPING = parameters.GRAVITY != 0

        self.last_div = 0

        s = []
        for _ in self.particles:
            s.append([0] * self.state_space)

        s = np.array(s)
        return s

    # def set_walls(self):
    #     self.walls.append(Wall(0, -100, 1, displayHeight+100))
    #     self.walls.append(Wall(-100, 0, displayWidth+100, 1))
    #     self.walls.append(Wall(displayWidth, -100, displayWidth-1, displayHeight+100))
    #     self.walls.append(Wall(-100, displayHeight, displayWidth+100, displayHeight-1))

    def add_particle(self, angle=None, vel_mult=1, x=None, y=None, r=None):
        self.particles.append(Particle(angle=angle, vel_mult=vel_mult, x=x, y=y, r=r))

        return

    def new_generation(self):
        for _ in range(parameters.NEW_GENERATION_NUM_PARTICLES):
            self.particles.append(Particle(r=parameters.NEW_GENERATION_RADIUS))

        return

    def divergence(self, f, sp):
        """ Computes divergence of vector field
        f: array -> vector field components [Fx,Fy,Fz,...]
        sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
        """
        num_dims = len(f)
        return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])

    def calc_div(self, xx, yy, u, v):
        xx.extend([0, 0, self.width, self.width])
        yy.extend([0, self.height, 0, self.height])
        # u.extend([self.max_vel, self.max_vel, self.max_vel, self.max_vel])
        u.extend([parameters.GRAVITY, parameters.GRAVITY, parameters.GRAVITY, parameters.GRAVITY])
        v.extend([0, 0, 0, 0])
        # add data points for each corner to xx, yy, u ,v
        xxx = np.linspace(0, self.width, self.final_mesh)
        yyy = np.linspace(0, self.height, self.final_mesh)
        xxxx, yyyy = np.meshgrid(xxx, yyy)

        points = np.transpose(np.vstack((xx, yy)))

        u_interp = interpolate.griddata(points, u, (xxxx, yyyy), method='cubic')
        v_interp = interpolate.griddata(points, v, (xxxx, yyyy), method='cubic')

        F = np.array([u_interp, v_interp])
        sp_x = np.diff(xxx)[0]
        sp_y = np.diff(yyy)[0]
        sp = [sp_x, sp_y]
        g = self.divergence(F, sp)

        self.div_norm = np.linalg.norm(g)

        return xxx, yyy, xxxx, yyyy, u_interp, v_interp

    def draw_arrow(self, surface, xxxx, yyyy, u_interp, v_interp, width):
        chain_xxx = list(itertools.chain.from_iterable(xxxx))
        chain_yyy = list(itertools.chain.from_iterable(yyyy))
        chain_u = list(itertools.chain.from_iterable(u_interp))
        chain_v = list(itertools.chain.from_iterable(v_interp))

        for i in range(len(chain_xxx)):
            x1 = chain_xxx[i]
            y1 = chain_yyy[i]
            x2 = x1 + (chain_u[i] * 4)
            y2 = y1 + (chain_v[i] * 4)

            dx = x2 - x1
            dy = y2 - y1
            length = sqrt(dx * dx + dy * dy)

            udx = dx / length
            udy = dy / length

            perpx = -udy
            perpy = udx

            L = 5
            H = 5

            x3 = x2 - L * udx + H * perpx
            y3 = y2 - L * udy + H * perpy

            x4 = x2 - L * udx - H * perpx
            y4 = y2 - L * udy - H * perpy

            pygame.draw.line(surface, (255, 255, 255), (x1, y1), (x2, y2), int(self.line_width / 2)).normalize()
            pygame.draw.polygon(surface, (255, 255, 255), [(x2, y2), (x3, y3), (x4, y4)])

    def reward(self, old_div, new_div, obs):
        reward = 0

        first_index = (2 + self.k)
        last_index = (2 + (2 * self.k))

        # first_index = (self.k)
        # last_index = (2 * self.k)

        new_obs = obs[first_index:last_index]

        if -0.05 <= new_div <= 0.05:
            reward += 1  # 100
        elif abs(old_div) > abs(new_div):
            reward += 40 - abs(self.div_norm)  # 40 random number here just cus it never goes above like 2 div
        elif abs(old_div) < abs(new_div):
            reward -= abs(self.div_norm) * 10
        for j in new_obs:
            if j < 0.05:
                reward -= 100

        return reward

    def update_div(self):
        xx = []
        yy = []
        u = []
        v = []

        for particle in self.particles:
            if self.render_images:
                if self.draw_particles:
                    particle.draw(self.screen)
                    # pygame.draw.line(self.screen, (255, 255, 255), (particle.x, particle.y), # draws NN direc
                    #                  (particle.x + particle.force_x * 100, particle.y + particle.force_y * 100),
                    #                  int(self.line_width / 2))
                    # pygame.draw.line(self.screen, (255, 100, 255), (particle.x, particle.y), # draw arrow on particle direc
                    #                  (particle.x + particle.vx * 20, particle.y + particle.vy * 20),
                    #                  int(self.line_width / 2))
            xx.append(particle.x)
            yy.append(particle.y)
            u.append(particle.vx)
            v.append(particle.vy)

        x_new = xx.copy()
        y_new = yy.copy()

        xxx, yyy, xxxx, yyyy, u_interp, v_interp = self.calc_div(x_new, y_new, u, v)

        return xxx, yyy, xxxx, yyyy, u_interp, v_interp

    def render(self, divergence):
        self.screen.fill((0, 0, 0))

        for i in self.walls:
            i.draw(self.screen)

        xxx, yyy, xxxx, yyyy, u_interp, v_interp = self.update_div()

        if self.draw_flow_field:
            self.draw_arrow(self.screen, xxxx, yyyy, u_interp, v_interp, self.line_width)

        # self.mouse_x, self.mouse_y = pygame.mouse.get_pos()
        # mousetext = self.myfont.render(str(self.mouse_x) + " and " + str(self.mouse_y), 1, (255, 255, 255))
        # self.screen.blit(mousetext, (400, 10))

        divtext = self.myfont2.render("Cur Divergence: " + str(f'{self.div_norm:.3f}'), 1, (255, 100, 255))
        self.screen.blit(divtext, (7, 10))

        # ftext = self.myfont.render("Flow Velocity: " + str(self.flow), 1, (255, 255, 255))
        # self.screen.blit(ftext, (7, 70))

        # ftext = self.myfont.render("Frame Number: " + str(self.display_num), 1, (255, 255, 255))
        # self.screen.blit(ftext, (7, 70))

        # divergencetext = self.myfont2.render("Ave Divergence: " + str(f'{divergence:.3f}'), 1, (255, 100, 255))
        # self.screen.blit(divergencetext, (330, 10))

        # pygame.draw.line(self.screen, (255, 255, 255), (100, 200), (200, 400), int(self.line_width / 2)).normalize()

        self.display_num += 1

        pygame.display.flip()

        pygame.event.pump()  #  stops pygame from crashing when run without input ie on a loop

        return

    # def add_wall_coord(self):
    #     self.mouse_x, self.mouse_y = pygame.mouse.get_pos()
    #     if len(self.wall_list) == 0:
    #         self.wall_list.append(self.mouse_x)
    #         self.wall_list.append(self.mouse_y)
    #     if len(self.wall_list) == 2 and self.mouse_x not in self.wall_list and self.mouse_y not in self.wall_list:
    #         self.wall_list.append(self.mouse_x)
    #         self.wall_list.append(self.mouse_y)
    #         self.walls.append(Wall(self.wall_list[0], self.wall_list[1],
    #                                self.wall_list[2], self.wall_list[3], int(2 * self.line_width)))
    #         self.wall_list = []
    #     return

    def save_pics(self, value, name):
        self.display_surface = pygame.display.get_surface()

        self.image3d = np.ndarray(
            (displayWidth, displayHeight, 3), np.uint8)

        pygame.pixelcopy.surface_to_array(
            self.image3d, self.display_surface)
        self.image3dT = np.transpose(self.image3d, axes=[1, 0, 2])
        im = Image.fromarray(self.image3dT)  # monochromatic image
        imrgb = im.convert('RGB')  # color image

        filename = ''.join(['Episode_',
                            str(value),
                            '-frame-',
                            str(name).zfill(5),
                            '.jpg'])
        foldername = ''.join(['./MADDPG_frames/Episode_', str(value)])
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        filenamepath = os.path.join(foldername, filename)
        imrgb.save(filenamepath)

    def particle_update(self, part, action):
        part.damping(parameters.DAMPING if parameters.DO_DAMPING else 1)
        part.gravity(parameters.GRAVITY)
        part.wall_collisions(-parameters.WALL_DAMPING if parameters.DO_DAMPING else -1)
        # part.force_x = (action[0] * 2) - 1 # used if tanh function on output of NN
        # part.force_y = (action[1] * 2) - 1
        part.force_x = action[0]
        part.force_y = action[1]

        part.move(parameters.SPEED_MULTIPLIER)
        for wall in self.walls:
            # particle.user_wall_collisions(particle.x, particle.y, wall.x1, wall.x2, wall.y1, wall.y2,
            # -parameters.WALL_DAMPING if parameters.DO_DAMPING else -1) particle.line_collision(particle.x,
            # particle.y, wall.x1, wall.x2, wall.y1, wall.y2, self.line_width, -parameters.WALL_DAMPING if
            # parameters.DO_DAMPING else -1)
            part.thickness_lines(part.x, part.y, wall.x1, wall.x2, wall.y1, wall.y2,
                                 self.line_width, -parameters.WALL_DAMPING if parameters.DO_DAMPING else -1)

        return

    def particle_get_state(self, part):
        angles = []
        nearest_n, nearest_distances, indices = self.knn([part.x, part.y], self.par)
        for i in range(self.k):
            angles.append(self.angle_btwn_points([part.x, part.y], nearest_n[i]))

        index = [self.particles[i] for i in indices]
        knn_vx = [j.vx / max_particle_speed for j in index]
        knn_vy = [j.vy / max_particle_speed for j in index]

        # knn_vx = [(j.vx - part.vx) / max_particle_speed for j in index]
        # knn_vy = [(j.vy - part.vy) / max_particle_speed for j in index]

        # knn_x = [j.x / self.width for j in index]
        # knn_y = [j.y / self.width for j in index]

        normalised_nearest_distances = [i / self.width for i in nearest_distances]

        # normalisedstate = [part.vx / max_particle_speed, part.vy / max_particle_speed, *angles,
        #                    *normalised_nearest_distances, *knn_vx, *knn_vy]

        # normalisedstate = [*angles,
        #                    *normalised_nearest_distances, *knn_vx, *knn_vy]

        # normalisedstate = [part.vx / max_particle_speed, part.vy / max_particle_speed, part.x / self.width,
        #                    part.y / self.height, *knn_x,
        #                    *knn_y, *knn_vx, *knn_vy]

        normalisedstate = [part.vx / max_particle_speed, part.vy / max_particle_speed]

        return np.array(normalisedstate)

    def angle_btwn_points(self, pointsA, pointsB):
        x1 = pointsA[0]
        x2 = pointsA[0]
        y1 = pointsA[1]
        y2 = pointsA[1] + 100
        x3 = pointsA[0]
        x4 = pointsB[0]
        y3 = pointsA[1]
        y4 = pointsB[1]
        angle = acos(((x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3)) / (
                (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5) * (((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))) * 180 / pi
        if x4 >= x1:
            angle = angle
        elif x4 < x1:
            angle = 360 - angle

        return angle / 360

    def knn(self, node, nodes):  # finds k nearest particles to each particle
        listu = cdist([node], nodes)
        result = nodes[np.argpartition(listu, range(self.k + 1))[:self.k + 1]]
        results = listu[0][np.argpartition(listu, range(self.k + 1))][:self.k + 1]
        indices = np.argpartition(listu, range(self.k + 1))[:self.k + 1]

        return result[0][1:self.k + 1], results[0][1:self.k + 1], indices[0][1:self.k + 1]

    def collisions(self):
        # Test inter-particle collisions
        self.par = np.asarray([[i.x, i.y] for i in self.particles])
        pairs = combinations(range(len(self.particles)), 2)  # All combinations of particles
        for i, j in pairs:
            p1, p2 = self.particles[i], self.particles[j]
            if dist(p1.x, p1.y, p2.x, p2.y, less_than=p1.r + p2.r):  # If they are touching
                particle_bounce_velocities(p1, p2)
                particle_bounce_positions(p1, p2)

        return

    def step(self, action_n, old_div, new_div):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # set action for each agent
        for i, agent in enumerate(self.particles):
            self.particle_update(agent, action_n[i])
        # advance world state

        # record observation for each agent
        for agent in self.particles:
            observations = self.particle_get_state(agent)
            obs_n.append(observations)
            reward_n.append(self.reward(old_div, new_div, observations))
            done_n.append(False)

            info_n['n'].append("empty")

        return obs_n, reward_n, done_n, info_n

    def close(self):
        self.exit = True


if __name__ == '__main__':
    env = Game('Render')

    s = env.reset()

    div_avrge2 = 0.846  # random value within basic game loop - has use within maddpg script

    time_steps = 1000  # 800
    total_x = []
    total_y = []
    total_u = []
    total_v = []
    for _ in range(time_steps):
        if env.gravity:
            parameters.GRAVITY = 0.025 if parameters.GRAVITY == 0 else 0  # originally 0.025
            parameters.DO_DAMPING = parameters.GRAVITY != 0

        env.collisions()

        if env.render_images:
            env.render(div_avrge2)

        action_n = []
        # x = []
        # y = []
        # u = []
        # v = []
        for turn in env.particles:
            action_n.append([uniform(-10, 10), uniform(-10, 10)])
            # x.append(turn.x)
            # y.append(turn.y)
            # u.append(turn.vx)
            # v.append(turn.vy)
        s, reward, done, _ = env.step(action_n, env.last_div, env.div_norm)

        # game.render()

        env.last_div = env.div_norm

        # total_x.append(x)
        # total_y.append(y)
        # total_u.append(u)
        # total_v.append(v)

    # with open('x.pkl', 'wb') as f:
    #     pickle.dump(total_x, f)
    # with open('y.pkl', 'wb') as f:
    #     pickle.dump(total_y, f)
    # with open('u.pkl', 'wb') as f:
    #     pickle.dump(total_u, f)
    # with open('v.pkl', 'wb') as f:
    #     pickle.dump(total_v, f)

    env.close()
