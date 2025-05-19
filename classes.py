import pygame 
import numpy as np
from settings import *
import math
from matrix_funcs import translate, rotate_x, rotate_y, rotate_z, scale
from random import randint
from funcs import intersect_numba, clip_triangle_numba, process_faces

class Camera:
    def __init__(self, pos):
       self.pos = np.array([*pos, 1.0])
       self.forward = np.array([0,0,1,1])
       self.up = np.array([0,1,0,1])
       self.right = np.array([1,0,0,1])
       self.h_fov = math.pi / 3
       self.v_fov = self.h_fov * (HEIGHT / WIDTH)
       self.near = 0.1
       self.far = 100
       self.move_speed = 1
       self.mouse_sensitivity = 0.001
       self.anglePitch = 0
       self.angleYaw = 0
       self.angleRoll = 0

    def control(self):
        key = pygame.key.get_pressed()
        if key[pygame.K_d]: self.pos += self.right * self.move_speed   # right
        if key[pygame.K_a]: self.pos -= self.right * self.move_speed   # left
        if key[pygame.K_w]: self.pos += self.forward * self.move_speed # forward
        if key[pygame.K_s]: self.pos -= self.forward * self.move_speed # backward
        if key[pygame.K_SPACE]: self.pos += self.up * self.move_speed  # up
        if key[pygame.K_LSHIFT]: self.pos -= self.up * self.move_speed # down

    def mouse_control(self, event):
        dx, dy = event.rel
        self.camera_yaw(-dx * self.mouse_sensitivity)
        self.camera_pitch(-dy * self.mouse_sensitivity)
        pygame.mouse.set_pos((H_WIDTH, H_HEIGHT))

    def camera_yaw(self, angle):
        self.angleYaw += angle

    def camera_pitch(self, angle):
        self.anglePitch += angle
        self.anglePitch = max(-math.pi / 2, min(math.pi / 2, self.anglePitch))

    def axiiIdentity(self):
        self.forward = np.array([0, 0, 1, 1])
        self.up = np.array([0, 1, 0, 1])
        self.right = np.array([1, 0, 0, 1])

    def camera_update_axii(self):
        rotate = rotate_x(self.anglePitch) @ rotate_y(self.angleYaw)
        self.axiiIdentity()
        self.forward = self.forward @ rotate
        self.right = self.right @ rotate
        self.up = self.up @ rotate


    def translate(self):
        x, y, z, w = self.pos
        return np.array([
            [1,0,0,0],
            [0,1,0,1],
            [0,0,1,0],
            [-x, -y, -z, 1]
        ])

    def rotate(self):
        fx, fy, fz, w = self.forward
        ux, uy, uz, w = self.up
        rx, ry, rz, w = self.right
        return np.array([
            [rx, ux, fx, 0],
            [ry, uy, fy, 0],
            [rz, uz, fz, 0],
            [0,0,0,1]
        ])

    def camera_matrix(self):
        self.camera_update_axii()
        return self.translate() @ self.rotate()


class Projection:
    def __init__(self, engine):
        NEAR = engine.camera.near
        FAR = engine.camera.far
        RIGHT = math.tan(engine.camera.h_fov / 2)
        LEFT = -RIGHT
        TOP = math.tan(engine.camera.v_fov / 2)
        BOTTOM = -TOP

        m00 = 2 / (RIGHT - LEFT)
        m11 = 2 / (TOP - BOTTOM)
        m22 = (FAR + NEAR) / (FAR - NEAR)
        m32 = -2 * NEAR * FAR / (FAR - NEAR)
        self.projection_matrix = np.array([
            [m00, 0, 0, 0],
            [0, m11, 0, 0],
            [0, 0, m22, 1],
            [0, 0, m32, 0]
        ], dtype=np.float32)

        HW, HH = H_WIDTH, H_HEIGHT
        self.screen_matrix = np.array([
            [HW, 0, 0, 0],
            [0, -HH, 0, 0],
            [0, 0, 1, 0],
            [HW, HH, 0, 1]
        ], dtype=np.float32)

class Engine:
    def __init__(self):
        pygame.init()
        self.dis = pygame.display.set_mode(RES)
        self.clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        self.camera = Camera([0,0,0])
        self.font = pygame.font.SysFont('Arial', 50, bold=True)
        self.projection = Projection(self)
        self.objects = []

    def draw(self):
        self.dis.fill((0,0,0))
        self.dis.blit(self.font.render(f"FPS: {str(int(self.clock.get_fps()))}", 0, (255,255,255)), (WIDTH-250, 5)) # drawing FPS
        for obj in self.objects:
            obj.draw()

    def run(self):
        self.draw()
        self.camera.control()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.MOUSEMOTION:
                self.camera.mouse_control(event)
        pygame.display.flip()
        self.clock.tick(FPS)

    def create_object(self, vertexes, faces):
        self.objects.append(Object(self, vertexes, faces))

    def get_object_from_file(self, file_path):
        vertex, faces = [], []
        with open(file_path) as f:
            for line in f:
                if line.startswith('v '):
                    vertex.append([float(i) for i in line.split()[1:]] + [1])
                elif line.startswith('f'):
                    faces_ = line.split()[1:]
                    faces.append([int(face_.split('/')[0]) - 1 for face_ in faces_])
        self.create_object(vertex, faces)


class Object:
    def __init__(self, engine, vertexes, faces):
        self.engine = engine
        self.vertexes = vertexes
        self.faces = self.triangulation(faces)

    def triangulation(self, faces):
        triangles = []
        for face in faces:
            for i in range(len(face)-1, 0, -1):
                triangles.append((face[0], face[i], face[i-1]))
                if i == 2: break
        return np.array(triangles)

    def screen_projection(self):
        vertexes = self.vertexes @ self.engine.camera.camera_matrix()
        vertexes = vertexes @ self.engine.projection.projection_matrix
        faces = self.faces.astype(np.int32)

        triangles, depths = process_faces(vertexes, faces, self.engine.projection.screen_matrix)

        sorted_tris = sorted(zip(depths, triangles), key=lambda x: x[0], reverse=True)

        for _, triangle in sorted_tris:
            pygame.draw.polygon(self.engine.dis, pygame.Color("orange"), triangle.tolist(), 3)

    def draw(self):
        self.screen_projection()

    def translate(self, pos):
        self.vertexes = self.vertexes @ translate(pos)

    def rotate_x(self, angle):
        self.vertexes = self.vertexes @ rotate_x(angle)

    def rotate_y(self, angle):
        self.vertexes = self.vertexes @ rotate_y(angle)

    def rotate_z(self, angle):
        self.vertexes = self.vertexes @ rotate_z(angle)

    def scale(self, size):
        self.vertexes = self.vertexes @ scale(size)



