import numpy as np
from noise import pnoise2
from random import randint

def create_obj_file():
    width, height = 64, 64 
    scale = 100.0
    octaves = 4
    persistence = 0.4
    lacunarity = 2.2
    seed = randint(0, 200)
    height_scale = 50.0 

    world = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(
                x / scale * 5,
                y / scale * 5,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=seed
            )
            world[y][x] = noise_val

    world = (world - world.min()) / (world.max() - world.min())

    # world generation
    vertices = []
    grid_scale = 10.0
    for y in range(height):
        for x in range(width):
            z = world[y][x] * height_scale
            vertices.append((x * grid_scale, z, y * grid_scale))

    # polygons generation
    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            i = y * width + x
            i_right = i + 1
            i_down = i + width
            i_diag = i_down + 1

            faces.append((i + 1, i_down + 1, i_right + 1))
            faces.append((i_right + 1, i_down + 1, i_diag + 1))

    # saving in .obj
    with open("res/terrain.obj", "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print("terrain.obj created.")

