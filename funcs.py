from numba import njit
import numpy as np

@njit(fastmath=True)
def intersect_numba(v1, v2, f1, f2):
    t = f1 / (f1-f2)
    return v1 + t * (v2 - v1)

@njit(fastmath=True, inline='always')
def dot4(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]

@njit(fastmath=True)
def clip_triangle_against_plane_numba(triangle, plane):
    inside = np.empty((3, 4), dtype=np.float32)
    outside = np.empty((3, 4), dtype=np.float32)
    inside_count = 0
    outside_count = 0

    for i in range(3):
        v = triangle[i]
        d = dot4(v, plane)
        if d >= 0:
            inside[inside_count] = v
            inside_count += 1
        else:
            outside[outside_count] = v
            outside_count += 1

    out = np.empty((2, 3, 4), dtype=np.float32)

    if inside_count == 0:
        return out, 0

    elif inside_count == 3:
        out[0] = triangle
        return out, 1

    elif inside_count == 1:
        v0 = inside[0]
        v1 = outside[0]
        v2 = outside[1]
        f0 = dot4(v0, plane)
        f1 = dot4(v1, plane)
        f2 = dot4(v2, plane)
        i1 = intersect_numba(v0, v1, f0, f1)
        i2 = intersect_numba(v0, v2, f0, f2)
        out[0][0], out[0][1], out[0][2] = v0, i1, i2
        return out, 1

    elif inside_count == 2:
        v0 = inside[0]
        v1 = inside[1]
        v2 = outside[0]
        f0 = dot4(v0, plane)
        f1 = dot4(v1, plane)
        f2 = dot4(v2, plane)
        i0 = intersect_numba(v0, v2, f0, f2)
        i1 = intersect_numba(v1, v2, f1, f2)
        out[0][0], out[0][1], out[0][2] = v0, v1, i1
        out[1][0], out[1][1], out[1][2] = v0, i1, i0
        return out, 2

@njit(fastmath=True)
def clip_triangle_numba(triangle):
    planes = np.array([
        [ 1.0,  0.0,  0.0,  1.0],   # left
        [-1.0,  0.0,  0.0,  1.0],   # right
        [ 0.0,  1.0,  0.0,  1.0],   # bottom
        [ 0.0, -1.0,  0.0,  1.0],   # top
        [ 0.0,  0.0,  1.0,  1.0],   # near
        [ 0.0,  0.0, -1.0,  1.0],   # far
    ], dtype=np.float32)

    current = np.empty((64, 3, 4), dtype=np.float32) # buffer of 64 polygons 
    next_ = np.empty_like(current)
    current[0] = triangle
    tri_count = 1

    for p in planes:
        next_count = 0
        for i in range(tri_count):
            clipped, count = clip_triangle_against_plane_numba(current[i], p)
            for j in range(count):
                next_[next_count] = clipped[j]
                next_count += 1
        if next_count == 0:
            return current, 0
        current[:next_count] = next_[:next_count]
        tri_count = next_count

    return current, tri_count

from numba import njit, prange
import numpy as np

@njit()
def process_faces(vertexes, faces, screen_matrix):
    max_output = faces.shape[0] * 4 
    tris_out = np.zeros((max_output, 3, 2), dtype=np.float32)
    depths = np.zeros(max_output, dtype=np.float32)
    count = 0

    for i in prange(faces.shape[0]):
        face = faces[i]
        triangle = vertexes[face].astype(np.float32)
        clipped_triangles, n = clip_triangle_numba(triangle)

        for j in range(n):
            tri = clipped_triangles[j]
            w = np.ascontiguousarray(tri[:, -1]).reshape(-1, 1)
            tri /= w

            z_mean = (tri[0, 2] + tri[1, 2] + tri[2, 2]) / 3.0

            tri_screen = tri @ screen_matrix
            tri2d = tri_screen[:, :2]

            idx = count + j
            if idx < max_output:
                tris_out[idx] = tri2d
                depths[idx] = z_mean

        count += n

    return tris_out[:count], depths[:count]
