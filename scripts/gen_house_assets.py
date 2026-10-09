#!/usr/bin/env python3
"""Generate the assets for worlds/house_chimney.sdf:
  - models/chimney/meshes/chimney.obj   square chimney with a round flue
  - models/house/materials/textures/{plaster,roof_tiles}.jpg
  - models/grass_ground/materials/textures/grass.jpg

All textures are noisy on purpose: stereo matching needs surface texture.
Run from the repo root:  python3 scripts/gen_house_assets.py
"""
import math
import os

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
rng = np.random.default_rng(42)


def fractal_noise(size, octaves=6):
    """Tileable value noise in [0, 1], sum of upsampled random grids."""
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        cells = 4 * 2 ** o
        grid = rng.random((cells, cells))
        img = Image.fromarray((grid * 255).astype(np.uint8))
        # Wrap-pad before resizing so the noise tiles seamlessly.
        tiled = np.tile(np.asarray(img), (3, 3))
        big = Image.fromarray(tiled).resize((3 * size, 3 * size), Image.BICUBIC)
        out += amp * np.asarray(big, dtype=float)[size:2 * size, size:2 * size] / 255
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (out.max() - out.min())


def save(rgb, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path, quality=92)
    print("wrote", os.path.relpath(path))


def grass(size=2048):
    n = fractal_noise(size)
    fine = rng.random((size, size))
    base = np.array([70, 125, 45], float)        # mid green
    dry = np.array([125, 140, 60], float)        # yellowish patches
    t = np.clip((n - 0.45) * 2.5, 0, 1)[..., None]
    rgb = base * (1 - t) + dry * t
    rgb *= (0.75 + 0.5 * fine)[..., None]        # per-pixel blade speckle
    return rgb


def plaster(size=1024):
    n = fractal_noise(size)
    fine = rng.random((size, size))
    rgb = np.array([225, 215, 190], float) * (0.85 + 0.12 * n + 0.06 * fine)[..., None]
    return rgb


def roof_tiles(size=1024, rows=16, cols=12):
    n = fractal_noise(size)
    fine = rng.random((size, size))
    y, x = np.mgrid[0:size, 0:size]
    row = y * rows // size
    xs = (x + (row % 2) * (size // cols // 2)) % size
    # Darken the gaps between tiles and shade each tile towards its lower edge.
    fy = (y * rows / size) % 1.0
    fx = (xs * cols / size) % 1.0
    gap = (fy > 0.92) | (fx < 0.04)
    tile_var = rng.random((rows, cols + 1))[row, xs * cols // size]
    shade = 0.75 + 0.25 * fy + 0.15 * tile_var + 0.1 * n + 0.05 * fine
    rgb = np.array([165, 65, 40], float) * shade[..., None]
    rgb[gap] *= 0.45
    return rgb


def chimney_obj(path, half=0.30, r_in=0.18, height=3.0, seg=64, u_tiles_in=2):
    """Square chimney (2*half wide) with a round flue of radius r_in.

    Outer: four flat walls, texture spans each face once (like a box).
    Inner: round flue wall. Top: flat cap between the square and the circle.
    """
    assert seg % 8 == 0, "seg must be a multiple of 8 so the square corners are vertices"
    v, vt, vn, f = [], [], [], []

    def quad(p, uv, n):
        """Append a quad (counter-clockwise seen from the normal side)."""
        vb, tb = len(v), len(vt)
        v.extend(p)
        vt.extend(uv)
        vn.append(n)
        ni = len(vn) - 1
        c = [(vb + k, tb + k, ni) for k in range(4)]
        f.append([c[0], c[1], c[2]])
        f.append([c[0], c[2], c[3]])

    # Outer walls. Corners counter-clockwise seen from above.
    h = half
    corners = [(h, -h), (h, h), (-h, h), (-h, -h)]
    normals = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for k in range(4):
        (x0, y0), (x1, y1) = corners[k], corners[(k + 1) % 4]
        quad([(x0, y0, 0), (x1, y1, 0), (x1, y1, height), (x0, y0, height)], uv, normals[k])

    # Round flue wall, normals pointing into the flue.
    angles = [2 * math.pi * i / seg for i in range(seg + 1)]
    vb, tb = len(v), len(vt)
    for z, tv in ((0.0, 0.0), (height, 1.0)):
        for i, a in enumerate(angles):
            v.append((r_in * math.cos(a), r_in * math.sin(a), z))
            vt.append((u_tiles_in * i / seg, tv))
    nb = len(vn)
    for a in angles:
        vn.append((-math.cos(a), -math.sin(a), 0.0))
    for i in range(seg):
        b0, b1 = i, i + 1
        t0, t1 = b0 + seg + 1, b1 + seg + 1
        c = lambda k, n: (vb + k, tb + k, nb + n)
        f.append([c(b0, i), c(t1, i + 1), c(b1, i + 1)])
        f.append([c(b0, i), c(t0, i), c(t1, i + 1)])

    # Top cap: each circle vertex is joined to the point on the square along
    # the same ray. With seg % 8 == 0 the rays at 45 deg hit the corners.
    vb, tb = len(v), len(vt)
    for a in angles:
        cx, cy = math.cos(a), math.sin(a)
        s_ = h / max(abs(cx), abs(cy))
        for x, y in ((r_in * cx, r_in * cy), (s_ * cx, s_ * cy)):
            v.append((x, y, height))
            vt.append((0.5 + x / (2 * h), 0.5 + y / (2 * h)))
    vn.append((0.0, 0.0, 1.0))
    nz = len(vn) - 1
    for i in range(seg):
        i0, o0, i1, o1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        c = lambda k: (vb + k, tb + k, nz)
        f.append([c(i0), c(o0), c(o1)])
        f.append([c(i0), c(o1), c(i1)])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# Square chimney with round flue, generated by scripts/gen_house_assets.py\n")
        fh.write(f"# outer={2 * half} x {2 * half} r_in={r_in} height={height}\n")
        for p in v:
            fh.write("v %.5f %.5f %.5f\n" % p)
        for t in vt:
            fh.write("vt %.5f %.5f\n" % t)
        for n in vn:
            fh.write("vn %.5f %.5f %.5f\n" % n)
        for tri in f:
            fh.write("f " + " ".join(f"{a + 1}/{b + 1}/{c + 1}" for a, b, c in tri) + "\n")
    print("wrote", os.path.relpath(path))


if __name__ == "__main__":
    chimney_obj(os.path.join(ROOT, "chimney/meshes/chimney.obj"))
    save(grass(), os.path.join(ROOT, "grass_ground/materials/textures/grass.jpg"))
    save(plaster(), os.path.join(ROOT, "house/materials/textures/plaster.jpg"))
    save(roof_tiles(), os.path.join(ROOT, "house/materials/textures/roof_tiles.jpg"))
