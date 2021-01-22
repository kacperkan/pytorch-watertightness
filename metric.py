import torch
import trimesh

from watertightness import watertightness_backend


def calc_watertigthness(
    origins: torch.Tensor, directions: torch.Tensor, triangles: torch.Tensor
) -> torch.Tensor:
    parity_test = watertightness_backend.watertightness(
        origins, directions, triangles
    )
    return parity_test.mean()


def calc_watertightness_trimesh(
    mesh: trimesh.Trimesh, count: int = 100_000, eps: float = 1e-1
) -> torch.Tensor:
    points_on_surface, face_indices = trimesh.sample.sample_surface(
        mesh, count=count
    )
    buffered = (1 + eps) * points_on_surface
    normals_of_points = mesh.face_normals[face_indices]
    inside_normals = -normals_of_points

    triangles = mesh.vertices[mesh.faces]

    origins = torch.from_numpy(buffered).float().cuda()
    directions = torch.from_numpy(inside_normals).float().cuda()
    triangles = torch.from_numpy(triangles).float().cuda()
    watertigthness = calc_watertigthness(origins, directions, triangles)
    return watertigthness


def run_example() -> torch.Tensor:
    import numpy as np

    original = trimesh.creation.icosphere()

    with_faces_removed = original.copy()
    mask = np.ones((len(with_faces_removed.faces),), dtype=np.bool)
    mask[:800] = False
    with_faces_removed.update_faces(mask)
    original_watertightness = calc_watertightness_trimesh(
        original, count=100_000
    )
    with_faces_removed_watertightness = calc_watertightness_trimesh(
        with_faces_removed, count=100_000
    )

    print(
        "Watertigtheness for the original mesh: {:.4f}".format(
            original_watertightness
        )
    )
    print(
        "Watertigtheness for the mesh with removed faces: {:.4f}".format(
            with_faces_removed_watertightness
        )
    )
