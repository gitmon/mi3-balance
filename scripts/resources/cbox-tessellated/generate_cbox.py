import sys
import numpy as np
from gpytoolbox import regular_square_mesh, icosphere, write_mesh

def generate(N: int = 5):
    # Define planes
    subdiv = 1 << N
    V2d, F = regular_square_mesh(subdiv, subdiv)
    Nv = V2d.shape[0]
    V = np.c_[V2d[:,0], V2d[:,1], np.zeros(Nv)]

    AxisX = np.array([1,0,0])
    AxisY = np.array([0,1,0])
    AxisZ = np.array([0,0,1])

    home_dir = '/home/jonathan/Documents/mi3-balance/scripts/resources/cbox-tessellated/meshes/'

    # Back wall, +Z
    backwall = (np.c_[-V[:,0], V[:,1], np.ones(Nv)], F.copy())
    write_mesh(home_dir + 'cbox_back.obj', *backwall)

    # Red wall, +X
    redwall = (np.c_[np.ones(Nv), -V[:,0], V[:,1]], F.copy())
    write_mesh(home_dir + 'cbox_redwall.obj', *redwall)

    # Green wall, -X
    greenwall = (np.c_[-np.ones(Nv), V[:,0], V[:,1]], F.copy())
    write_mesh(home_dir + 'cbox_greenwall.obj', *greenwall)

    # Floor, -Y
    floor = (np.c_[V[:,0], -np.ones(Nv), -V[:,1]], F.copy())
    write_mesh(home_dir + 'cbox_floor.obj', *floor)

    # Ceiling, +Y
    ceiling = (np.c_[V[:,0], np.ones(Nv), V[:,1]], F.copy())
    write_mesh(home_dir + 'cbox_ceiling.obj', *ceiling)

    # Luminiare, +Y
    SCALE = 1.0
    luminiare = (np.c_[0.2 * SCALE * V[:,0], 0.995 * np.ones(Nv), 0.1 * SCALE * V[:,1]], F.copy())
    write_mesh(home_dir + 'cbox_luminaire.obj', *luminiare)

    # Define cubes
    V_, F = icosphere(max(1, N - 1))
    V_ += AxisY[None,:]

    smallcube = (0.3 * V_ + np.array([[-0.5, -1.0, -0.3]]), F.copy())
    write_mesh(home_dir + 'cbox_smallbox.obj', *smallcube)

    largecube = (0.4 * V_ + np.array([[0.4, -1.0, 0.4]]), F.copy())
    write_mesh(home_dir + 'cbox_largebox.obj', *largecube)

# shapes = [backwall, redwall, greenwall, floor, ceiling, luminiare, cube1, cube2]

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    generate(N)
