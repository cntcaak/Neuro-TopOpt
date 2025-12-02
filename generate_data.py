import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import os
import random

# --- CONFIGURATION ---
NUM_SAMPLES = 500   # How many examples to generate
NELX = 60           # Grid width
NELY = 20           # Grid height
VOLFRAC = 0.4       # Volume fraction
PENAL = 3.0
RMIN = 1.5

# Create folders for data if they don't exist
if not os.path.exists('data'):
    os.makedirs('data/loads')
    os.makedirs('data/structures')

# --- OPTIMIZER FUNCTION (The Physics Engine) ---


def optimize_structure(force_node_x, force_node_y):
    # setup
    x = VOLFRAC * np.ones(NELY*NELX, dtype=float)
    xPhys = x.copy()

    # Boundary Conditions: Fix Left Edge
    dofs = np.arange(2*(NELX+1)*(NELY+1))
    fixed = np.union1d(dofs[0:2*(NELY+1):2], np.array([2*(NELX+1)*(NELY+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    # FORCE SETUP: Apply load at specific X, Y coordinate
    # Node index calculation: (NELY+1)*x + y
    # Degrees of freedom: 2 per node. Y-direction is 2*node + 1
    force_node = (NELY+1) * force_node_x + force_node_y
    force_dof = 2 * force_node + 1

    f = np.zeros((2*(NELY+1)*(NELX+1), 1))
    f[force_dof, 0] = -1  # Downward force

    # Pre-calculate Stiffness Matrix Elements
    E, nu = 1, 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                               [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                               [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                               [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                               [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                               [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                               [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                               [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])

    edofMat = np.zeros((NELX*NELY, 8), dtype=int)
    for elx in range(NELX):
        for ely in range(NELY):
            el = ely+elx*NELY
            n1 = (NELY+1)*elx+ely
            n2 = (NELY+1)*(elx+1)+ely
            edofMat[el, :] = np.array(
                [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Optimization Loop
    loop = 0
    change = 1
    while change > 0.1 and loop < 30:  # Faster convergence for data gen
        loop += 1
        sK = ((KE.flatten()[np.newaxis, :]).T *
              (xPhys**PENAL)).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(
            2*(NELY+1)*(NELX+1), 2*(NELY+1)*(NELX+1))).tocsc()

        K = K[free, :][:, free]
        u = np.zeros((2*(NELY+1)*(NELX+1), 1))
        u[free, 0] = spsolve(K, f[free, 0])

        ce = np.dot(u[edofMat].reshape(NELX*NELY, 8), KE)
        c = np.dot(u[edofMat].reshape(NELX*NELY, 8), ce.T)
        dc = -PENAL*(xPhys**(PENAL-1))*np.diag(c)
        # Note: Reshape removed as per previous fix

        # Filter / Update
        l1, l2, move = 0, 100000, 0.2
        while (l2-l1)/(l1+l2 + 1e-10) > 1e-3:
            lmid = 0.5*(l2+l1)
            xnew = np.maximum(0.001, np.maximum(
                xPhys-move, np.minimum(1.0, np.minimum(xPhys+move, xPhys*np.sqrt(-dc/lmid)))))
            if np.sum(xnew) - VOLFRAC*NELX*NELY > 0:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew-x))
        x = xnew
        xPhys = x

    return xPhys.reshape((NELX, NELY)).T  # Return as 20x60 grid


# --- MAIN GENERATION LOOP ---
print(f"Generating {NUM_SAMPLES} samples...")

for i in range(NUM_SAMPLES):
    try:
        # 1. Randomize Load Location
        rand_x = random.randint(5, NELX)
        rand_y = random.randint(0, NELY)

        # 2. Create Input Image
        input_grid = np.zeros((NELY, NELX))
        if rand_y < NELY and rand_x < NELX:
            input_grid[rand_y, rand_x-1] = 1.0

        # 3. Run Physics Engine
        print(f"Sample {i+1}/{NUM_SAMPLES} | Force at X:{rand_x} Y:{rand_y}")
        structure_grid = optimize_structure(rand_x, rand_y)

        # 4. Save Data
        np.save(f'data/loads/{i}.npy', input_grid)
        np.save(f'data/structures/{i}.npy', structure_grid)

    except Exception as e:
        print(f"!!! Error on Sample {i+1}: {e} -> SKIPPING !!!")
        continue

print("Data Generation Complete!")
