import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# --- PARAMETERS ---
nelx = 60       # Number of elements in X direction (Width)
nely = 20       # Number of elements in Y direction (Height)
volfrac = 0.4   # Volume fraction (We want to keep 40% of material)
penal = 3.0     # Penalization power (Standard for SIMP method)
rmin = 1.5      # Filter radius (prevents checkerboard patterns)
ft = 1          # Filter type

# --- FINITE ELEMENT ANALYSIS FUNCTIONS ---


def lk():
    E = 1
    nu = 0.3
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
    return KE

# --- MAIN OPTIMIZATION LOOP ---


def main(nelx, nely, volfrac, penal, rmin, ft):
    print(f"Starting Optimization: Grid {nelx}x{nely}")

    # Initial Density (Start with a solid block of uniform density)
    x = volfrac * np.ones(nely*nelx, dtype=float)
    xold = x.copy()
    xPhys = x.copy()

    # Boundary Conditions (Cantilever Beam)
    # Fix the left side of the mesh
    dofs = np.arange(2*(nelx+1)*(nely+1))
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    # Force Vector (Load applied at bottom right)
    f = np.zeros((2*(nely+1)*(nelx+1), 1))
    f[2*(nelx+1)*(nely+1)-2, 0] = -1  # Downward force

    # FEA Setup
    KE = lk()
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely+elx*nely
            n1 = (nely+1)*elx+ely
            n2 = (nely+1)*(elx+1)+ely
            edofMat[el, :] = np.array(
                [2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])

    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()

    # Iteration Loop
    loop = 0
    change = 1
    while change > 0.04 and loop < 40:  # Max 40 loops or until convergence
        loop += 1

        # Setup Stiffness Matrix
        sK = ((KE.flatten()[np.newaxis, :]).T *
              (xPhys**penal)).flatten(order='F')
        K = coo_matrix((sK, (iK, jK)), shape=(
            2*(nely+1)*(nelx+1), 2*(nely+1)*(nelx+1))).tocsc()

        # Solve System
        K = K[free, :][:, free]
        u = np.zeros((2*(nely+1)*(nelx+1), 1))
        u[free, 0] = spsolve(K, f[free, 0])

        # Sensitivity Analysis (Calculating where stress is high)
        ce = np.dot(u[edofMat].reshape(nelx*nely, 8), KE)
        c = np.dot(u[edofMat].reshape(nelx*nely, 8), ce.T)
        dc = -penal*(xPhys**(penal-1))*np.diag(c)

        # Update Densities (Optimality Criteria)
        l1, l2, move = 0, 100000, 0.2
        while (l2-l1)/(l1+l2) > 1e-3:
            lmid = 0.5*(l2+l1)
            xnew = np.maximum(0.001, np.maximum(
                xPhys-move, np.minimum(1.0, np.minimum(xPhys+move, xPhys*np.sqrt(-dc/lmid)))))
            if np.sum(xnew) - volfrac*nelx*nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew-x))
        x = xnew
        xPhys = x

        print(f" It.: {loop} | Change: {change:.3f}")

    # Plotting result
    plt.figure(figsize=(10, 4))
    plt.imshow(-xPhys.reshape((nelx, nely)).T,
               cmap='gray', interpolation='nearest')
    plt.title("Optimized Structure (Ground Truth)")
    plt.axis('off')
    plt.savefig(f'optimized_beam.png')
    plt.show()
    print("Optimization Complete. Image saved as 'optimized_beam.png'")


if __name__ == "__main__":
    main(nelx, nely, volfrac, penal, rmin, ft)
