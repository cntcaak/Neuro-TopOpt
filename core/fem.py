import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def get_stiffness_matrix(E, nu):
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu /
                 8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return E/(1-nu**2)*np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                 [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                 [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                 [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                 [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                 [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                 [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                 [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])


def run_fem_validation(density_grid, load_x, load_y, load_magnitude=1000):
    """
    Runs FEA. Returns Displacement, Von Mises Stress proxy, and Safety Factor Map.
    load_magnitude: Force in Newtons.
    """
    nelx, nely = 60, 20
    # Young's Modulus (e.g., Plastic/Metal mix for simulation scaling)
    E0 = 1e9
    Emin = 1e-9
    nu = 0.3

    # 1. Setup Stiffness
    KE = get_stiffness_matrix(E0, nu)
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

    xPhys = density_grid.T.flatten()
    sK = ((KE.flatten()[np.newaxis, :]).T *
          (Emin + xPhys**3 * (E0-Emin))).flatten(order='F')
    K = coo_matrix((sK, (iK, jK)), shape=(
        2*(nely+1)*(nelx+1), 2*(nely+1)*(nelx+1))).tocsc()

    # 2. Apply Load
    dofs = np.arange(2*(nelx+1)*(nely+1))
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    f = np.zeros((2*(nely+1)*(nelx+1), 1))
    force_node = (nely+1) * load_x + load_y
    f[2*force_node + 1, 0] = -load_magnitude  # Apply Variable Load

    # 3. Solve
    K = K[free, :][:, free]
    u = np.zeros((2*(nely+1)*(nelx+1), 1))
    try:
        u[free, 0] = spsolve(K, f[free, 0])
    except:
        return None, None, None

    # 4. Process Results
    u_x = u[0::2]
    u_y = u[1::2]
    disp_mag = np.sqrt(u_x**2 + u_y**2)
    disp_grid = disp_mag[:nelx*nely].reshape((nelx, nely)).T

    # Calculate Stress (Strain Energy Density)
    ce = np.dot(u[edofMat].reshape(nelx*nely, 8), KE)
    c = np.dot(u[edofMat].reshape(nelx*nely, 8), ce.T)
    stress_grid = np.diag(c).reshape((nelx, nely)).T

    return disp_grid, stress_grid
