import numpy as np
import matplotlib.pyplot as plt

def potential_energy(q, eps=1):
    """
    Calculate the potential energy for the Henon-Heiles system.
    The potential energy is given by:
    V = (1/2) * (q[0]^2 + q[1]^2) + eps * (q[0]^2 * q[1] - q[1]^3 / 3)
    Parameters:
    q (list or array-like): Generalized coordinates [q0, q1].
    eps (float, optional): Parameter epsilon, default is 1.
    Returns:
    float: The value of the potential energy.
    """

    return (1/2)*(q[0]**2 + q[1]**2) + eps*((q[0]**2)*q[1] - (q[1]**3)/3)

def hamiltonian(q, p, eps=1):
    """
    Calculate the Hamiltonian for the Henon-Heiles system.
    The Hamiltonian is given by:
    H = (1/2) * (p[0]^2 + p[1]^2 + q[0]^2 + q[1]^2) + eps * (q[0]^2 * q[1] - q[1]^3 / 3)
    Parameters:
    q (list or array-like): Generalized coordinates [q0, q1].
    p (list or array-like): Generalized momenta [p0, p1].
    eps (float, optional): Parameter epsilon, default is 1.
    Returns:
    float: The value of the Hamiltonian.
    """

    return (1/2)*(p[0]**2 + p[1]**2) + potential_energy(q, eps)

def hamiltonian_q_gradient(q, eps=1):
    """
    Calculate the gradient of the Hamiltonian with respect to the generalized coordinates q.
    Parameters:
    q (list or array-like): Generalized coordinates [q0, q1].
    p (list or array-like): Generalized momenta [p0, p1].
    eps (float, optional): Parameter epsilon, default is 1.
    Returns:
    array: The gradient of the Hamiltonian with respect to the generalized coordinates.
    """

    return np.array([q[0] + 2*eps*q[0]*q[1], q[1] + eps*(q[0]**2 - q[1]**2)])

def stormer_verlet(q_n, p_n, h=1e-2):
    """
    Perform one step of the Stormer-Verlet integration method for Hamiltonian systems.
    Parameters:
    q_n (array-like): The generalized coordinates at the current time step.
    p_n (array-like): The generalized momenta at the current time step.
    h (float, optional): The time step size. Default is 1e-2.
    Returns:
    tuple: A tuple containing the updated generalized coordinates and momenta (q_{n+1}, p_{n+1}).
    """

    q_int = q_n + (h/2)*p_n
    dp = -h*hamiltonian_q_gradient(q_int)
    dq = (h/2)*(p_n + dp)
    return q_int + dq, p_n + dp

def simulate_henon_heiles(q0, p0, h=1e-2, n_steps=10000):
    """
    Simulate the Henon-Heiles system using the Stormer-Verlet integration method.
    Parameters:
    q0 (array-like): Initial generalized coordinates [q0, q1].
    p0 (array-like): Initial generalized momenta [p0, p1].
    h (float, optional): The time step size. Default is 1e-2.
    n_steps (int, optional): Number of time steps to simulate. Default is 10000.
    Returns:
    tuple: A tuple containing the generalized coordinates and momenta at each time step.
    """

    q = np.zeros((n_steps, 2))
    p = np.zeros((n_steps, 2))

    q[0] = q0
    p[0] = p0

    for i in range(n_steps-1):
        q[i+1], p[i+1] = stormer_verlet(q[i], p[i], h)
    return q, p

def initial_condition(energy, eps=1):
    """
    Generate a random initial condition for the Henon-Heiles system at a fixed energy.
    Parameters:
    energy (float): The fixed energy value.
    eps (float, optional): Parameter epsilon, default is 1.
    Returns:
    tuple: A tuple containing the generalized coordinates and momenta at the initial condition.
    """
    q0 = np.random.uniform(-1, 1, 2)
    while potential_energy(q0, eps) > energy:
        q0 = np.random.uniform(-1, 1, 2)
    p0 = np.random.uniform(-1, 1, 2)
    p0 = (p0/np.linalg.norm(p0))*np.sqrt(2*(energy - potential_energy(q0, eps)))
    return q0, p0

def create_mask(q, p, accuracy=1e-3):
    return (np.abs(q[:, 0]) < accuracy) & (p[:, 0] > 0)

if __name__=="__main__":
    h = 1e-2
    E = [1/12, 1/8, 1/7]
    n_steps = 1000000

    # question 3.2
    q0, p0 = initial_condition(1/12)
    q, p = simulate_henon_heiles(q0, p0, h, n_steps)
    energy = [hamiltonian(q[i], p[i]) for i in range(n_steps)]
    plt.plot(energy, label='Hamiltonien')
    plt.plot([1/12]*n_steps, label='E = 1/12')
    plt.xlabel('Itérations n')
    plt.ylabel('Energie')
    plt.legend()
    plt.title('Energie du système de Hénon-Heiles')
    plt.show()

    # question 3.3/3.4/3.6
    for energy in E:
        for i in range(50):
            q0, p0 = initial_condition(energy)
            q, p = simulate_henon_heiles(q0, p0, h, n_steps)
            mask = create_mask(q, p)
            plt.plot(q[mask, 1], p[mask, 1], 'o', color='b', markersize=1)
        plt.legend([f'E = {energy:.2f}'])
        plt.xlabel('$q_2$')
        plt.ylabel('$p_2$')
        plt.title('Portrait de phase pour le système de Hénon-Heiles')
        plt.show()
