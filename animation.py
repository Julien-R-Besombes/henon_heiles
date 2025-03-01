import henon_heiles as h
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__=="__main__":
    step_size = 1e-2
    frames = 20
    E = np.logspace(np.log10(1/12), np.log10(1/7), frames)
    n_steps = 1000000
    n_initial_conditions = 50
    
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o', color='b', markersize=1)
    ax.set_xlim(-1/2, 1/2)
    ax.set_ylim(-1/2, 1/2)
    ax.set_xlabel('$q_2$')
    ax.set_ylabel('$p_2$')
    ax.set_title('Portrait de phase pour le système de Hénon-Heiles')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        energy = E[frame]
        q_data = []
        p_data = []
        for i in range(n_initial_conditions):
            q0, p0 = h.initial_condition(energy)
            q, p = h.simulate_henon_heiles(q0, p0, step_size, n_steps)
            mask = h.create_mask(q, p)
            q_data.extend(q[mask, 1])
            p_data.extend(p[mask, 1])
        line.set_data(q_data, p_data)
        ax.legend([f'E = {energy:.2f}'])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
    ani.save('henon_heiles.gif', writer='imagemagick', fps=10/frames)
