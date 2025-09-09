# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 22:39:12 2023

@author: mikes
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import LineCollection
matplotlib.rcParams['font.size'] = 16


import numpy as np


from fipy import Gmsh3D


def make_mesh(radius, cellSize):
    # Makes a roughly uniform spherical mesh
    mesh = Gmsh3D('''

        cellSize = %(cellSize)g;

        radius = %(radius)g;
    
    
        // create hemisphere
    
        Point(1) = {0, 0, 0, cellSize};
        Point(2) = {0, radius, 0, cellSize};
        Point(3) = {radius, 0, 0, cellSize};
        Point(4) = {0, 0, radius, cellSize};
        Point(5) = {0, -radius, 0, cellSize};
    
        Circle(1) = {2, 1, 3};
        Circle(2) = {3, 1, 4};
        Circle(3) = {4, 1, 2};
        
        Circle(4) = {3, 1, 5};
        Circle(5) = {5, 1, 4};
        
        Line(7) = {2, 1};
        Line(8) = {1, 5};
    
        Line Loop(1) = {1, 2, 3};
        Line Loop(2) = {4, 5, -2};
        Line Loop(3) = {5, 3, 7, 8};
        Line Loop(4) = {1, 4, -8, -7};
        
        Surface(1) = {1};
        Surface(2) = {2};
        Surface(3) = {3};
        Surface(4) = {4};

        Surface Loop(100)={1, 2, 3, 4};
        
        Volume(100) = {100};
    
    ''' % locals(), overlap=2)
    
    verts = mesh.vertexCoords.T
    
    edges = []
    outer = []
    
    for face in range(mesh.faceVertexIDs.shape[1]):
        # Extract springs from face triangles
        vs = mesh.faceVertexIDs[:, face]
        
        for i in range(3):
            v1, v2 = vs[i], vs[(i+1) % 3]
            
            if (v1, v2) not in edges and (v2, v1) not in edges:
                edges.append((v1, v2))
                # outer.append(mesh.exteriorFaces.value[face])
                
    edges = np.array(edges)
    

    r = np.sum(verts ** 2, axis=1) ** 0.5
    surface_verts = r > radius * 0.999
    y_boundary = verts[:, 1] < 1e-2
    z_boundary = verts[:, 2] < 1e-2
    
    outer = surface_verts[edges[:, 0]] * surface_verts[edges[:, 1]]

    return verts, edges, outer#, y_boundary, z_boundary


def get_mesh(radius=1, cellSize=0.25):
    # Loads premade meshes to save time
    
    file_name = 'meshes/quarter_r_{:.1f}_cellSize_{:.2f}'.format(radius, cellSize)
    
    try:
        verts = np.load(file_name+'_verts.npy')
        edges = np.load(file_name+'_edges.npy')
        outer = np.load(file_name+'_outer.npy')
            
        print('Mesh loaded')
    except:
        verts, edges, outer = make_mesh(radius, cellSize)
        
        np.save(file_name+'_verts.npy', verts)
        np.save(file_name+'_edges.npy', edges)
        np.save(file_name+'_outer.npy', outer)

    return verts, edges, outer
    


def get_length_and_directions(verts, edges):
    # Calculate length and direction of edges
    
    dx = verts[edges[:, 1]] - verts[edges[:, 0]]
    length = np.sum(dx ** 2, axis=1) ** 0.5
    
    # Hacky length trick - direction will still be zero
    length[length == 0] = 1000
    
    direction = dx / length[:, np.newaxis]
    
    return length, direction



def force(verts, edges, stress, length=None, direction=None,
          vert_weight=1, edge_weight=1, x_boundary=False, z_boundary=False):
    # Calculate the force on each vertex - uses symmetries to speed up
    
    if length is None or direction is None:
        length, direction = get_length_and_directions(verts, edges)
    
    vertex_force = np.zeros(verts.shape)
    
    edge_force = stress * edge_weight
    
    for i in range(3):
        np.add.at(vertex_force[:, i], edges[:, 0], -direction[:, i] * edge_force)
        np.add.at(vertex_force[:, i], edges[:, 1], direction[:, i] * edge_force)

    
    # Set forces across symmetry planes to zero
    vertex_force[x_boundary, 0] = 0
    vertex_force[z_boundary, 2] = 0
    
    
    # Also stop vertices crossing boundaries
    is_left = verts[:, 0] < 0
    is_back = verts[:, 2] < 0
    
    vertex_force[is_left & (vertex_force[:, 0] < 0), 0] = 0
    vertex_force[is_back & (vertex_force[:, 2] < 0), 2] = 0
    
    verts[is_left, 0] = 0
    verts[is_back, 2] = 0
    

    return vertex_force / vert_weight[:, np.newaxis]



def draw_shell(verts, edges, outer_edges, tension, savename=None, interphase=False):
    # Draw the simulation
    
    fig, axes = plt.subplots(1, 3)
    
    for j in range(3):
        
        segments = []
        colors = []
        
        segments_cable = []
        colors_cable = []
        
        
        for i in range(len(edges)):
            
            if not outer_edges[i]:
                continue

            
            if j < 2:
                if verts[edges[i, 0], 2 - 2 * j] < -1 and verts[edges[i, 1], 2 - 2 * j] < -1:
                    continue
                
                x1, x2 = verts[[edges[i, 0], edges[i, 1]], 2 * j]
                y1, y2 = verts[[edges[i, 0], edges[i, 1]], 1]
            else:
                if verts[edges[i, 0], 1] < 0 and verts[edges[i, 1], 1] < 0 and tension[i] == 0:
                    continue
                
                x1, x2 = verts[[edges[i, 0], edges[i, 1]], 0]
                y1, y2 = verts[[edges[i, 0], edges[i, 1]], 2]
            
            
            if tension[i] != 0:
                
                col = (-tension[i] / cable_tension / 2 + 0.5,
                       0.5 + tension[i] / cable_tension / 2,
                       0.5 + tension[i] / cable_tension / 2)

                segments_cable.append(((x1, y1), (x2, y2)))
                colors_cable.append(col)
            else:
                col = (0.5, 0.5, 0.5)
                
                segments.append(((x1, y1), (x2, y2)))
                colors.append(col)
 
        if interphase:
            axes[0].set_title('Interphase: Less Fluid')
        else:
            axes[0].set_title('M-Phase: More Fluid')

        if j < 2:
            axes[j].set_ylim(0, 400)
        else:
            axes[j].set_ylim(-400, 400)
            
        axes[j].set_aspect('equal', adjustable='datalim')
        
        lc = LineCollection(segments, colors=colors, linewidths=3)
        axes[j].add_collection(lc)
        
        lc2 = LineCollection(segments_cable, colors=colors_cable, linewidths=3)
        axes[j].add_collection(lc2)
    
        
    # plt.axis('equal')
    fig.set_size_inches(18, 6)
    
    if savename is not None:
        fig.savefig(savename + '.png', format='png', bbox_inches='tight')
    plt.show()



def run_sim(E_inter, eta1_inter, eta2_inter,
            E_m, eta1_m, eta2_m,
            cable_tension, savename=None):
    
    # Runs the simulation
    
    # Get mesh
    verts, edges, outer_edges = get_mesh(radius=R, cellSize=cellSize * R)
    
    velocity = np.zeros(verts.shape)

    
    r = np.sum(verts ** 2, axis=1) ** 0.5
    length, direction = get_length_and_directions(verts, edges)

    
    # Set up symmetries
    x_boundary = verts[:, 0] < 1e-2
    z_boundary = verts[:, 2] < 1e-2
    
    # How many times the vertex or edges are mirrored
    vert_weight = (2 - x_boundary) * (2 - z_boundary)
    edge_weight = vert_weight[edges].max(axis=1)
    
    # Initialise variables
    band_amount = np.zeros(edges.shape[0])
    l0 = length
    
    tension = np.zeros(edges.shape[0])
    
    stress = np.zeros(edges.shape[0])
    stress_old = np.zeros(edges.shape[0])
    
    eps = np.zeros(edges.shape[0])
    eps_old = np.zeros(edges.shape[0])
    
    d_eps = np.zeros(edges.shape[0])
    d_eps_old = np.zeros(edges.shape[0])
    
    # Find middle vertices and edges 
    middle_verts = np.abs(verts[:, 0]) < 1
    allowed_verts = middle_verts
    
    middle_edges = allowed_verts[edges[:, 0]] * allowed_verts[edges[:, 1]] * outer_edges
    
    # Within some range
    theta = np.arccos(verts[:, 1] / r)
    
    
    # Get edges from pole to pole - used for growing the band
    middle_edge_ids = [i for i in range(len(edges)) if middle_edges[i]]
    middle_edge_thetas = [min(theta[edges[mid, 0]], theta[edges[mid, 1]]) for mid in middle_edge_ids]
    middle_edge_ids = [x for _, x in sorted(zip(middle_edge_thetas, middle_edge_ids))]
    
    E = E_inter
    eta1 = eta1_inter
    eta2 = eta2_inter
    


    def get_stress(v, E, eta1, eta2, length=None, direction=None):    
        # Calculates the stress on each edge
        
        if length is None or direction is None:
            length, direction = get_length_and_directions(v, edges)
        
        eps = (length - l0) / l0
        
        d_eps = (eps - eps_old) / dt
        dd_eps = (d_eps - d_eps_old) / dt
        d_stress =  (-(stress_old) + eta2 * d_eps + eta1 * eta2 / E * dd_eps) / ((eta1 + eta2) / E)
        
        stress = stress_old + d_stress * dt
        
        return stress, eps, d_eps
    


    def force_balance(verts, E, eta1, eta2):
        # Approximately balances forces using an interative approach
        
        new_verts = verts + dt * velocity
        
        mini_dt = dt

        old_max = 1000
        for i in range(1000000):
            length, direction = get_length_and_directions(new_verts, edges)
            stress, _, _ = get_stress(new_verts, E, eta1, eta2, length, direction)
            f = force(new_verts, edges, stress + tension, length, direction, vert_weight, edge_weight, x_boundary, z_boundary)
            
            new_verts -= f * mini_dt
            
            if f.max() < 1e-5:
                break
            
            # Use a finer step to prevent oscillations
            if f.max() > old_max:
                # Undo motion and reduce timestep
                new_verts += f * mini_dt
                mini_dt /= 2
                
                # Stop when tiny timestep is needed
                if mini_dt < 1e-12:
                    break
                
            else:
                old_max = f.max()
            
        # print('Iterations:', i, 'max force', f.max())
        return new_verts

        
    
    
    # Now we run the simulation
    t = 0
    time_step = 0
    
    ts = []
    displacements = []
    z_positions = []
    band_lengths = []
    
    vert_positions = []
    edge_tensions = []
    
    shortest_length = []
    while t <= duration:
        
        if savename is not None:
            vert_positions.append(verts.copy())
            edge_tensions.append(tension.copy())
        
        # in_interphase = (t < interphase_duration)
        if t > interphase_duration:
            E = E_m
            eta1 = eta1_m
            eta2 = eta2_m
        
        length, direction = get_length_and_directions(verts, edges)
        
        displacements.append((verts[:, 1].max() - verts[middle_verts, 1].max()))
        z_positions.append(verts[middle_verts, 1].max())
        band_lengths.append(np.sum(length * band_amount))
        ts.append(t)
        
        shortest_length.append(length.min())
        
        # Update band position

        # Slow list use? But not the bottleneck
        new_band_length = dt * actin_v
        if time_step == 0:
            new_band_length = 0


        for edge_id in middle_edge_ids:

            if band_amount[edge_id] == 1:
                continue

            if band_amount[edge_id] * length[edge_id] + new_band_length > length[edge_id]:
                new_band_length -= (1 - band_amount[edge_id]) * length[edge_id]
                band_amount[edge_id] = 1
            else:
                band_amount[edge_id] += new_band_length / length[edge_id]
                break
        
        tension = band_amount * cable_tension
        
        
        # Now apply forces for one timestep and recalculate stuff
        new_verts = force_balance(verts, E, eta1, eta2)
        
        velocity = (new_verts - verts) / dt
        verts = new_verts

        # verts = energy_minimise(verts, E, eta1, eta2)
        stress, eps, d_eps = get_stress(verts, E, eta1, eta2)
  
        # Update old values
        stress_old = stress
        eps_old = eps
        d_eps_old = d_eps

        t += dt
        time_step += 1
        print(t, stress.max(), length.min())
    
    
    if savename is not None:
        np.save('data/' + savename + '_verts.npy', vert_positions)
        np.save('data/' + savename + '_tensions.npy', edge_tensions)

    
    return ts, displacements, z_positions, band_lengths


if __name__ == '__main__':

    """ Experimental parameters """ 
    

    # Cell radius
    R = 350 
    
    # gmesh size of each "cell" meaning volume in the mesh
    cellSize = 0.1
    
    # Actin growth speed per min
    actin_v = 25
    
    
    # Interphase and Mphase time
    interphase_duration = 15
    mphase_duration = 10

    actin_dtheta_dt = actin_v / R
    


    """ Simulation parameters """
    dt = 0.01
    t = 0
    duration = interphase_duration + mphase_duration

    
    """ Estimated parameters """

    cable_tension = 2
    
    E_inter = 51.46474860443133
    E_m = 32.13624503389098
    
    eta1_inter = 66.13304767230532 / 60
    eta1_m = 27.553645048847265 / 60
    
    eta2_inter = 249.41799758719338 / 60
    eta2_m = 58.754299539013516 / 60
    
    
    
    t, displacement, z_position, length = run_sim(E_inter, eta1_inter, eta2_inter,
                                      E_m, eta1_m, eta2_m,
                                      cable_tension,
                                      savename='wt')

    np.savetxt('data/wt_displacement_tension_{:.2f}_growth_{:.2f}.csv'.format(cable_tension, actin_v), displacement, delimiter=',')
    np.savetxt('data/wt_length_tension_{:.2f}_growth_{:.2f}.csv'.format(cable_tension, actin_v), length, delimiter=',')
    np.savetxt('data/wt_z_position_tension_{:.2f}_growth_{:.2f}.csv'.format(cable_tension, actin_v), z_position, delimiter=',')
    

    
