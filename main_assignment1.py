# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 21:14:28 2025

@author: GREY
"""
#%% imports
import numpy as np
import matrixmethod as mm
import importlib
import matplotlib.pyplot as plt

#%% draw structure
mm.Node.clear()

#%% inputs
l1 = 11.75 #m
l2 = 4 #m
nodes = []
nodes.append(mm.Node(0, 0))
nodes.append(mm.Node(l1, 0))
nodes.append(mm.Node(2*l1, 0))
nodes.append(mm.Node(3*l1, 0))
nodes.append(mm.Node(4*l1, 0))

nodes.append(mm.Node(4*l1+l2, 0))
nodes.append(mm.Node(4*l1+l2, l2)) 
nodes.append(mm.Node(4*l1+l2, -l2))
nodes.append(mm.Node(4*l1,-l2))
nodes.append(mm.Node(4*l1,-2*l2))
nodes.append(mm.Node(4*l1+l2,-2*l2))
nodes.append(mm.Node(4*l1+l2,-3*l2))
nodes.append(mm.Node(4*l1,-3*l2))
nodes.append(mm.Node(4*l1,-4*l2))


elems = []
elems.append(mm.Element(nodes[0], nodes[1]))
elems.append(mm.Element(nodes[1], nodes[2]))
elems.append(mm.Element(nodes[2], nodes[3]))
elems.append(mm.Element_onehinge(nodes[3],nodes[4]))
elems.append(mm.Element_truss(nodes[3], nodes[9]))
elems.append(mm.Element_truss(nodes[4], nodes[5]))
elems.append(mm.Element_truss(nodes[4], nodes[6]))
elems.append(mm.Element_truss(nodes[5], nodes[6]))
elems.append(mm.Element_truss(nodes[4], nodes[7]))
elems.append(mm.Element_truss(nodes[5], nodes[7]))
elems.append(mm.Element_truss(nodes[4], nodes[8]))
elems.append(mm.Element_truss(nodes[7], nodes[8]))
elems.append(mm.Element_truss(nodes[8], nodes[9]))
elems.append(mm.Element_truss(nodes[7], nodes[9]))
elems.append(mm.Element_truss(nodes[9], nodes[10]))
elems.append(mm.Element_truss(nodes[7], nodes[10]))
elems.append(mm.Element_truss(nodes[10], nodes[11]))
elems.append(mm.Element_truss(nodes[9], nodes[11]))
elems.append(mm.Element_truss(nodes[9], nodes[12]))
elems.append(mm.Element_truss(nodes[11], nodes[12]))
elems.append(mm.Element_truss(nodes[11], nodes[13]))
elems.append(mm.Element_truss(nodes[12], nodes[13]))
elems.append(mm.Element_truss(nodes[1], nodes[13]))
elems.append(mm.Element_truss(nodes[2], nodes[12]))

section = {}
section['EI'] = 400e6
section['EA'] = 1.3e9

for elem in elems:
    elem.set_section(section)

con = mm.Constrainer()
con.fix_dof(nodes[0],1,0.1) # fix the z dof of node 2
con.fix_dof(nodes[6],0,0) # fix the z dof of node 2
con.fix_dof(nodes[6],1,0) # fix the z dof of node 2
con.fix_dof(nodes[4],2,0)
for node in nodes[5:14]:
    con.fix_dof(node,2,0) # fix the z dof of node 2


elems[0].add_distributed_load([0,20e3])
elems[1].add_distributed_load([0,20e3])
elems[2].add_distributed_load([0,20e3])
elems[3].add_distributed_load([0,20e3])
nodes[-3].add_load([400e3,300e3,0])
nodes[-4].add_load([400e3,300e3,0])
nodes[7].add_load([400e3,300e3,0])
global_f=np.zeros(3*len(nodes))
global_k = np.zeros ((3*len(nodes), 3*len(nodes)))

for e in elems:
    elmat = e.stiffness()
    idofs = e.global_dofs()
    global_k[np.ix_(idofs,idofs)] += elmat
for node in nodes:
    global_f[node.dofs]+=node.p

Kff,Kf=con.constrain(global_k,global_f)

#%% solve for free dofs
u_free=np.matmul(np.linalg.inv(Kff),Kf)
print(u_free)

#%% solve for support reactions
print(con.support_reactions(global_k,u_free,global_f))

#%% postprocessing
plt.figure()
for elem in elems:
    elem.plot_displaced(con.full_disp(u_free)[elem.global_dofs()],20,True,scale=8)
plt.figure()
for elem in elems:
    elem.plot_moment_diagram(con.full_disp(u_free)[elem.global_dofs()],500,True,scale=1e-5)
#%% check nodal results