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

nodes = []
nodes.append(mm.Node(0, 0))
nodes.append(mm.Node(2, 0))
nodes.append(mm.Node(4, 0))


elems = []
elems.append(mm.Element(nodes[0], nodes[1]))
elems.append(mm.Element_onehinge(nodes[1], nodes[2]))

section1 = {}
section1['EI'] = 4
section1['EA'] = 1

for elem in elems:
    elem.set_section(section1)

con = mm.Constrainer()
con.fix_dof(nodes[0],1,0) # fix the z dof of node 2
con.fix_dof(nodes[2],0,0) # fix the z dof of node 2
con.fix_dof(nodes[2],1,0) # fix the z dof of node 2
con.fix_dof(nodes[2],2,0) # fix the z dof of node 2

elems[0].add_distributed_load([0,2])
elems[1].add_distributed_load([0,2])

global_f=np.zeros(3*len(nodes))
global_k = np.zeros ((3*len(nodes), 3*len(nodes)))
for e in elems:
    elmat = e.stiffness()
    print(elmat)
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
    elem.plot_moment_diagram(con.full_disp(u_free)[elem.global_dofs()],100,True,scale=1)
plt.figure()
for elem in elems:
    elem.plot_displaced(con.full_disp(u_free)[elem.global_dofs()],100,True,scale=1)
