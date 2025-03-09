# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 21:14:28 2025

@author: GREY
"""
#%% imports
import numpy as np
import matrixmethod as mm
import importlib

#%% draw structure
mm.Node.clear()

nodes = []
nodes.append(mm.Node(0, 0))
nodes.append(mm.Node(2, 0))
nodes.append(mm.Node(1, -1))


elems = []
elems.append(mm.Element_truss(nodes[0], nodes[1]))
elems.append(mm.Element_truss(nodes[1], nodes[2]))
elems.append(mm.Element_truss(nodes[2], nodes[0]))

section = {}
section['EI'] = 4000
section['EA'] = 1000
for elem in elems:
    elem.set_section(section)

con = mm.Constrainer()
con.fix_node(nodes[0])
con.fix_dof(nodes[1],1,0)
con.fix_dof(nodes[1],2,0)
con.fix_dof(nodes[2],2,0) # fix the yy dof of node 2

mm.Node.add_load(nodes[-1],[0,100,0])

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
for elem in elems:
    elem.plot_displaced(con.full_disp(u_free)[elem.global_dofs()],100, global_c=True, scale=1)