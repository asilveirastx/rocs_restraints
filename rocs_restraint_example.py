import time
from simtk.openmm import XmlSerializer
from simtk.openmm import app
import simtk.openmm as openmm
from itertools import combinations
from simtk import unit
import numpy as np
import mdtraj as md

def pairs_overlap(ri, rj, alphai, alphaj, p=2.7):

    volume = p**2*(np.pi/(alphai+alphaj))**(3/2)*np.exp(-alphai*alphaj*(np.linalg.norm(ri - rj)**2)/(alphai + alphaj))
    return volume

def triplets_overlap(ri, rj, rk, alphai, alphaj, alphak, p=2.7):

    volume = p**3*(np.pi/(alphai+alphaj+alphak))**(3/2)*np.exp(-(alphai*alphaj*(np.linalg.norm(ri - rj)**2) + alphai*alphak*(np.linalg.norm(ri - rk)**2) + alphaj*alphak*(np.linalg.norm(rj-rk)**2))/(alphai + alphaj + alphak))
    return volume

def quartets_overlap(ri, rj, rk, rl, alphai, alphaj, alphak, alphal, p=2.7):

    volume=p**4*(np.pi/(alphai+alphaj+alphak+alphal))**(3/2)*np.exp(-( alphai*alphaj*(np.linalg.norm(ri - rj)**2) + alphai*alphak*(np.linalg.norm(ri - rk)**2) + alphaj*alphak*(np.linalg.norm(rj - rk)**2) + alphai*alphal*(np.linalg.norm(ri - rl)**2) + alphaj*alphal*(np.linalg.norm(rj - rl)**2) + alphak*alphal*(np.linalg.norm(rk - rl)**2))/(alphai + alphaj + alphak + alphal))
    return volume

start = time.time()
with open('./system.xml', 'r') as infile:
    system = XmlSerializer.deserialize(infile.read())
pdb = app.PDBFile('./ligands.pdb')
topo = md.Topology.from_openmm(pdb.topology)

lig1 = topo.select("resname l10").tolist()
lig2 = topo.select("resname lg3").tolist()
radius = {}
alpha = {}

kappa = np.pi/(1.5514**(2./3.))
for force in system.getForces():
    if isinstance(force, openmm.NonbondedForce):
        for index in range(force.getNumParticles()):
            if index in (lig1 + lig2):
                (q, sig, eps) = force.getParticleParameters(index)
                if sig.in_units_of(unit.nanometers) < 0.1*unit.nanometers:
                    sig = 0.1*unit.nanometers
                radius[index] = sig/2.0
                alpha[index] = kappa/((sig/2.0)**2)

lig1_heavy_atoms  = topo.select("resname l10 and mass > 2.0").tolist()
lig2_heavy_atoms  = topo.select("resname lg3 and mass > 2.0").tolist()

# Recompute radius for coarse-grain heavy atoms

n_hydrogens = {} # number of hydrogens bonded to each heavy atom
index_hydrogens = {} # list of indices of hydrogens bonded to each heavy atom
for i in lig1_heavy_atoms + lig2_heavy_atoms:
    n_hydrogens[i] = 0
    index_hydrogens[i] = []
    for bond in pdb.topology.bonds():
        if i in [bond[0].index, bond[1].index]:
            if bond[0].element.symbol == 'H':
                n_hydrogens[i] += 1
                index_hydrogens[i].append(bond[0].index)
            elif bond[1].element.symbol=='H':
                n_hydrogens[i] += 1
       	       	index_hydrogens[i].append(bond[1].index)

    if n_hydrogens[i] > 0:
        if n_hydrogens[i] == 1:
            ri = pdb.positions[i].in_units_of(unit.nanometers)
            rj = pdb.positions[index_hydrogens[i][0]].in_units_of(unit.nanometers)
            j = index_hydrogens[i][0]
            overlap = pairs_overlap(ri, rj, alpha[i], alpha[j])
            volume_sum = (4./3.)*np.pi*(radius[i]**3 + radius[j]**3)
            radius[i] = ((volume_sum - overlap)*(3./(4.0*np.pi)))**(1./3.)

        elif n_hydrogens[i] == 2:
            ri = pdb.positions[i].in_units_of(unit.nanometers)
            j = index_hydrogens[i][0]
            k = index_hydrogens[i][1]
            rj = pdb.positions[j].in_units_of(unit.nanometers)
            rk = pdb.positions[k].in_units_of(unit.nanometers)
            overlap = pairs_overlap(ri, rj, alpha[i], alpha[j])
            overlap += pairs_overlap(ri, rk, alpha[i], alpha[k])
            overlap += pairs_overlap(rj, rk, alpha[j], alpha[k])
            overlap -= triplets_overlap(ri, rj, rk, alpha[i], alpha[j], alpha[k])
            volume_sum = (4./3.)*np.pi*(radius[i]**3 + radius[j]**3 +  radius[k]**3)
            radius[i] = ((volume_sum - overlap)*(3./(4.0*np.pi)))**(1./3.)

        elif n_hydrogens[i] == 3:
            pairs = [(i, j) for i, j in combinations([i, index_hydrogens[i][0], index_hydrogens[i][1], index_hydrogens[i][2]], 2)]
            triplets = [(i, j, k) for i, j, k in combinations([i, index_hydrogens[i][0], index_hydrogens[i][1], index_hydrogens[i][2]], 3)]
            quartets = [(i, j, k, l) for i, j, k, l in combinations([i, index_hydrogens[i][0], index_hydrogens[i][1], index_hydrogens[i][2]], 4)]
            overlap = 0*unit.nanometers**3
            for p in pairs:
                ri = pdb.positions[p[0]].in_units_of(unit.nanometers)
                rj = pdb.positions[p[1]].in_units_of(unit.nanometers)
                overlap += pairs_overlap(ri, rj, alpha[p[0]], alpha[p[1]])
            for t in triplets:
                ri = pdb.positions[t[0]].in_units_of(unit.nanometers)
       	        rj = pdb.positions[t[1]].in_units_of(unit.nanometers)
                rk = pdb.positions[t[2]].in_units_of(unit.nanometers)
       	        overlap -= triplets_overlap(ri, rj, rk, alpha[t[0]], alpha[t[1]], alpha[t[2]])
            for q in quartets:
                ri = pdb.positions[q[0]].in_units_of(unit.nanometers)
                rj = pdb.positions[q[1]].in_units_of(unit.nanometers)
       	        rk = pdb.positions[q[2]].in_units_of(unit.nanometers)
                rl = pdb.positions[q[3]].in_units_of(unit.nanometers)
                overlap += quartets_overlap(ri, rj, rk, rl, alpha[q[0]], alpha[q[1]], alpha[q[2]], alpha[q[3]])

            volume_sum = (4./3.)*np.pi*(radius[i]**3 + radius[index_hydrogens[i][0]]**3 + radius[index_hydrogens[i][1]]**3 + radius[index_hydrogens[i][2]]**3)
            radius[i] = ((volume_sum - overlap)*(3./(4.0*np.pi)))**(1./3.)
            alpha[i] = kappa/((radius[i] )**2)

heavy_atoms = lig1_heavy_atoms + lig2_heavy_atoms

atom_index = {atom:i for i, atom in enumerate(heavy_atoms)}
is_neighbor = np.zeros([len(heavy_atoms)]*2, dtype=bool)

epsilon = 0.3*unit.nanometers
neighbors = []
for index, i in enumerate(heavy_atoms):
    for j in heavy_atoms[(index+1):]:
        rij = np.linalg.norm(pdb.positions[i].in_units_of(unit.nanometers) - pdb.positions[j].in_units_of(unit.nanometers))
        radius_sum = radius[i] + radius[j]
        if rij <= (radius_sum + epsilon):
            neighbors.append(sorted([i, j]))
            is_neighbor[atom_index[i], atom_index[j]] = True

# pairs, where the indices belong to different molecules
pairs = []
for n in neighbors:
    if not (n[0] in lig1_heavy_atoms and n[1] in lig1_heavy_atoms):
        if not (n[0] in lig2_heavy_atoms and n[1] in lig2_heavy_atoms):
            pairs.append(n)
triplets = []
for p in pairs:
    for k in heavy_atoms:
        if (k not in p) and all(is_neighbor[atom_index[atom], atom_index[k]] for atom in p):
            triplets.append([p[0], p[1], k])

quartets = []
for t in triplets:
    for k in heavy_atoms:
        if (k not in t) and all(is_neighbor[atom_index[atom], atom_index[k]] for atom in t):
            quartets.append([t[0], t[1], t[2], k])

quintents = []
for q in quartets:
    for k in heavy_atoms:
        if (k not in q) and all(is_neighbor[atom_index[atom], atom_index[k]] for atom in q):
            quintents.append([q[0], q[1], q[2], q[3], k])


end = time.time()

print(f"Runtime is {end - start}")

height = 1
pairs_energy = 'height*p^2*(pi/(alpha1+alpha2))^(3/2)*exp(-alpha1*alpha2*distance(p1,p2)^2/(alpha1 + alpha2)); p=2.7; pi=3.14159265'
pairs_vol = openmm.CustomCompoundBondForce(2, pairs_energy)
pairs_vol.addPerBondParameter('alpha1')
pairs_vol.addPerBondParameter('alpha2')
pairs_vol.addGlobalParameter('height', height)
for pair in pairs:
    pairs_vol.addBond([pair[0], pair[1]], [alpha[pair[0]], alpha[pair[1]]])

triplets_energy= '-height*p^3*(pi/(alpha1+alpha2+alpha3))^(3/2)*exp(-((alpha1*alpha2)*distance(p1,p2)^2 + (alpha1*alpha3)*distance(p1,p3)^2 + (alpha2*alpha3)*distance(p2,p3)^2)/(alpha1 + alpha2 + alpha3) );  p=2.7; pi=3.14159265'
triplets_vol = openmm.CustomCompoundBondForce(3, triplets_energy)
triplets_vol.addPerBondParameter('alpha1')
triplets_vol.addPerBondParameter('alpha2')
triplets_vol.addPerBondParameter('alpha3')
triplets_vol.addGlobalParameter('height', height)
for triplet in triplets:
    triplets_vol.addBond([triplet[0], triplet[1], triplet[2]], [alpha[triplet[0]], alpha[triplet[1]], alpha[triplet[2]]])

quartets_energy= 'height*p^4*(pi/(alpha1+alpha2+alpha3+alpha4))^(3/2)*exp(-((alpha1*alpha2)*distance(p1,p2)^2 + (alpha1*alpha3)*distance(p1,p3)^2 + (alpha2*alpha3)*distance(p2,p3)^2 + alpha1*alpha4*distance(p1,p4)^2 + alpha2*alpha4*distance(p2,p4)^2 + alpha3*alpha4*distance(p3,p4)^2)/(alpha1 + alpha2 + alpha3 + alpha4) );  p=2.7; pi=3.14159265'
quad_vol = openmm.CustomCompoundBondForce(4, quartets_energy)
quad_vol.addPerBondParameter('alpha1')
quad_vol.addPerBondParameter('alpha2')
quad_vol.addPerBondParameter('alpha3')
quad_vol.addPerBondParameter('alpha4')
quad_vol.addGlobalParameter('height', height)
for q in quartets:
    quad_vol.addBond([q[0], q[1], q[2], q[3]], [alpha[q[0]], alpha[q[1]], alpha[q[2]], alpha[q[3]]])


quintent_energy = '-height*p^5*(pi/(alpha1+alpha2+alpha3+alpha4+alpha5))^(3/2)*exp(-((alpha1*alpha2)*distance(p1,p2)^2 + (alpha1*alpha3)*distance(p1,p3)^2 + (alpha2*alpha3)*distance(p2,p3)^2 + alpha1*alpha4*distance(p1,p4)^2 + alpha2*alpha4*distance(p2,p4)^2 + alpha3*alpha4*distance(p3,p4)^2 + alpha1*alpha5*distance(p1,p5)^2 + alpha2*alpha5*distance(p2,p5)^2 + alpha3*alpha5*distance(p3,p5)^2 + alpha4*alpha5*distance(p4,p5)^2 )/(alpha1 + alpha2 + alpha3 + alpha4 + alpha5) );  p=2.7; pi=3.14159265'
quintent_vol = openmm.CustomCompoundBondForce(5, quintent_energy)
quintent_vol.addPerBondParameter('alpha1')
quintent_vol.addPerBondParameter('alpha2')
quintent_vol.addPerBondParameter('alpha3')
quintent_vol.addPerBondParameter('alpha4')
quintent_vol.addPerBondParameter('alpha5')
quintent_vol.addGlobalParameter('height', height)

for q in quintents:
    quintent_vol.addBond([q[0], q[1], q[2], q[3], q[4]], [alpha[q[0]], alpha[q[1]], alpha[q[2]], alpha[q[3]], alpha[q[4]]])

vol0 =0.5301799*unit.nanometer**3
K = 50000*unit.kilojoules_per_mole/unit.nanometer**6
energy = '(K/2)*((p + t + q + qui) - vol0)^2;'
cvforce = openmm.CustomCVForce(energy)
cvforce.addCollectiveVariable('p', pairs_vol)
cvforce.addCollectiveVariable('t', triplets_vol)
cvforce.addCollectiveVariable('q', quad_vol)
cvforce.addCollectiveVariable('qui', quintent_vol)
cvforce.addGlobalParameter('K', K)
cvforce.addGlobalParameter('vol0', vol0)
cvforce.setForceGroup(29)
system.addForce(cvforce)

K_c = 200*unit.kilojoules_per_mole/unit.angstroms**2
force = openmm.CustomCentroidBondForce(2, '(K_c/2)*distance(g1, g2)^2')
force.addGlobalParameter('K_c', K_c)
force.addGroup([int(index) for index in lig1_heavy_atoms])
force.addGroup([int(index) for index in lig2_heavy_atoms])
force.addBond([0,1], [])
system.addForce(force)
