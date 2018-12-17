#!/usr/bin/env python3

# This script solves shape optimization problems with convexity
# constraints as described in the preprint
# https://arxiv.org/abs/1810.10735.
#
# Requires FEniCS, tested with FEniCS 2018.1
# 
# In particular, all the numerical examples of the paper can be solved by
# uncommenting one of the lines below.
# Beware, this is overwritten by the first command line argument.
# exstring = "example 6.1"
# exstring = "example 6.2"
exstring = "example 6.3"
# exstring = "example 6.3 without convexity"

# Do you want plots?
do_plot = False

# Imports and basic settings {{{
from dolfin import *
from DiscreteShapeProblem import *
import numpy
import sys

import matplotlib.pyplot as plt
from math import floor, ceil

# Do not bother us with many messages
set_log_level(LogLevel.WARNING)

# We do not want to care about DoF maps
parameters["reorder_dofs_serial"] = False
# }}}
# Helper routines {{{
def to_scipy(Q):
	# Convert PETScMatrix to scipy array
	# https://fenicsproject.org/qa/9661/convert-to-a-scipy-friendly-format/
	ai, aj, av = Q.mat().getValuesCSR()
	Q = scipy.sparse.csr_matrix((av, aj, ai))
	return Q
# }}}
# Parameters for the gradient method {{{
# Number of gradient steps
graditer = 1000

# Parameters for line search
sigma = 0.1
beta = 0.5

# Initial line search step size
alpha = 1.

# Parameters for merit function
M = 1e-9
M_factor = 10.
# }}}
# Parse 'exstring' {{{
import sys
if len(sys.argv) > 1:
	exstring = sys.argv[1]

if exstring == "example 6.1":
	ex = 1
	convex = True
elif exstring == "example 6.2":
	ex = 2
	convex = True
elif exstring == "example 6.3":
	ex = 3
	convex = True
elif exstring == "example 6.3 without convexity":
	ex = 3
	convex = False
else:
	raise ValueError('Unknown value of "exstring": "%s"' % exstring)
# }}}
# Parameters for discretization {{{
if ex == 3:
	dim = 3
	refinements = 3
else:
	dim = 2
	refinements = 5

initial_refinement = 3
# }}}
# Some initialization {{{
use_quadprog = False

if use_quadprog:
	import quadprog
else:
	import scipy.sparse
	import osqp

	if dim == 2:
		tolerance = 1e-10

		tolerance_accept = 1e-9
	else:
		tolerance = 1e-9

		tolerance_accept = 2e-7



directory = "solutions/" + exstring + "/"
f1 = File(directory + 'solution.pvd')
f2 = File(directory + 'shape_grad_L2.pvd')
f3 = File(directory + 'shape_grad.pvd')
if dim == 3:
	f4 = File(directory + 'curvature.pvd')
# }}}

# Export initial mesh and all optimal meshes
export_views = [0]
curr_view = -1

# Export number of iterations
iters = []

try:
	for i in range(refinements):
		# Setup the shape problem and some functions {{{
		if i == 0:
			dsp = DiscreteShapeProblem( initial_refinement, dim=dim, ex=ex, convex=convex )
			old_coordinates = dsp.mesh.coordinates().copy()
		else:
			# Refine mesh and dsp {{{
			if dim == 2:
				mesh = refine(dsp.mesh)
			else:
				# In three dimensions, we use a more elaborate refinement to get
				# nicer triangulations
				old_mesh = dsp.mesh
		
				# Reconstruct overall displacement field
				V = Function(VectorFunctionSpace(old_mesh, "CG", 1))
				V.vector()[:] = np.reshape(old_mesh.coordinates() - old_coordinates, [-1], order='F')
		
				# Warp mesh back to initial position (this is needed for the subsequent interpolation)
				old_mesh.coordinates()[:] = old_coordinates
		
				n = initial_refinement * 2**i
		
				mesh = BoxMesh(Point(-.5,-.5,-.5), Point(.5,.5,.5), n, n, n )
				old_coordinates = mesh.coordinates().copy()
		
				VV = Function(VectorFunctionSpace(mesh, "CG", 1))
				VV.interpolate(V)
		
				mesh.coordinates()[:] += np.reshape(VV.vector(), [-1,3], order='F')
			dsp = DiscreteShapeProblem( ex=ex, mesh=mesh, convex=convex )
			# }}}

		V = Function( dsp.U )
		V_step = Function( dsp.U )

		directional_shape_derivative = Form(action(dsp.get_shape_derivative_form(),V))

		n_boundary = dsp.n_boundary
		n_convconst = dsp.n_convconst
		n_total = dsp.n_total

		# Needed for elasticity form:
		dsp.compute_shape_gradient( V )
		# }}}
		for j in range(graditer):
			print("\nStep: %d" % j)

			curr_view += 1

			# Increase alpha a little bit
			alpha /= beta

			obj = dsp.compute_objective()

			print("Obj:             % e" % obj)

			# Output the solution and the shape gradient
			f1 << dsp.u
			dsp.compute_shape_gradient_L2( V )
			f2 << V



			# Compute V
			## Setup the quadratic program{{{
			t = Timer("Block system setup")
			# Compute elasticity matrix{{{
			Q = PETScMatrix()
			assemble(dsp.shape_grad["lhs"], tensor=Q)
			# }}}
			# Compute linear term{{{
			d = assemble(dsp.get_shape_derivative_form()).get_local()
			# }}}
			# Fill constraint matrix and constraint vector {{{
			b = dsp.get_convexity()
			if use_quadprog:
				# Set up constraint matrix as dense matrix
				A = numpy.zeros([n_convconst, numpy.uint64(dsp.dim)*n_total])
			else:
				# Set up constraint matrix as sparse matrix
				A = scipy.sparse.lil_matrix((n_convconst, numpy.uint64(dsp.dim)*n_total))
			if dsp.dim == 2:
				for i in range(n_boundary):
					j1 = dsp.boundary_nodes[i]
					j2 = dsp.boundary_nodes[i+1]
					j3 = dsp.boundary_nodes[i+2]

					x1 = dsp.mesh.coordinates()[ j1 ]
					x2 = dsp.mesh.coordinates()[ j2 ]
					x3 = dsp.mesh.coordinates()[ j3 ]

					A[i, j1 + 0*n_total] = alpha*(  x3[1] - x2[1])
					A[i, j1 + 1*n_total] = alpha*(-(x3[0] - x2[0]))
					A[i, j2 + 0*n_total] = alpha*(  x1[1] - x3[1])
					A[i, j2 + 1*n_total] = alpha*(-(x1[0] - x3[0]))
					A[i, j3 + 0*n_total] = alpha*(  x2[1] - x1[1])
					A[i, j3 + 1*n_total] = alpha*(-(x2[0] - x1[0]))
			else:
				for i in range(n_convconst):
					j0 = dsp.boundary_edges[i][0]
					j1 = dsp.boundary_edges[i][1]
					j2 = dsp.boundary_edges[i][2]
					j3 = dsp.boundary_edges[i][3]

					x0 = dsp.mesh.coordinates()[ j0 ]
					x1 = dsp.mesh.coordinates()[ j1 ]
					x2 = dsp.mesh.coordinates()[ j2 ]
					x3 = dsp.mesh.coordinates()[ j3 ]

					A[i, j0 + 0*n_total] =  alpha*np.linalg.det([[x2[1]-x1[1],x2[2]-x1[2]],[x3[1]-x1[1],x3[2]-x1[2]]])
					A[i, j0 + 1*n_total] = -alpha*np.linalg.det([[x2[0]-x1[0],x2[2]-x1[2]],[x3[0]-x1[0],x3[2]-x1[2]]])
					A[i, j0 + 2*n_total] =  alpha*np.linalg.det([[x2[0]-x1[0],x2[1]-x1[1]],[x3[0]-x1[0],x3[1]-x1[1]]])

					A[i, j1 + 0*n_total] = -alpha*np.linalg.det([[x2[1]-x0[1],x2[2]-x0[2]],[x3[1]-x0[1],x3[2]-x0[2]]])
					A[i, j1 + 1*n_total] =  alpha*np.linalg.det([[x2[0]-x0[0],x2[2]-x0[2]],[x3[0]-x0[0],x3[2]-x0[2]]])
					A[i, j1 + 2*n_total] = -alpha*np.linalg.det([[x2[0]-x0[0],x2[1]-x0[1]],[x3[0]-x0[0],x3[1]-x0[1]]])

					A[i, j2 + 0*n_total] =  alpha*np.linalg.det([[x1[1]-x0[1],x1[2]-x0[2]],[x3[1]-x0[1],x3[2]-x0[2]]])
					A[i, j2 + 1*n_total] = -alpha*np.linalg.det([[x1[0]-x0[0],x1[2]-x0[2]],[x3[0]-x0[0],x3[2]-x0[2]]])
					A[i, j2 + 2*n_total] =  alpha*np.linalg.det([[x1[0]-x0[0],x1[1]-x0[1]],[x3[0]-x0[0],x3[1]-x0[1]]])

					A[i, j3 + 0*n_total] = -alpha*np.linalg.det([[x1[1]-x0[1],x1[2]-x0[2]],[x2[1]-x0[1],x2[2]-x0[2]]])
					A[i, j3 + 1*n_total] =  alpha*np.linalg.det([[x1[0]-x0[0],x1[2]-x0[2]],[x2[0]-x0[0],x2[2]-x0[2]]])
					A[i, j3 + 2*n_total] = -alpha*np.linalg.det([[x1[0]-x0[0],x1[1]-x0[1]],[x2[0]-x0[0],x2[1]-x0[1]]])

			print("ConvViol:        % e" % (b[b>0].sum()))
			# }}}
			# Output curvature {{{
			if dsp.dim == 3:
				mf = MeshFunction("double", dsp.mesh, 1)

				# Set an absurd value on the inner edges.
				mf.set_all(-5.0)

				dsp.mesh.init(0,1)

				for i in range(n_convconst):
					i0 = dsp.boundary_edges[i][0]
					i1 = dsp.boundary_edges[i][1]

					v0 = Vertex(dsp.mesh, i0)
					v1 = Vertex(dsp.mesh, i1)

					e_index = (set(v0.entities(1)) & set(v1.entities(1))).pop()
					mf[Edge(dsp.mesh,e_index)] = -b[i]

				f4 << mf
			# }}}
			## Setup normal-vector matrix
			# Now, we are going to compute the projected shape gradient

			N = PETScMatrix()
			assemble( inner(TestFunction(dsp.U), FacetNormal(dsp.mesh)) * TrialFunction(dsp.V) * ds, tensor=N)
			# }}}
			# Solve the quadratic program{{{
			if use_quadprog:
				# Convert to numpy array
				Q = Q.array()
				N = N.array()

				# Remove the non-boundary columns
				N = N[:,dsp.boundary_dofmap]

				# Solve
				sol = quadprog.solve_qp(Q, -d, A.T, b)

				# Save solution
				V.vector()[:] = sol[0]
			else:
				# We use OSQP

				# Convert to scipy array
				Q = to_scipy(Q)
				N = to_scipy(N)

				# Remove the non-boundary columns
				N.eliminate_zeros()
				N = N[:,dsp.boundary_dofmap]

				# Setup the block matrices
				QQ = scipy.sparse.bmat( [[Q, None], [None, scipy.sparse.csc_matrix((n_boundary, n_boundary))]], format='csc')
				AA = scipy.sparse.bmat( [[Q, -N], [A, None]], format='csc')
				dd = numpy.hstack([d, numpy.zeros(n_boundary)])
				ll = numpy.hstack([numpy.zeros(dsp.U.dim()), b])
				uu = numpy.hstack([numpy.zeros(dsp.U.dim()), numpy.full(b.shape,fill_value=numpy.inf)])

				solver = osqp.OSQP()
				solver.setup(QQ, dd, AA, l = ll, u = uu, verbose=False)

				eps_abs = 1e-3
				eps_rel = 1e-3

				solver.update_settings( eps_abs = eps_abs, eps_rel = eps_rel )
				solver.update_settings( polish = 1 )

				t = Timer("Solve QP")
				sol = solver.solve()

				while sol.info.pri_res > tolerance or sol.info.dua_res > tolerance:
					# print("Need to solve again (%d, %e, %e)" % (sol.info.status_polish, sol.info.pri_res, sol.info.dua_res))
					eps_abs /= 10
					eps_rel /= 10
					if eps_abs < 1e-10:
						if dsp.dim == 2:
							if sol.info.pri_res < tolerance_accept and sol.info.dua_res < tolerance_accept:
								break
							else:
								raise Exception('I am not able to polish!')
						else:
							# In dimension 3, accept everything
							print('!!!!!I am not able to polish!!!!!!')
							break
					solver.update_settings( eps_abs = eps_abs, eps_rel = eps_rel )
					solver.warm_start(x = sol.x, y = sol.y)
					sol = solver.solve()

				t.stop()

				# Save solution
				V.vector()[:] = sol.x[0:dsp.U.dim()]

				elapsed = t.elapsed()[0]
				print("Solved QP in %fs" % elapsed)
			# }}}

			# dsp.compute_shape_gradient( V )
			f3 << V

			d_obj = assemble(directional_shape_derivative)

			print("Dir. Derivative: % e" % d_obj)
			if sqrt(abs(d_obj)) < 1e-6:
				export_views.append(curr_view)
				iters.append(j+1)
				break

			# Perform a line search
			merit = obj + M * b[b>0].sum()
			d_merit = d_obj + M * (-A*sol.x[0:dsp.U.dim()]/alpha)[b>0].sum()

			while d_merit >= 0:
				M *= M_factor

				if M > 1e10:
					# print(ll - AA*sol.x)
					print(d_obj)
					print((-A*sol.x[0:dsp.U.dim()]/alpha)[b>0].sum())

					export_views.append(curr_view)
					iters.append(j+1)
					raise Exception("No M")
					break
				print("======================\nNew M: %e" % M)
				merit = obj + M * b[b>0].sum()
				d_merit = d_obj + M * (-A*sol.x[0:dsp.U.dim()]/alpha)[b>0].sum()

			while True:
				if dsp.is_step_valid( alpha, V):
					# Move the mesh
					V_step.vector()[:] = alpha * V.vector()[:]
					dsp.move_mesh( V_step )

					# Compute the objective
					obj_step = dsp.compute_objective()
					b_step = dsp.get_convexity()
					merit_step = obj_step + M * b_step[b_step>0].sum()

					# For checking the directional derivative with finite differences
					# print("%e" % ((obj_step - obj)/alpha))
					# print("%e vs %e" % (d_merit, (merit_step - merit)/alpha))

					# Line search successful?
					if merit_step < merit + sigma * alpha * d_merit:
						break

					# If not, restore the mesh and decrease alpha
					dsp.restore_mesh()
				alpha *= beta

				# If alpha is too small, something went wrong.
				if alpha < 1e-10:

					export_views.append(curr_view)
					iters.append(j+1)
					raise Exception("Line search failed")

			print('alpha:           % e' % alpha )

			if do_plot:
				plt.clf()
				plot( dsp.mesh )

				minx = floor(dsp.mesh.coordinates()[:,0].min())
				miny = floor(dsp.mesh.coordinates()[:,1].min())
				maxx = ceil(dsp.mesh.coordinates()[:,0].max())
				maxy = ceil(dsp.mesh.coordinates()[:,1].max())

				axes = plt.gca()
				axes.set_xlim([minx, maxx])
				axes.set_ylim([miny, maxy])

				plt.show(block=False)
				plt.pause(0.05)
				# input()

			sys.stdout.flush()

finally:

	# Print some timings
	t = timings(TimingClear.keep, \
			[ \
			TimingType.wall \
			])

	print("\n" + t.str(True))


	print(export_views)
	f = open(directory + 'views.py', 'w')
	f.write('%r\n' % export_views)
	f.close()

	f = open('iters ' + exstring + '.py', 'w')
	f.write('iters = %r\n' % iters)
	f.close()

# vim: fdm=marker noet
