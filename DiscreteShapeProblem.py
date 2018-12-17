# -*- coding: utf8 -*-
from dolfin import *
from ufl import replace
import numpy as np

class DiscreteShapeProblem:
	def __init__( self, n=4, dim=2, ex=1, mesh=None, convex=True ):

		self.ex = ex
		self.convex = convex
		if mesh:
			self.mesh = mesh

			self.dim = mesh.geometry().dim()
		else:
			# Save the dimension
			self.dim = dim

			# Create the mesh
			if self.dim == 2:
				self.mesh = UnitDiscMesh.create( MPI.comm_world, n, 1, self.dim)
				# self.mesh = UnitSquareMesh( n, n)
			else:
				# filename = 'sphere_%d.xml.gz' % (n)
				# try:
				# 	self.mesh = Mesh(filename)
				# except RuntimeError:
				# 	from mshr import Sphere, generate_mesh
				# 	radius = 1.0
				# 	sphere = Sphere(Point(0.0, 0.0, 0.0), radius)
				#
				# 	self.mesh = generate_mesh(sphere, n)
				#
				# 	File(filename) << self.mesh

				self.mesh = BoxMesh(Point(-.5,-.5,-.5), Point(.5,.5,.5), n, n, n )


		if self.dim == 2:
			self.cell = triangle
		else:
			self.cell = tetrahedron

		if self.ex == 1 or self.ex == 2:
			self.dirichlet_bc = True
		else:
			self.dirichlet_bc = False

		# Create the function spaces
		self.V = FunctionSpace(self.mesh, "CG", 1)
		self.U = VectorFunctionSpace(self.mesh, "CG", 1)
		self.DG0 = FunctionSpace(self.mesh, "DG", 0)

		# Create boundary condition for PDE
		if self.dirichlet_bc:
			# We use Dirichlet BC
			def boundary_D(x, on_boundary):
				return on_boundary
			zero = Constant(0.0)
			self.bc = DirichletBC(self.V, zero, boundary_D)
		else:
			# We use Neumann boundary conditions
			self.bc = []

		# Create the functions. These are used in all the forms.
		# Therefore, compute_state() should be called before compute_obj().
		self.u = Function( self.V )
		self.u.rename("u", "u")
		self.p = Function( self.V )
		self.p.rename("p", "p")

		# For storing the old solution after moving the mesh
		self.old_u = Function( self.V )
		self.old_p = Function( self.V )

		# Create a boolean to remember whether [u,p] is valid on the current mesh
		self.has_u = False
		self.has_p = False


		# Compute boundary mesh
		bmesh = BoundaryMesh( self.mesh, 'exterior' )
		bmesh.init(0,1)

		# Make it visible
		self.bmesh = bmesh

		# Compute some dimensions
		self.n_boundary = bmesh.num_vertices()
		self.n_total = np.uint64(self.mesh.num_vertices())

		# And the boundary dofmap
		boundary_dofmap = [bmesh.entity_map(0)[i] for i in range(bmesh.num_vertices())]

		# Make it visible
		self.boundary_dofmap = boundary_dofmap

		if self.convex:
			# Setup structure for convexity constraint {{{
			if self.dim == 2:
				# Two dimensions {{{

				# Set up a list of ordered nodes
				first_node = 0
				last_edge = None
				boundary_nodes = [first_node,]

				node = first_node

				while True:
					next_edges = Vertex(bmesh, node).entities(1)

					next_edge = next_edges[np.argwhere(next_edges != last_edge)[0][0]]

					next_nodes = bmesh.cells()[next_edge]
					next_node = next_nodes[np.argwhere( next_nodes != node )[0][0]]

					if next_node == first_node:
						break

					boundary_nodes.append(next_node)
					node = next_node
					last_edge = next_edge

				# Convert to global indices
				boundary_nodes = [boundary_dofmap[i] for i in boundary_nodes]

				# Add the first two nodes to the end
				boundary_nodes = np.hstack([boundary_nodes, boundary_nodes[0:2]])

				# Make it visible
				self.boundary_nodes = boundary_nodes

				self.n_convconst = self.n_boundary

				# Check orientation of boundary
				s = self.get_convexity()
				if s.max() > 0:
					self.boundary_nodes = np.flipud(self.boundary_nodes)

				# }}}
			else:
				# Three dimensions {{{
				# Initialize connectivities
				self.mesh.init(2, 1)
				self.mesh.init(1, 2)
				self.mesh.init(2, 3)

				# Initialize a list of boundary edges
				be = []
				for i in range(self.bmesh.num_cells()):
					global_index = self.bmesh.entity_map(2)[i]
					f = Facet(self.mesh, global_index)

					be += list(f.entities(1))

				# Extract a list of unique edges
				be = sorted(set(be))

				# Initialize structure for convexity
				self.boundary_edges = []

				# Loop over all boundary edges
				for i in be:
					e = Edge(self.mesh, i)

					# Get the vertices
					vi0, vi1 = e.entities(0)

					v0 = Vertex(self.mesh, vi0)
					v1 = Vertex(self.mesh, vi1)

					# Get the facets
					facets = []
					for j in e.entities(2):
						f = Facet(self.mesh, j)
						if f.num_entities(3) == 1:
							facets += [f,]

					assert( len(facets) == 2)
					f1, f2 = facets

					# Get the other vertices
					s = set(f1.entities(0)) - set([vi0, vi1])
					assert(len(s) == 1)
					v2 = Vertex(self.mesh, s.pop())

					s = set(f2.entities(0)) - set([vi0, vi1])
					assert(len(s) == 1)
					v3 = Vertex(self.mesh, s.pop())

					# Now, check which of v2 and v3 is left of (v0,v1)
					c = Cell(self.mesh, f1.entities(3)[0])
					s = set(c.entities(0)) - set([vi0, vi1, v2.index()])
					assert(len(s) == 1)
					v_inner = Vertex(self.mesh, s.pop())

					# Get the coordinates
					x0 = v0.point().array()
					x1 = v1.point().array()
					x2 = v2.point().array()
					x_inner = v_inner.point().array()

					vol = np.linalg.det(np.vstack([x1 - x0, x2 - x0, x_inner - x0]))

					if vol > 0:
						vr = v2
						vl = v3
					else:
						vr = v3
						vl = v2

					self.boundary_edges.append([vi0,vi1, vl.index(), vr.index()])

				print('# boundary edges: %d' % len(self.boundary_edges))

				self.n_convconst = len(self.boundary_edges)



				# }}}
			# }}}
		else:
			# Disable the convexity constraints
			self.boundary_edges = []
			self.n_convconst = len(self.boundary_edges)

	def f( self, x, y, z = None ):
		if self.dim == 2:
			if self.ex == 1:
				# This is the function in the shape optimization problem
				return Constant(20.)*(x + .4 - y**2)**2 + x**2 + y**2 - 1

			elif self.ex == 2:
				# Encouraging a triangle
				expr = Constant(-0.5) + Constant(0.8)*(x**2+y**2)
				s = Constant(8.)
				
				n = 5
				for i in range(n):
					r = 1.0
					xi = Constant(r * sin( (i+.5) * 2 * pi / n ));
					yi = Constant(r * cos( (i+.5) * 2 * pi / n ));
					expr += Constant(2.)*exp(-((x - xi)**2 + (y - yi)**2)*s)

				for i in range(n):
					r = 1.2
					xi = Constant(r * sin( i * 2 * pi / n ));
					yi = Constant(r * cos( i * 2 * pi / n ));
					expr -= exp(-((x - xi)**2 + (y - yi)**2)*s)

				return expr

		else:
			# This is the function in the shape optimization problem
			return (x**2 + y**2 + z**2) - Constant(1.)
			# return Constant(20.)*(x + .4 - y**2)**2 + x**2 + y**2 + z**2 - 1

			# lmbda = 8.
			# return (Constant((lmbda+1)/2.)*(x**2+y**2) + Constant(lmbda)*x*y + Constant(lmbda)*z**2) - Constant(1.)
			# return (x+y)**2*Constant(lmbda/2.) + ((x-y)/sqrt(2))**8 + Constant(lmbda)*z**2 - 1.0

			# lmbda = 8
			# return ((x+y)/sqrt(2))**lmbda + ((x-y)/sqrt(2))**lmbda + z**lmbda - Constant(.5)

		# return Constant(20.0)*((x**2-1)**2 + .4 - 2*y**2)**2 + Constant(50)*x**2 + y**2 - Constant(20);



	def lagrangian( self, V, u, p ):
		# Deformation gradient, this is the Jacobian of the transformation
		F = Identity(self.dim) + grad(V)
		# Some helper routines {{{
		class WeightedMeasureWrapper:
			# This class is needed, because FEniCS does not like
			#   f*(y*dx)
			def __init__(self, factor, measure):
				self.factor = factor
				self.measure = measure
			def __rmul__(self, other):
				return (other*self.factor)*self.measure
		n = FacetNormal(self.mesh)
		dxV = WeightedMeasureWrapper(det(F), dx(self.mesh))
		dsV = WeightedMeasureWrapper(det(F)*sqrt(inner(inv(F).T*n,inv(F).T*n)), ds(self.mesh))
		def gradV(f):
			return inv(F.T)*grad(f)
		# }}}
		x = SpatialCoordinate(self.mesh)

		# This is the objective
		obj = u * dxV


		if self.dirichlet_bc:
			# This is the PDE
			pde = inner(gradV(u),gradV(p)) * dxV - self.f(*[x[i] + V[i] for i in range(self.dim)])*p*dxV
		else:
			pde = (inner(gradV(u),gradV(p))+u*p) * dxV - self.f(*[x[i] + V[i] for i in range(self.dim)])*p*dxV

		# Built the Lagrangian
		L = obj + pde

		return L

	def compute_objective( self ):
		self.compute_state()
		# Prepare form {{{
		if not hasattr(self, 'obj'):
			Vref = Constant([0.0] * self.dim, self.cell)
			p = Constant(0.0, self.cell)
			self.obj = Form(self.lagrangian( Vref, self.u, p ) )
		# }}}
		return assemble( self.obj )

	def compute_state( self ):
		# Prepare PDE solver {{{
		if not hasattr(self, 'pde'):
			# Prepare a solver for the PDE
			u = Function( self.V )
			p = Function( self.V )
			Vref = Constant([0.0] * self.dim, self.cell)

			L = self.lagrangian( Vref, u, p )
			pde_form = derivative(L, p, TestFunction(self.V))
			pde_form = replace(pde_form, {u: TrialFunction(self.V)})

			problem = LinearVariationalProblem( lhs(pde_form), rhs(pde_form), self.u, self.bc )
			solver = LinearVariationalSolver( problem )

			self.pde = dict()
			self.pde["solver"] = solver
		# }}}
		if not self.has_u:
			# Solve the PDE
			self.pde["solver"].solve()
			self.has_u = True

	def compute_adjoint( self ):
		self.compute_state()
		# Prepare adjoint solver {{{
		if not hasattr(self, 'adjoint'):
			# Prepare a solver for the adjoint PDE
			u = Function( self.V )
			p = Function( self.V )
			Vref = Constant([0.0] * self.dim, self.cell)

			L = self.lagrangian( Vref, u, p )
			adjoint_form = derivative(L, u, TestFunction(self.V))
			adjoint_form = replace(adjoint_form, {p: TrialFunction(self.V)})
			adjoint_form = replace(adjoint_form, {u: self.u})

			problem = LinearVariationalProblem( lhs(adjoint_form), rhs(adjoint_form), self.p, self.bc )
			solver = LinearVariationalSolver( problem )

			self.adjoint = dict()
			self.adjoint["solver"] = solver
		# }}}
		if not self.has_p:
			# Solve the adjoint PDE
			self.adjoint["solver"].solve()
			self.has_p = True

	def get_shape_derivative_form( self ):
		# This method returns a form for computing the shape derivative.
		# Before using the form, one has to call compute_adjoint()!
		# Prepare the form {{{
		if not hasattr(self, 'shape'):
			# Prepare the bilinear form and the linear form of the adjoint PDE
			Vref = Function( self.U )

			L = self.lagrangian( Vref, self.u, self.p )
			V = TestFunction(self.U)
			shape_form = derivative(L, Vref, V)

			Vzero = Constant([0.0] * self.dim, self.cell)
			shape_form = replace(shape_form, {Vref: Vzero})

			self.shape = dict()
			self.shape["form"] = shape_form
		# }}}
		return self.shape["form"]

	def compute_shape_gradient( self, V ):
		# This routine computes the (neg) shape gradient w.r.t. the elasticity inner product in \Omega.
		self.compute_state()
		self.compute_adjoint()
		# Prepare the shape gradient solver {{{
		if not hasattr(self, 'shape_grad'):
			def eps(u):
				return sym(grad(u))

			V_ = Function(self.U)

			E = 1e-0
			nu = 0.4
			lame1 = nu * E /((1+nu)*(1-2*nu))
			lame2 = E / (2*(1+nu))
			damp = 1e+1

			W = TestFunction( self.U )
			W2 = TrialFunction( self.U )
			lhs_elasti = \
					2*Constant(lame2) *inner(eps(W2), eps(W)) * dx \
					+ Constant(lame1) *div(W2)*div(W)*dx \
					+ Constant( damp )*inner(W,W2)*dx

			shape_gradient_problem = LinearVariationalProblem( lhs_elasti, -self.get_shape_derivative_form(), V_ )
			shape_gradient_solver = LinearVariationalSolver( shape_gradient_problem )

			self.shape_grad = dict()
			self.shape_grad["solver"] = shape_gradient_solver
			self.shape_grad["V"] = V_
			self.shape_grad["lhs"] = lhs_elasti
		# }}}
		self.shape_grad["solver"].solve()
		V.assign(self.shape_grad["V"])

	def compute_shape_gradient_L2( self, V ):
		# This routine computes the (neg) shape gradient w.r.t. the L^2 inner product in \Omega.
		# It is merely useful for visualization of the shape derivative.

		self.compute_state()
		self.compute_adjoint()
		# Prepare the shape gradient solver {{{
		if not hasattr(self, 'shape_grad_L2'):
			V_ = Function(self.U)

			W = TestFunction( self.U )
			W2 = TrialFunction( self.U )
			lhs_L2 = \
					inner(W,W2)*dx

			shape_gradient_L2_problem = LinearVariationalProblem( lhs_L2, -self.get_shape_derivative_form(), V_ )
			shape_gradient_L2_solver = LinearVariationalSolver( shape_gradient_L2_problem )

			self.shape_grad_L2 = dict()
			self.shape_grad_L2["solver"] = shape_gradient_L2_solver
			self.shape_grad_L2["V"] = V_
		# }}}
		self.shape_grad_L2["solver"].solve()
		V.assign(self.shape_grad_L2["V"])

	def get_convexity( self ):
		s = np.zeros( self.n_convconst )

		if self.dim == 2:
			for i in range(self.n_boundary):
				j1 = self.boundary_nodes[i]
				j2 = self.boundary_nodes[i+1]
				j3 = self.boundary_nodes[i+2]

				x1 = self.mesh.coordinates()[ j1 ]
				x2 = self.mesh.coordinates()[ j2 ]
				x3 = self.mesh.coordinates()[ j3 ]
				s[i] = -np.linalg.det(np.vstack([x1 - x2, x3 - x2]))
		else:
			for i in range(self.n_convconst):
				j0 = self.boundary_edges[i][0]
				j1 = self.boundary_edges[i][1]
				j2 = self.boundary_edges[i][2]
				j3 = self.boundary_edges[i][3]

				x0 = self.mesh.coordinates()[ j0 ]
				x1 = self.mesh.coordinates()[ j1 ]
				x2 = self.mesh.coordinates()[ j2 ]
				x3 = self.mesh.coordinates()[ j3 ]

				s[i] = np.linalg.det(np.vstack([x1 - x0, x2 - x0, x3 - x0]))

		return s

	def move_mesh( self, displacement ):
		self.old_coordinates = self.mesh.coordinates().copy()
		self.old_u.assign(self.u)
		self.old_has_u = self.has_u
		self.old_p.assign(self.p)
		self.old_has_p = self.has_p
		ALE.move(self.mesh, displacement )
		self.has_u = False
		self.has_p = False

	def restore_mesh( self ):
		self.mesh.coordinates()[:] = self.old_coordinates
		self.u.assign(self.old_u)
		self.has_u = self.old_has_u
		self.p.assign(self.old_p)
		self.has_p = self.old_has_p

	def is_step_valid( self, alpha, V ):
		# Check whether the step alpha*V is geometrically valid,
		# i.e., check whether
		#    .5 < det( I + alpha V' ) < 2
		#   alpha * || V' || < .3
		# in all cells.
		# Prepare solver {{{
		if not hasattr(self, 'validity_solver'):
			alpha_constant = Constant(alpha)
			V_ref = Function(self.U)
			a_det = TestFunction(self.DG0)*TrialFunction(self.DG0)*dx
			L_det = det(Identity(self.dim) + alpha_constant*grad(V_ref))*TestFunction(self.DG0)*dx

			solution = Function(self.DG0)

			problem = LinearVariationalProblem( a_det, L_det, solution )
			solver = LinearVariationalSolver( problem )

			self.validity_solver = dict()
			self.validity_solver["solver_det"] = solver
			self.validity_solver["solution_det"] = solution
			self.validity_solver["alpha"] = alpha_constant
			self.validity_solver["V"] = V_ref

			# Now check the norm of the deformation gradient

			a_defgrad = TrialFunction(self.DG0)*TestFunction(self.DG0)*dx
			L_defgrad = sqrt(inner(grad(V_ref),grad(V_ref)))*TestFunction(self.DG0)*dx

			normf = Function(self.DG0)

			problem = LinearVariationalProblem( a_defgrad, L_defgrad, normf )
			solver = LinearVariationalSolver( problem )

			self.validity_solver["solver_defgrad"] = solver
			self.validity_solver["solution_defgrad"] = normf

		# }}}
		# Assign the functions
		self.validity_solver["alpha"].assign(alpha)
		self.validity_solver["V"].assign(V)

		# Compute all the determinants
		self.validity_solver["solver_det"].solve()
		# And the smallest and biggest one
		min_det = self.validity_solver["solution_det"].vector().get_local().min()
		max_det = self.validity_solver["solution_det"].vector().get_local().max()

		# Compute all the norms of the deformation gradients
		self.validity_solver["solver_defgrad"].solve()
		# And the largest one, scaled with alpha (this is not done above)
		max_def_grad = alpha * (self.validity_solver["solution_defgrad"].vector().get_local().max())
		print('Max def grad: %e' % max_def_grad)

		return min_det > .5 and max_det < 2. and max_def_grad < .3

# vim: fdm=marker noet
