############################################
# Imports
############################################

import numpy as np
EPSILON = 1e-10

############################################
# Class: Observable
############################################

class Observable(object):
	"""Custom observable class.
	Contains different functions to calculate 1-dim observables.
	"""
	def __init__(self):
		self.epsilon = 1e-16

	def momentum(self, x, entry, particle_id = [0]):
		"""Parent function giving the ith
		momentum entry of n particles.

		# Arguments
			input: Array with input data
			entry: the momentum entry which should be returned
			particle_id: Integers, particle IDs of n particles
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Ps = 0
		for particle in particle_id:
			Ps	+= x[:, entry + particle * 4]

		return np.array(Ps)

	def coordinate_0(self, x, particle_id = [0]):
		return x[:,0]

	def coordinate_1(self, x, particle_id = [0]):
		return x[:,1]


	def energy(self, x, particle_id = [0]):
		return self.momentum(x, 0, particle_id)

	def x_momentum(self, x, particle_id = [0]):
		return self.momentum(x, 1, particle_id)

	def x_momentum_over_abs(self, x, particle_id = [0]):
		momentum = self.x_momentum(x, particle_id)
		energy = np.abs(self.x_momentum(x, [0])) + np.abs(self.x_momentum(x, [1])) + np.abs(self.x_momentum(x, [2])) + np.abs(self.x_momentum(x, [3]))
		return momentum/energy

	def y_momentum_over_abs(self, x, particle_id = [0]):
		momentum = self.y_momentum(x, particle_id)
		energy = np.abs(self.y_momentum(x, [0])) + np.abs(self.y_momentum(x, [1])) + np.abs(self.y_momentum(x, [2])) + np.abs(self.y_momentum(x, [3]))
		return momentum/energy

	def y_momentum(self, x, particle_id = [0]):
		return self.momentum(x, 2, particle_id)

	def z_momentum(self, x, particle_id = [0]):
		return self.momentum(x, 3, particle_id)

	def momentum_product(self, x, y, particle_id = [0,1]):
		EE  = 1.
		PPX = 1.
		PPY = 1.
		PPZ = 1.
		for particle in particle_id:
			EE  *= x[:,0 + particle * 4]
			PPX *= x[:,1 + particle * 4]
			PPY *= x[:,2 + particle * 4]
			PPZ *= x[:,3 + particle * 4]

		return np.array(EE -PPX - PPY - PPZ)

	def momentum_sum(self, x, y, particle_id = [0, 1]):
		Es	= 0
		PXs = 0
		PYs = 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PXs += x[:,1 + particle * 4]
			PYs += x[:,2 + particle * 4]
			PZs += x[:,3 + particle * 4]

		mom_sum = np.sqrt(Es**2 - PXs**2 - PYs**2 - PZs**2) 

		return np.array(mom_sum)

	def identity(self, x, particle_id = [0,1]):
		"""Simply gives the output back
		"""
		return np.array(x)

	def invariant_mass(self, x, particle_id = [0,1]):
		"""Invariant Mass.
		This function gives the invariant mass of n particles.

		# Arguments
			input: Array with input data
				that will be used to calculate the invariant mass from.
			particle_id: Integers, particle IDs of n particles.
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Es	= 0
		PXs = 0
		PYs = 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PXs += x[:,1 + particle * 4]
			PYs += x[:,2 + particle * 4]
			PZs += x[:,3 + particle * 4]

		m2 = np.square(Es) - np.square(PXs) - np.square(PYs) - np.square(PZs)
		m = np.sqrt(np.clip(m2, EPSILON, None))
		return np.array(m)

	def reduced_mass(self, x, particle_id = [0,1]):
		"""
		# Arguments
			input: Array with input data
				that will be used to calculate the invariant mass from.
			particle_id: Integers, particle IDs of n particles.
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Es	= 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PZs += x[:,1 + particle * 4]

		m2 = np.square(Es) - np.square(PZs)
		m = np.sqrt(np.clip(m2, EPSILON, None))
		return np.array(m)

	def invariant_mass_square(self, x, particle_id = [0,1]):
		"""Invariant Mass.
		This function gives the invariant mass of n particles.

		# Arguments
			input: Array with input data
				that will be used to calculate the invariant mass from.
			particle_id: Integers, particle IDs of n particles.
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Es	= 0
		PXs = 0
		PYs = 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PXs += x[:,1 + particle * 4]
			PYs += x[:,2 + particle * 4]
			PZs += x[:,3 + particle * 4]

		m2 = np.square(Es) - np.square(PXs) - np.square(PYs) - np.square(PZs)
		return np.array(m2)

	def transverse_momentum(self, x, particle_id = [0]):
		"""This function gives the transverse momentum of n particles.

		# Arguments
			particle_id: Integers, particle IDs of n particles
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		PXs = 0
		PYs = 0
		for particle in particle_id:
			PXs += x[:,1 + particle * 4]
			PYs += x[:,2 + particle * 4]

		PXs2 = np.square(PXs)
		PYs2 = np.square(PYs)

		pTs = PXs2 + PYs2

		m = np.sqrt(pTs)
		return np.array(m)

	def rapidity(self, x, particle_id = [0]):
		"""Rapidity.
		This function gives the rapidity of n particles.

		# Arguments
			particle_id: Integers, particle IDs of n particles
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Es	= 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PZs += x[:,3 + particle * 4]

		y = 0.5 * (np.log(np.clip(np.abs(Es + PZs), self.epsilon, None)) -
				   np.log(np.clip(np.abs(Es - PZs), self.epsilon, None)))

		return np.array(y)

	def phi(self, x, particle_id = [0]):
		"""Azimuthal angle phi.
		This function gives the azimuthal angle oftthe particle.

		# Arguments
			particle_id: Integers, particle IDs of two particles given in
				the shape:
				`particle_id = [particle_1, particle_2]`.
		"""
		PX1s  = 0
		PY1s = 0
		for particle in particle_id:
			PX1s  += x[:,1 + particle * 4]
			PY1s += x[:,2 + particle * 4]

		phi = np.arctan2(PY1s,PX1s)


		return np.array(phi)

	def pseudo_rapidity(self, x, particle_id = [0]):
		"""Psudo Rapidity.
		This function gives the pseudo rapidity of n particles.

		# Arguments
			particle_id: Integers, particle IDs of n particles
				If there is only one momentum that has to be considered
				the shape is:
				`particle_id = [particle_1]`.

				If there are more then one momenta (q = p_1 + p_2 + ..) the shape is:
				`particle_id = [particle_1, particle_2,..]`.
		"""
		Es	= 0
		PXs = 0
		PYs = 0
		PZs = 0
		for particle in particle_id:
			Es	+= x[:,0 + particle * 4]
			PXs += x[:,1 + particle * 4]
			PYs += x[:,2 + particle * 4]
			PZs += x[:,3 + particle * 4]

		Ps = np.sqrt(np.square(PXs) + np.square(PYs) + np.square(PZs))
		eta = 0.5 * (np.log(np.clip(np.abs(Ps + PZs), self.epsilon, None)) -
					 np.log(np.clip(np.abs(Ps - PZs), self.epsilon, None)))

		return np.array(eta)

	def delta_phi(self, x, particle_id = [0,1]):
		"""Delta Phi.
		This function gives the difference in the azimuthal angle of 2 particles.

		# Arguments
			particle_id: Integers, particle IDs of two particles given in
				the shape:
				`particle_id = [particle_1, particle_2]`.
		"""

		PX1s = x[:, 1 + particle_id[0] * 4]
		PY1s = x[:, 2 + particle_id[0] * 4]

		PX2s = x[:, 1 + particle_id[1] * 4]
		PY2s = x[:, 2 + particle_id[1] * 4]

		phi1s = np.arctan2(PY1s,PX1s)
		phi2s = np.arctan2(PY2s,PX2s)

		dphi = np.fabs(phi1s - phi2s)
		dphimin = np.where(dphi>np.pi, 2.0 * np.pi - dphi, dphi)

		return np.array(dphimin)

	def delta_rapidity(self, x, particle_id = [0,1]):
		"""Delta Rapidity.
		This function gives the rapidity difference of 2 particles.

		# Arguments
			particle_id: Integers, particle IDs of two particles given in
				the shape:
				`particle_id = [particle_1, particle_2]`.
		"""
		if not self._check_format:
			x = self._normalize_format(x, particle_id)

		E1s  = x[:, 0 + particle_id[0] * 4]
		PZ1s = x[:, 3 + particle_id[0] * 4]

		E2s  = x[:, 0 + particle_id[1] * 4]
		PZ2s = x[:, 3 + particle_id[1] * 4]

		y1 = 0.5 * (np.log((E1s + PZ1s)) - np.log((E1s - PZ1s)))
		y2 = 0.5 * (np.log((E2s + PZ2s)) - np.log((E2s - PZ2s)))
		dy = np.abs(y1-y2)

		return np.array(dy)
