import numpy as np
from enum import Enum
import math
import matplotlib.pyplot as plt

class KType(Enum):
		PRISMATIC = 1
		REVOLUTE = 2

class KinematicLinkage(object):

	def __init__(self):
		self.joints = []
		self.transforms = []
		self.operations = []


	# Add a transform in the chain
	def transform(self, name, ktype, j_min, j_max, alpha, a, d, theta):

		current_joint = np.array([name, ktype, j_min, j_max, alpha, a, d, theta])
		current_transform = np.array([self.transform_matrix(alpha, a, d, theta)])

		if len(self.joints) == 0:
			self.joints = current_joint
			self.transforms = current_transform
			self.operations = current_transform.copy()
		else:
			self.joints = np.vstack((self.joints, current_joint))
			self.transforms = np.append(self.transforms, current_transform, axis=0)
			self.operations = np.append(self.operations, [np.matmul(self.operations[-1], self.transforms[-1])], axis=0)

		return self


	# Edit d_i or theta_i on an existing transform
	def edit(self, index, delta):
		i = self.get_index_from_name(index)

		# Check joint type and constraints
		if self.joints[i,1] == KType.PRISMATIC:
			self.joints[i,6] += delta
			
			if self.joints[i,6] + delta <= self.joints[i,2]:
				self.joints[i,6] = self.joints[i,2]
			elif self.joints[i,6] + delta > self.joints[i,3]:
				self.joints[i,6] = self.joints[i,3]

		elif self.joints[i,1] == KType.REVOLUTE:
			self.joints[i,7] += delta

			# Do not limit full revolutions
			if not (self.joints[i,2] == -180 and self.joints[i,3] == 180):
				if self.joints[i,7] + delta <= self.joints[i,2]:
					self.joints[i,7] = self.joints[i,2]
				elif self.joints[i,7] + delta > self.joints[i,3]:
					self.joints[i,7] = self.joints[i,3]

			# Simplify theta
			while self.joints[i,7] < -180:
				self.joints[i,7] += 360
			while self.joints[i,7] >= 180:
				self.joints[i,7] -= 360
		
		joint = self.joints[i]

		self.transforms[i] = self.transform_matrix(joint[4], joint[5], joint[6], joint[7])		
		
		for j in range(i, len(self.joints)):
			self.operations[j] = np.matmul(self.operations[j-1], self.transforms[j])

		return self


	# Position of end effector from operations matrix
	def end_position(self):
		e = self.operations[-1]
		x = e[0,3]
		y = e[1,3]
		z = e[2,3]

		# Convert to degrees and keep within [0, 359]
		roll = (math.atan2(e[1,0], e[0,0]) * 180. / np.pi) % 360
		pitch = (math.atan2(-e[2,0], math.sqrt(e[2,1]**2 + e[2,2]**2)) * 180. / np.pi) % 360
		yaw = (math.atan2(e[2,1], e[2,2]) * 180 / np.pi) % 360

		return np.round(np.array([x, y, z, roll, pitch, yaw]), decimals=5)


	def jacobian(self, x='na', y='na', z='na', roll='na', pitch='na', yaw='na'):
		exclusion = np.array([x, y, z, roll, pitch, yaw])
		J = np.array([[0 for v in exclusion if not v == 'na']])

		for i in range(1, len(self.joints)):

			# Determine joint type
			if self.joints[i,1] == KType.PRISMATIC:
				delta = 0.1
			elif self.joints[i,1] == KType.REVOLUTE:
				delta = 1 # degrees

			# Store original end effector position
			orig_pos = np.array([v for k, v in enumerate(self.end_position()) if not exclusion[k] == 'na'])

			# Add a small delta to variable for this joint
			self.edit(i, delta)

			# Calculate change in position
			new_pos = np.array([v for k, v in enumerate(self.end_position()) if not exclusion[k] == 'na'])

			# Append row to Jacobian
			J = np.append(J, [np.divide(new_pos - orig_pos, delta)], axis=0)

			# Revert change
			self.edit(i, -delta)

		np.delete(J, (0), axis=0)
		return J.T


	def set_effector_position(self, x='na', y='na', z='na', roll='na', pitch='na', yaw='na'):
		exclusion = np.array([x, y, z, roll, pitch, yaw])
		target = np.array([float(v) for v in exclusion if not v == 'na'])
		pos = np.array([v for k, v in enumerate(self.end_position()) if not exclusion[k] == 'na'])
		delta = target - pos
		learning_rate = 0.1
		path = np.array([pos])

		while np.linalg.norm(delta) > 0.1:

			J = self.jacobian(x, y, z, roll, pitch, yaw)
			
			# Pseudo-inverse of J
			J = np.linalg.pinv(J)

			# Change in d and theta
			change = learning_rate * np.matmul(J, delta)

			# Update
			for i in range(1, len(self.joints)):
				self.edit(i, change[i])

			pos = np.array([v for k, v in enumerate(self.end_position()) if not exclusion[k] == 'na'])
			path = np.append(path, [pos], axis=0)
			delta = target - pos

		return path


	def transform_matrix(self, alpha, a, d, theta):
		alpha = alpha * np.pi / 180.
		theta = theta * np.pi / 180.
		return np.array([
			[np.cos(theta),                -np.sin(theta)             ,  0            ,  a              ],
			[np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
			[np.sin(theta)*np.sin(alpha),  np.cos(theta)*np.sin(alpha),  np.cos(alpha),  d*np.cos(alpha)],
			[0                          ,  0                          ,  0            ,  1              ]])


	def get_index_from_name(self, name):
		if isinstance(name, str):
			for ind, val in enumerate(self.joints):
				if val == name:
					return ind

			# name is string but not found
			return -1

		# name is not a string, probably index
		return name


	def get_joints(self):
		return self.joints


	def get_transforms(self):
		return np.round(self.transforms, decimals=3)


	def get_operations(self):
		return np.round(self.operations, decimals=3)


# Set up linkage
Linkage = KinematicLinkage()
Linkage.transform('base', KType.REVOLUTE, -180, 180, 0, 0, 0, 0)
Linkage.transform('height adjustment', KType.PRISMATIC, 7, 13, 0, 0, 10, 0)
Linkage.transform('main arm', KType.REVOLUTE, -180, 180, 0, 0, 0, 0)
Linkage.transform('secondary arm', KType.REVOLUTE, -90, 90, 0, 3, 1, -30)
Linkage.transform('reflector', KType.REVOLUTE, -180, 180, 90, 6, 0, 0)
Linkage.transform('end effector', KType.REVOLUTE, -180, 180, 0, 3, 0, 0)

# Swing secondary arm to front -- naive path
initial_pos = Linkage.end_position()
print("Initial position: ", initial_pos)
print("Initial operation space:\n ", Linkage.get_operations()[-1])
path = Linkage.set_effector_position(3, 9, 11)
final_pos = Linkage.end_position()
print("Final position: ", final_pos)
print("Final operation space:\n ", Linkage.get_operations()[-1])

# Plot path [x, y]
x_straight = np.linspace(path[0,0], path[-1,0], 10)
y_straight = np.linspace(path[0,1], path[-1,1], 10)
plt.plot(path[:,0], path[:,1])
plt.plot(x_straight, y_straight, linestyle='--', linewidth=1)
plt.plot()
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Movement Path')
plt.show()