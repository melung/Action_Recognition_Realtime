import scipy.io as io
import numpy as np

############ Skeleton Enum ############
ODMI_JOINT_POSITION_PELVIS = 0
ODMI_JOINT_POSITION_SPINE_NAVAL = 1
ODMI_JOINT_POSITION_SPINE_CHEST = 2
ODMI_JOINT_POSITION_NECK = 3
ODMI_JOINT_POSITION_CLAVICLE_LEFT = 4
ODMI_JOINT_POSITION_SHOULDER_LEFT = 5
ODMI_JOINT_POSITION_ELBOW_LEFT = 6
ODMI_JOINT_POSITION_WRIST_LEFT = 7
ODMI_JOINT_POSITION_HAND_LEFT = 8
ODMI_JOINT_POSITION_HANDTIP_LEFT = 9
ODMI_JOINT_POSITION_THUMB_LEFT = 10
ODMI_JOINT_POSITION_CLAVICLE_RIGHT = 11
ODMI_JOINT_POSITION_SHOULDER_RIGHT = 12
ODMI_JOINT_POSITION_ELBOW_RIGHT = 13
ODMI_JOINT_POSITION_WRIST_RIGHT = 14
ODMI_JOINT_POSITION_HAND_RIGHT = 15
ODMI_JOINT_POSITION_HANDTIP_RIGHT = 16
ODMI_JOINT_POSITION_THUMB_RIGHT = 17
ODMI_JOINT_POSITION_HIP_LEFT = 18
ODMI_JOINT_POSITION_KNEE_LEFT = 19
ODMI_JOINT_POSITION_ANKLE_LEFT = 20
ODMI_JOINT_POSITION_FOOT_LEFT = 21
ODMI_JOINT_POSITION_HIP_RIGHT = 22
ODMI_JOINT_POSITION_KNEE_RIGHT = 23
ODMI_JOINT_POSITION_ANKLE_RIGHT = 24
ODMI_JOINT_POSITION_FOOT_RIGHT = 25
ODMI_JOINT_POSITION_HEAD = 26
ODMI_JOINT_POSITION_COUNT = 27
######################################

class ODMI_SKELETON:
	vJointPositions: float = None
	fJointReliablity: float = None

class Skeleton_Fusion:
	def __init__(self, camnum, joint, pre_joint, vive_joints):
		self.camnum = camnum
		self.pre_joint = pre_joint
		self.joint = self.data_preprocess(joint)
		self.angle = []
		# self.joint = self.data_preprocess(joint) # It is for new joints
		self.Center = np.zeros([camnum, 3])
		self.bone_length = io.loadmat('BodyInfo.mat')['BodyInfo'] # Should be revised
		self.vivejoints = vive_joints

	def data_preprocess(self, data): # Should be revised
		skel = []
		tmp_A = 4 * ODMI_JOINT_POSITION_COUNT
		#tmp_B = tmp_A * self.camnum

		for i in range(0, self.camnum):
			skel.append(ODMI_SKELETON())
			skel_data = data[tmp_A * i:tmp_A * (i + 1)]
			tmp_skel = []
			tmp_real = []
			for j in range(0, ODMI_JOINT_POSITION_COUNT):
				tmp_skel.append([skel_data[4 * j], skel_data[4 * j + 1], skel_data[4 * j + 2]])
				tmp_real.append(skel_data[4 * j + 3])
			skel[i].vJointPositions = np.asarray(tmp_skel)
			skel[i].fJointReliability = np.asarray(tmp_real)
			#print(skel[i].vJointPositions)
			#print(skel[i].fJointReliability)
		return skel

	def Extract_FV_azure(self, skel, idx1, idx2):
		left_x = skel.vJointPositions[idx1][0]
		left_y = skel.vJointPositions[idx1][1]

		right_x = skel.vJointPositions[idx2][0]
		right_y = skel.vJointPositions[idx2][1]

		temp_wcVec = [0, 0]
		temp_wcVec[0] = right_x - left_x
		temp_wcVec[1] = right_y - left_y

		mag = pow(pow(temp_wcVec[0], 2) + pow(temp_wcVec[1], 2), 1 / 2.0)

		temp_wcVec[0] = temp_wcVec[0] / mag
		temp_wcVec[1] = temp_wcVec[1] / mag

		# Rotation in z axis (world coordinate)
		result = [0, 0]
		result[0] = -temp_wcVec[1]
		result[1] = temp_wcVec[0]

		return result

	def reliability_decision(self):
		# Weight Selection
		weight = np.ones([ODMI_JOINT_POSITION_COUNT, self.camnum])

		# Front Camera Selection
		if (np.size(self.pre_joint)) != 0: # Not for first frame
			Refer_FV = self.Extract_FV_azure(self.pre_joint, ODMI_JOINT_POSITION_HIP_LEFT, ODMI_JOINT_POSITION_HIP_RIGHT)

			for cam_idx in range(0, self.camnum):
				temp_FV = self.Extract_FV_azure(self.joint[cam_idx], ODMI_JOINT_POSITION_SHOULDER_LEFT, ODMI_JOINT_POSITION_SHOULDER_RIGHT)

				if(abs(temp_FV[0]*Refer_FV[0]+temp_FV[1]*Refer_FV[1]) <= 0.5):
					weight[:, cam_idx] = 0

		# Tracked/Inferred/Not tracked Skeleton decision
		tmp = [0.0] * ODMI_JOINT_POSITION_COUNT
		for skl_idx in range(0, ODMI_JOINT_POSITION_COUNT):
			sum = 0
			for cam_idx in range(0, self.camnum):
				if (self.joint[cam_idx].fJointReliability[skl_idx] == 2):
					weight[skl_idx, cam_idx] = weight[skl_idx, cam_idx] * 1
				elif (self.joint[cam_idx].fJointReliability[skl_idx] == 1):
					weight[skl_idx, cam_idx] = weight[skl_idx, cam_idx] * 0.01
				else:
					weight[skl_idx, cam_idx] = 0
				sum += weight[skl_idx, cam_idx]
			if (sum >= 1):
				tmp[skl_idx] = 1.0
			elif (sum == 0):
				tmp[skl_idx] = 0.0
			else:
				tmp[skl_idx] = 0.5

		return tmp, weight

	def position_decision(self, weight):
		F_U_L = np.zeros(2)
		F_U_R = np.zeros(2)
		F_L_L = np.zeros(2)
		F_L_R = np.zeros(2)

		for cam_idx in range(0, self.camnum):
			temp_ori = [0, 0]
			temp_ori[0] = self.Center[cam_idx, 0] - self.joint[cam_idx].vJointPositions[0, 0]
			temp_ori[1] = self.Center[cam_idx, 1] - self.joint[cam_idx].vJointPositions[0, 1]

			temp_norm = np.sqrt(pow(temp_ori[0], 2) + pow(temp_ori[1], 2))
			temp_ori = temp_ori / temp_norm

			F_U = self.Extract_FV_azure(self.joint[cam_idx], ODMI_JOINT_POSITION_SHOULDER_LEFT, ODMI_JOINT_POSITION_SHOULDER_RIGHT)
			F_L = self.Extract_FV_azure(self.joint[cam_idx], ODMI_JOINT_POSITION_HIP_LEFT, ODMI_JOINT_POSITION_HIP_RIGHT)
			angle = 3.141592 / 3.0

			F_U_L[0] = F_U[0] * np.cos(angle) - F_U[1] * np.sin(angle)
			F_U_L[1] = F_U[0] * np.sin(angle) + F_U[1] * np.cos(angle)
			F_U_R[0] = F_U[0] * np.cos(-angle) - F_U[1] * np.sin(-angle)
			F_U_R[1] = F_U[0] * np.sin(-angle) + F_U[1] * np.cos(-angle)

			F_L_L[0] = F_L[0] * np.cos(angle) - F_L[1] * np.sin(angle)
			F_L_L[1] = F_L[0] * np.sin(angle) + F_L[1] * np.cos(angle)
			F_L_R[0] = F_L[0] * np.cos(-angle) - F_L[1] * np.sin(-angle)
			F_L_R[1] = F_L[0] * np.sin(-angle) + F_L[1] * np.cos(-angle)

			W_U_Left = F_U_L[0] * temp_ori[0] + F_U_L[1] * temp_ori[1]
			W_U_Center = F_U[0] * temp_ori[0] + F_U[1] * temp_ori[1]
			W_U_Right = F_U_R[0] * temp_ori[0] + F_U_R[1] * temp_ori[1]
			W_L_Left = F_L_L[0] * temp_ori[0] + F_L_L[1] * temp_ori[1]
			W_L_Right = F_L_R[0] * temp_ori[0] + F_L_R[1] * temp_ori[1]

			direction_weight = 0.01

			# Left Upper Body
			if (W_U_Left < 0):
				weight[ODMI_JOINT_POSITION_SHOULDER_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_SHOULDER_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_ELBOW_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_ELBOW_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_WRIST_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_WRIST_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_HAND_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_HAND_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_HANDTIP_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_HANDTIP_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_THUMB_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_THUMB_LEFT, cam_idx] * direction_weight

			# Middle Upper Body
			if (W_U_Center < 0):
				weight[ODMI_JOINT_POSITION_PELVIS, cam_idx] = weight[ODMI_JOINT_POSITION_PELVIS, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_SPINE_NAVAL, cam_idx] = weight[ODMI_JOINT_POSITION_SPINE_NAVAL, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_NECK, cam_idx] = weight[ODMI_JOINT_POSITION_NECK, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_HEAD, cam_idx] = weight[ODMI_JOINT_POSITION_HEAD, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_SPINE_CHEST, cam_idx] = weight[ODMI_JOINT_POSITION_SPINE_CHEST, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_CLAVICLE_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_CLAVICLE_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_CLAVICLE_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_CLAVICLE_RIGHT, cam_idx] * direction_weight

			# Right Upper Body
			if (W_U_Right < 0):
				weight[ODMI_JOINT_POSITION_SHOULDER_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_SHOULDER_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_ELBOW_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_ELBOW_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_WRIST_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_WRIST_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_HAND_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_HAND_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_HANDTIP_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_HANDTIP_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_THUMB_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_THUMB_RIGHT, cam_idx] * direction_weight

			# Left Lower Body
			if (W_L_Left < 0):
				weight[ODMI_JOINT_POSITION_HIP_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_HIP_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_KNEE_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_KNEE_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_ANKLE_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_ANKLE_LEFT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_FOOT_LEFT, cam_idx] = weight[ODMI_JOINT_POSITION_FOOT_LEFT, cam_idx] * direction_weight

			# Right Lower Body
			if (W_L_Right < 0):
				weight[ODMI_JOINT_POSITION_HIP_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_HIP_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_KNEE_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_KNEE_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_ANKLE_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_ANKLE_RIGHT, cam_idx] * direction_weight
				weight[ODMI_JOINT_POSITION_FOOT_RIGHT, cam_idx] = weight[ODMI_JOINT_POSITION_FOOT_RIGHT, cam_idx] * direction_weight

		pos = [0.0] * ODMI_JOINT_POSITION_COUNT
		for i in range(0, ODMI_JOINT_POSITION_COUNT):
			sum_weight = 0
			for cam_idx in range(0, self.camnum):
				sum_weight += weight[i, cam_idx]
			weight[i, :] = weight[i, :] / sum_weight

			for cam_idx in range(0, self.camnum):
				pos[i] += self.joint[cam_idx].vJointPositions[i] * np.asarray(weight[i, cam_idx])

		return pos

	def position_normalization(self, skel):
		move_index = [1, 18, 22, 2, 3, 4, 11, 26, 5, 12, 6, 13, 7, 14, 8, 15, 9, 16, 10, 17, 19, 23, 20, 24, 21, 25]
		fix_index = [0, 0, 0, 1, 2, 2, 2, 3, 4, 11, 5, 12, 6, 13, 7, 14, 8, 15, 7, 14, 18, 22, 19, 23, 20, 24]

		temp_skel = np.asarray(skel)
		skel[0] = [0, 0, 0]

		for i in range(0, ODMI_JOINT_POSITION_COUNT - 1):
			temp_norm = np.linalg.norm(temp_skel[move_index[i]] - temp_skel[fix_index[i]])
			if (temp_norm != 0.0):
				unit_vector = (temp_skel[move_index[i]] - temp_skel[fix_index[i]]) / temp_norm
				skel[move_index[i]] = list(skel[fix_index[i]] + unit_vector * self.bone_length[i])
			else:
				skel[move_index[i]] = skel[fix_index[i]]

		if self.angle:
			for i in range(1, ODMI_JOINT_POSITION_COUNT):
				skel[i][0] = np.cos(self.angle)*skel[i][0] - np.sin(self.angle)*skel[i][1]
				skel[i][1] = np.sin(self.angle)*skel[i][0] + np.cos(self.angle)*skel[i][1]
		else:
			self.angle = self.calculate_angle(skel)
			for i in range(1, ODMI_JOINT_POSITION_COUNT):
				skel[i][0] = np.cos(self.angle)*skel[i][0] - np.sin(self.angle)*skel[i][1]
				skel[i][1] = np.sin(self.angle)*skel[i][0] + np.cos(self.angle)*skel[i][1]

		return skel

	def calculate_angle(self, skel):
		hip_x = skel[ODMI_JOINT_POSITION_HIP_RIGHT][0] - skel[ODMI_JOINT_POSITION_HIP_LEFT][0]
		hip_y = skel[ODMI_JOINT_POSITION_HIP_RIGHT][1] - skel[ODMI_JOINT_POSITION_HIP_LEFT][1]
		angle = np.arcsin(hip_y/np.sqrt(np.power(hip_x,2) + np.power(hip_y,2)))

		return angle

	def Fusion(self):
		result = ODMI_SKELETON()

		# Reliability decision
		[result.fJointReliablity, weight] = self.reliability_decision()

		# Fused positions calculation
		result.vJointPositions = self.position_decision(weight)

		# Fused positions normalization
		result.vJointPositions = self.position_normalization(result.vJointPositions)

		return result


