import numpy as np
# Parameters as in the traffic simulator. Change to do experiments.
width = 10
height = 10
carTypes = [0]#[0, 1, 2] # 0 = EV, 1 = Petrol, 2 = Diesel
pollutants = ['CO2']#['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMexhaustprueba', 'PMnonexhaust25', 'PMnonexhaust10']
times = 10
gamma = 0.01
delta = 0.1
corner_factor = 1
WN = 0.3
WE = 0.5
# Asume as given a family of tensors G of the form
G = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
				for cartype in carTypes
				for pollutant in pollutants}
# and an accessibility matrix
acc = np.ones((width+2,height+2))

for key in G:
	G[key][:width//2,:height//2,:times//2]=100

def get_wind(acc):
	# WE = East component of the wind, in the range [-1,1]
	# WN = North component of the wind, in the range [-1,1]
	# acc = Binary matrix of buildings, 1 if no building, 0 if building (acc stands for accesible, to pollution)
	displ_N = np.zeros((height+2, width+2))
	displ_S = np.zeros_like(displ_N)
	displ_E = np.zeros_like(displ_N)
	displ_W = np.zeros_like(displ_N)
	displ_NW = np.zeros_like(displ_N)
	displ_NE = np.zeros_like(displ_N)
	displ_SW = np.zeros_like(displ_N)
	displ_SE = np.zeros_like(displ_N)
	stays = np.ones_like(displ_N)
	sign_WN = np.sign(WN).astype(int)
	sign_WE = np.sign(WE).astype(int)
	for p in range(1, width+1):
		for q in range(1, height+1):
				displ_N[p,q] = acc[p,q] * np.maximum(WN, 0) * (1 - np.maximum(acc[p + sign_WE, q - 1], acc[p + sign_WE, q]) * abs(WE)) * acc[p, q - 1]
				displ_S[p,q] = acc[p,q] * np.maximum(-WN,0) * (1 - np.maximum(acc[p + sign_WE, q + 1], acc[p + sign_WE, q]) * abs(WE)) * acc[p, q + 1]
				displ_E[p,q] = acc[p,q] * np.maximum(WE, 0) * (1 - np.maximum(acc[p + 1, q - sign_WN], acc[p, q - sign_WN]) * abs(WN)) * acc[p + 1, q]
				displ_W[p,q] = acc[p,q] * np.maximum(-WE,0) * (1 - np.maximum(acc[p - 1, q - sign_WN], acc[p, q - sign_WN]) * abs(WN)) * acc[p - 1, q]
				displ_NE[p,q] = acc[p,q] * np.maximum(WN, 0) * np.maximum(WE, 0) * acc[p + 1, q - 1]
				displ_NW[p,q] = acc[p,q] * np.maximum(WN, 0) * np.maximum(-WE,0) * acc[p - 1, q - 1]
				displ_SE[p,q] = acc[p,q] * np.maximum(-WN,0) * np.maximum(WE, 0) * acc[p + 1, q + 1]
				displ_SW[p,q] = acc[p,q] * np.maximum(-WN,0) * np.maximum(-WE,0) * acc[p - 1, q + 1]
	stays += -(displ_N + displ_S + displ_E + displ_W + displ_NE + displ_NW + displ_SE + displ_SW)
	wind = (displ_N[1:-1, 2:], displ_S[1:-1, :-2], displ_E[:-2, 1:-1], displ_W[2:, 1:-1], displ_NE[:-2, 2:], displ_NW[2:, 2:], displ_SE[:-2, :-2], displ_SW[2:, :-2], stays[1:-1, 1:-1])
	return wind # The 9 components represent the wind displacements for the 8 neighbors and the cell itself, for each cell (therefore are matrices).

def get_difMatrix(acc):
	# The diffusion matrix, depending on the accesibility matrix.
	acc_neig_edge = (
		acc[0:-2, 1:-1] + acc[2:, 1:-1] +
		acc[1:-1, 0:-2] + acc[1:-1, 2:]
	)
	acc_neig_corner = (
		acc[0:-2, 0:-2] + acc[2:, 2:] +
		acc[2:, 0:-2] + acc[0:-2, 2:]
	)
	dif_matrix = (1 - (acc_neig_edge + acc_neig_corner * corner_factor) * delta / (4 + 4 * corner_factor))
	return dif_matrix

def get_P_last(acc):
	# This takes the emissions tensors G and the accesibility matrix acc, and produces the last pollution values (consuming less memory)
	# In the traffic simulator, the whole tensor G is not used, but instead is rewritten like P every timestep.
	P = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2), dtype=np.float32)
				for cartype in carTypes
				for pollutant in pollutants}
	wind = get_wind(acc)
	dif_matrix = get_difMatrix(acc)
	acc = acc[1:-1,1:-1]
	for i in range(times+1):
		for key in P:
			diffusion_edge = (
				P[key][0:-2, 1:-1] + P[key][2:, 1:-1] +    # Left and Right
				P[key][1:-1, 0:-2] + P[key][1:-1, 2:]      # Up and Down
			)
			diffusion_corner = (
				P[key][0:-2, 0:-2] + P[key][2:, 2:] +      # Diagonals
				P[key][2:, 0:-2] + P[key][0:-2, 2:]        # Diagonals
			)	
			# Diffusion
			P[key][1:-1,1:-1] = acc * (dif_matrix * P[key][1:-1, 1:-1] + delta * (diffusion_edge + diffusion_corner * corner_factor) / (4 + 4 * corner_factor))
			# Wind, sources and loss to atmosphere
			P[key][1:-1, 1:-1] = (1-gamma) * acc * (wind[0]*P[key][1:-1, 2:] + wind[1]*P[key][1:-1, :-2] + wind[2]*P[key][:-2, 1:-1] + wind[3]*P[key][2:, 1:-1] + wind[4]*P[key][:-2, 2:] + wind[5]*P[key][2:, 2:] + wind[6]*P[key][:-2, :-2] + wind[7]*P[key][2:, :-2] + wind[8]*P[key][1:-1, 1:-1] + G[key][1:-1, 1:-1, i])
	return P # returns last pollution map.

def get_P_whole(acc):
	# This takes the emissions tensor G and the accesibility matrix acc, and produces the whole pollution tensor P (consumes a lot of memory)
	P = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
				for cartype in carTypes
				for pollutant in pollutants}
	wind = get_wind(acc)
	dif_matrix = get_difMatrix(acc)
	acc = acc[1:-1,1:-1]
	for i in range(times+1):
		for key in P:
			diffusion_edge = (
				P[key][0:-2, 1:-1, i] + P[key][2:, 1:-1, i] +    # Left and Right
				P[key][1:-1, 0:-2, i] + P[key][1:-1, 2:, i]      # Up and Down
			)
			diffusion_corner = (
				P[key][0:-2, 0:-2, i] + P[key][2:, 2:, i] +      # Diagonals
				P[key][2:, 0:-2, i] + P[key][0:-2, 2:, i]        # Diagonals
			)	
			# Diffusion
			P[key][1:-1,1:-1, i] = acc * (dif_matrix * P[key][1:-1, 1:-1, i] + delta * (diffusion_edge + diffusion_corner * corner_factor) / (4 + 4 * corner_factor))
			# Wind, sources and loss to atmosphere
			if i < times:
				P[key][1:-1, 1:-1, i+1] = (1-gamma) * acc * (wind[0]*P[key][1:-1, 2:, i] + wind[1]*P[key][1:-1, :-2,i] + wind[2]*P[key][:-2, 1:-1,i] + wind[3]*P[key][2:, 1:-1,i] + wind[4]*P[key][:-2, 2:,i] + wind[5]*P[key][2:, 2:,i] + wind[6]*P[key][:-2, :-2,i] + wind[7]*P[key][2:, :-2,i] + wind[8]*P[key][1:-1, 1:-1,i] + G[key][1:-1, 1:-1, i])
	return P # returns whole pollution tensor. Useful to produce videos and debug
