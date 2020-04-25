import numpy as np
import copy

EPSILON = 10e-10

class DubinsIntermediateResults:
	alpha = 0
	beta = 0
	d = 0
	sa = 0
	sb = 0
	ca = 0
	cb = 0
	c_ab = 0
	d_sq = 0
	def __init__(self):
		pass

class DubinsPath:
	qi = [0,0,0]
	param = [0,0,0]
	rho = 0
	type = "000"
	def __init__(self):
		pass

SegmentType = np.array(["LSL","LSR","RSL","RSR","RLR","LRL"])

def fmodr(x,y):
	return x - y * np.floor(x/y)

def mod2pi(theta):
	return fmodr(theta, 2 * np.pi)

def dubins_shortest_path(path, q0, q1, rho):
	params = [0,0,0]
	best_cost = 1e10
	best_word = -1
	inter = dubins_intermediate_results(q0, q1, rho)
	SegmentType = np.array(["LSL","LSR","RSL","RSR","RLR","LRL"])

	path.qi[0] = q0[0]
	path.qi[1] = q0[1]
	path.qi[2] = q0[2]
	path.rho = rho
	for i in range(6):
		pathType = SegmentType[i]
		dubins_word(inter, pathType, params)
		cost = params[0] + params[1] + params[2]
		if(cost < best_cost):
			best_word = i
			best_cost = cost
			path.param[0] = params[0]
			path.param[1] = params[1]
			path.param[2] = params[2]
			path.type = pathType;
	if(best_word == -1):
		print("no connection between configurations with this word")
	return

def dubins_path(path,q0, q1,rho,pathType):
	inter = dubins_intermediate_results(q0, q1, rho);
	params = [0,0,0];
	dubins_word(inter, pathType, params);
	path.param[0] = params[0]
	path.param[1] = params[1]
	path.param[2] = params[2]
	path.qi[0] = q0[0]
	path.qi[1] = q0[1]
	path.qi[2] = q0[2]
	path.rho = rho
	path.type = pathType
	return

def dubins_path_length(path):
	length = 0.0
	length += path.param[0];
	length += path.param[1];
	length += path.param[2];
	length = length * path.rho;
	return length;

def dubins_segment_length(path,i):
	if( (i < 0) or (i > 2) ):
		return 1e10
	return path.param[i] * path.rho

def dubins_segment_length_normalized(path,i):
	if((i < 0) or (i > 2)):
		return 1e10
	return path.param[i]

def dubins_path_type(path):
	return path.type

def dubins_segment(t,qi,qt,type):
	st = np.sin(qi[2]);
	ct = np.cos(qi[2]);
	if(type == "L"):
		qt[0] = np.sin(qi[2]+t) - st;
		qt[1] = -np.cos(qi[2]+t) + ct;
		qt[2] = t;
	elif(type == "R"):
		qt[0] = -np.sin(qi[2]-t) + st;
		qt[1] = np.cos(qi[2]-t) - ct;
		qt[2] = -t;
	elif(type == "S"):
		qt[0] = ct * t;
		qt[1] = st * t;
		qt[2] = 0.0;
	qt[0] += qi[0];
	qt[1] += qi[1];
	qt[2] += qi[2];
	return

def dubins_path_sample(path, t, q):
	# tprime is the normalised variant of the parameter t
	tprime = t / path.rho;
	# qi[3]; # The translated initial configuration
	# q1[3]; # end-of segment 1
	# q2[3]; # end-of segment 2
	qi = [0,0,0]
	q1 = [0,0,0]
	q2 = [0,0,0]
	# SegmentType = np.array(["LSL","LSR","RSL","RSR","RLR","LRL"])
	# print(path.type)
	# types = SegmentType[path.type];
	types = path.type
	if( t < 0 or t > dubins_path_length(path) ):
		print("Path parameterisitation error")
		return

	# initial configuration
	qi[0] = 0.0;
	qi[1] = 0.0;
	qi[2] = path.qi[2];

	# generate the target configuration
	p1 = path.param[0]
	p2 = path.param[1]
	dubins_segment(p1, qi, q1, types[0])
	dubins_segment(p2, q1, q2, types[1])
	if( tprime < p1 ):
		dubins_segment(tprime, qi, q, types[0]);
	elif( tprime < (p1+p2) ):
		dubins_segment(tprime-p1, q1, q, types[1]);
	else:
		dubins_segment(tprime-p1-p2, q2, q, types[2]);

	#scale the target configuration, translate back to the original starting point
	q[0] = q[0] * path.rho + path.qi[0];
	q[1] = q[1] * path.rho + path.qi[1];
	q[2] = mod2pi(q[2]);
	return

def dubins_path_sample_many(path, stepSize):
	length = dubins_path_length(path);
	x = 0.0
	q = [0,0,0]
	pts = []
	while(x < length):
		dubins_path_sample(path, x, q)
		pts.append(copy.copy(q))
		x += stepSize
	pts = np.array(pts)
	return pts,x

# def dubins_shortest_path(path, q0, q1, rho):
	# best_cost = 1e10
	# best_word = -1
	# inter = dubins_intermediate_results(q0,q1,rho)

def dubins_path_endpoint(path, q):
	return dubins_path_sample(path, dubins_path_length(path) - 1e-10, q)

def dubins_extract_subpath(path, t, newpath):
	tprime = t / path.rho
	if(t<0 or t > dubins_path_length(path)):
		print("Path parameterisitation error")
		return
	newpath.qi = copy.copy(path.qi)
	newpath.rho = path.rho
	newpath.type = path.type
	newpath.param[0] = fmin(path.param[0],tprime)
	newpath.param[1] = fmin(path.param[1], tprime - newpath.param[0]);
	newpath.param[2] = fmin( path.param[2], tprime - newpath.param[0] - newpath.param[1]);
	return 0

def dubins_intermediate_results(q0, q1, rho):
	if rho <= 0:
		print("rho value is invalid")
		return None
	dx = q1[0] - q0[0];
	dy = q1[1] - q0[1];
	D = ( dx * dx + dy * dy ) ** 0.5;
	d = D / rho;
	theta = 0;
	if (d > 0):
		theta = mod2pi(np.arctan2(dy,dx))
	alpha = mod2pi(q0[2] - theta)
	beta = mod2pi(q1[2] - theta)
	inter = DubinsIntermediateResults()
	inter.alpha = alpha
	inter.beta = beta
	inter.d = d
	inter.sa = np.sin(alpha)
	inter.sb = np.sin(beta)
	inter.ca = np.cos(alpha)
	inter.cb = np.cos(beta)
	inter.c_ab = np.cos(alpha-beta)
	inter.d_sq = d**2
	return inter

def dubins_LSL(inter,out):
	tmp0 = inter.d + inter.sa - inter.sb
	p_sq = 2 + inter.d_sq - 2*inter.c_ab + 2*inter.d*(inter.sa-inter.sb)
	if p_sq >= 0:
		tmp1 = np.arctan2(inter.cb-inter.ca,tmp0)
		out[0] = mod2pi(tmp1 - inter.alpha)
		out[1] = p_sq**0.5
		out[2] = mod2pi(inter.beta-tmp1)
		return
	# print("No connection between configurations with this word")
	return

def dubins_RSR(inter,out):
	tmp0 = inter.d - inter.sa + inter.sb
	p_sq = 2 + inter.d_sq - 2*inter.c_ab + 2*inter.d*(inter.sa-inter.sb)
	if p_sq >= 0:
		tmp1 = np.arctan2(inter.ca-inter.cb,tmp0)
		out[0] = mod2pi(inter.alpha - tmp1)
		out[1] = p_sq**0.5
		out[2] = mod2pi(tmp1 - inter.beta)
		return
	# print("No connection between configurations with this word")
	return

def dubins_LSR(inter,out):
	p_sq = 2 + inter.d_sq + 2*inter.c_ab + 2*inter.d*(inter.sa+inter.sb)
	if p_sq >= 0:
		p = p_sq**0.5
		tmp0 = np.arctan2(-inter.ca-inter.cb,inter.d+inter.sa+inter.sb)-np.arctan2(-2.0,p)
		out[0] = mod2pi(tmp0 - inter.alpha)
		out[1] = p
		out[2] = mod2pi(tmp0 - mod2pi(inter.beta))
		return
	# print("No connection between configurations with this word")
	return

def dubins_RSL(inter,out):
	p_sq = -2 + inter.d_sq + 2*inter.c_ab - 2*inter.d*(inter.sa+inter.sb)
	if p_sq >= 0:
		p = p_sq**0.5
		tmp0 = np.arctan2(inter.ca+inter.cb,inter.d-inter.sa-inter.sb)-np.arctan2(2.0,p)
		out[0] = mod2pi(inter.alpha-tmp0)
		out[1] = p
		out[2] = mod2pi(mod2pi(inter.beta)-tmp0)
		return
	# print("No connection between configurations with this word")
	return

def dubins_RLR(inter,out):
	tmp0 = (6.0 - inter.d_sq + 2*inter.c_ab + 2*inter.d*(inter.sa-inter.sb))/8.0
	phi = np.arctan2(inter.ca-inter.cb,inter.d-inter.sa+inter.sb)
	if abs(tmp0) <= 1:
		p = mod2pi((2*np.pi)-np.arccos(tmp0))
		t = mod2pi(inter.alpha - phi + mod2pi(p/2.0))
		out[0] = t
		out[1] = p
		out[2] = mod2pi(inter.alpha - inter.beta - t + mod2pi(p))
		return
	# print("No connection between configurations with this word")
	return

def dubins_LRL(inter,out):
	tmp0 = (6.0 - inter.d_sq + 2*inter.c_ab + 2*inter.d*(inter.sb-inter.sa))/8.0
	phi = np.arctan2(inter.ca-inter.cb,inter.d+inter.sa-inter.sb)
	if abs(tmp0) <= 1:
		p = mod2pi((2*np.pi)-np.arccos(tmp0))
		t = mod2pi(-inter.alpha - phi + (p/2.0))
		out[0] = t
		out[1] = p
		out[2] = mod2pi(mod2pi(inter.beta) - inter.alpha - t + mod2pi(p))
		return
	# print("No connection between configurations with this word")
	return
# SegmentType = np.array([[0,1,0],[0,1,2],[2,1,0],[2,1,2],[2,0,2],[0,2,0]])

def dubins_word(inter,pathType,out):
	if pathType == "LSL":
		return dubins_LSL(inter,out)
	elif pathType == "RSL":
		return dubins_RSL(inter,out)
	elif pathType == "LSR":
		return dubins_LSR(inter,out)
	elif pathType == "RSR":
		return dubins_RSR(inter,out)
	elif pathType == "LRL":
		return dubins_LRL(inter,out)
	elif pathType == "RLR":
		return dubins_RLR(inter,out)
	# print("Path parameterisitation error")
	return

def get_pts(q0,q1,turning_radius,step_size):
	q0[0], q0[1] = q0[1], q0[0]
	q1[0], q1[1] = q1[1], q1[0]
	path = DubinsPath()
	dubins_shortest_path(path,q0,q1,turning_radius)
	pts,cost = dubins_path_sample_many(path,step_size)
	pts[:,[0,1]] = pts[:,[1,0]]
	return pts,cost

def get_cost(q0,q1,turning_radius,step_size):
	q0[0], q0[1] = q0[1], q0[0]
	q1[0], q1[1] = q1[1], q1[0]
	path = DubinsPath()
	dubins_shortest_path(path,q0,q1,turning_radius)
	pts,cost = dubins_path_sample_many(path,step_size)
	return cost
