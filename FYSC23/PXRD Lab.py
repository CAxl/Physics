import numpy as np
import matplotlib.pyplot as plt



# class CrystalStructure():
	
# 	def __init__(self, *args):
# 		samples = []
# 		for i in args:
# 			samples[i] = i
		
# 		self.samples = samples
		
# 	def get_lattice_param(self, double_angle):
# 		if type(double_angle) != np.ndarray:
# 			double_angle = np.array(double_angle)
# 		theta = double_angle/2
		
# 		sin_sq = (np.sin(theta))**2
# 		normalized = sin_sq/min(sin_sq)
		
# 		return normalized



# # ----------------- raw sample data ------------------
	
# sample_one = np.array([17.61, 20.33, 28.86, 33.88, 35.42]) * np.pi/180
# sample_two = np.array([20.38, 28.87, 35.46]) * np.pi / 180

# print(sample_two*180/(2*np.pi))
# print((np.sin(sample_two/2))**2)

# # ---------------- functions ------------------
# def interp(double_angle):
# 	theta = double_angle/2
# 	sin_sq = (np.sin(theta))**2
# 	normalized = sin_sq/min(sin_sq)
# 	return normalized



# def d(theta, lamb = 0.7107):
# 	return lamb / (2*np.sin(theta))


# def a(d,h,k,l):
# 	return d*np.sqrt(h**2 + k**2 + l**2)


# # ---------------------- sample 1 ------------------------
# n1 = interp(sample_one)

# print("normalized sample_1:", n1)
# print(3*n1)
# # -> (111) (200) (220) (311) (222)

# ds1 = d(sample_one/2)

# print("lattice parameter, a, sample 1:")
# print(a(ds1[0],1,1,1))
# print(a(ds1[1],2,0,0))
# print(a(ds1[2],2,2,0))
# print(a(ds1[3],3,1,1))
# print(a(ds1[4],2,2,2))

# print("Average lattice param: a_1_avg = ", ( a(ds1[0],1,1,1) + a(ds1[1],2,0,0) + a(ds1[2],2,2,0) + a(ds1[3],3,1,1) + a(ds1[4],2,2,2) ) / 5 )


# # ----------------- sample 2 --------------------

# n2 = interp(sample_two)
# print("normalized sample_2:", n2)
# print(2*n2)
# # -> (110) (200) (211)

# ds2 = d(sample_two/2)

# print("lattice parameter, a, sample 2:")
# print(a(ds2[0],1,1,0))
# print(a(ds2[1],2,0,0))
# print(a(ds2[2],2,1,1))

# print("Average lattice param: a_2_avg = ", ( a(ds2[0],1,1,0) + a(ds2[1],2,0,0) + a(ds2[2],2,1,1) ) / 3)




# def Scherrer(theta,beta,k=0.94,lamb=0.7107):
# 	t = lamb*k /(beta * np.cos(theta))
# 	return t

# FWHM1 = np.array([0.385, 0.438, 0.399, 0.443, 0.456]) * np.pi/180
# FWHM2 = np.array([0.338, 0.386, 0.407]) * np.pi/180
# t_1 = Scherrer(sample_one/2, FWHM1)
# t_2 = Scherrer(sample_two/2, FWHM2)

# print("t-values")
# print(np.average(t_1))
# print(np.average(t_2))




# # def d_orthorhommb(h,k,l):
# # 	a = 0.574
# # 	b = 0.796
# # 	c = 0.495
	
# # 	return 1/np.sqrt((h**2 / a**2) + (k**2 / b**2) + (l**2 / c**2))



# # print(d_orthorhommb(1,0,0))
# # print(d_orthorhommb(0,1,0))
# # print(d_orthorhommb(1,1,1))

# # def theta(d, n=1, lamb=0.083):
# # 	return np.arcsin((n*lamb) / (2*d))

# # print("theta(1,0,0) = ", theta(d_orthorhommb(1,0,0)) * 180/np.pi)
# # print("theta(0,1,0) = ", theta(d_orthorhommb(0,1,0)) * 180/np.pi)
# # print("theta(1,1,1) = ", theta(d_orthorhommb(1,1,1)) * 180/np.pi)





h = 6.626 * 1e-34
e = 1.667 * 1e-19
m_e = 9.1093837 * 1e-31
lamb = 1.7 * 1e-10
c = 300000000

print((h/lamb) * c * 6.24150907 * 1e18)
print(h/lamb)

print(3000 / (np.pi * 0.002**2))









hbar = h/(2*np.pi)
a = 0.357 * 1e-9
TA = 90 * 1e-3 * 1.60217663 * 10**(-19)	# J
LA = 150 * 1e-3 * 1.60217663 * 10**(-19)

print("v_AT = ", a*TA / (2*np.pi*hbar))
print("v_LA = ", a*LA / (2*np.pi*hbar))


sigma = 6.2 * 1e5
n = 5.85 * 1e22
print(m_e * sigma /(n*e**2) )



