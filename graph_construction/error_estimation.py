import numpy as np

def errors_parabola(u: np.array, sig_v: np.array):
  # Calculates the erorrs of the parabola parameters in the (u,v) space
  assert len(u) == len(sig_v)
  # We assume that all points have the same unitary weights f_n
  f_n = np.ones_like(u)
  F0 = f_n @ u**0
  F1 = f_n @ u**1
  F2 = f_n @ u**2
  F3 = f_n @ u**3
  F4 = f_n @ u**4
  det2334 = np.linalg.det([[F2,F3],[F3,F4]])
  det1234 = np.linalg.det([[F1,F2],[F3,F4]])
  det1223 = np.linalg.det([[F1,F2],[F2,F3]])
  det1324 = np.linalg.det([[F1,F3],[F2,F4]])
  det0224 = np.linalg.det([[F0,F2],[F2,F4]])
  det0123 = np.linalg.det([[F0,F1],[F2,F3]])
  det0112 = np.linalg.det([[F0,F1],[F1,F2]])
  u2 = u**2
  R = f_n * (  det2334 - u*det1234 + u2*det1223 )
  P = f_n * (- det1324 + u*det0224 - u2*det0123 )
  Q = f_n * (  det1223 - u*det0123 + u2*det0112 )
  # sig_x == sigma_x, sig2_x == (sigma_x)^2
  sig2 = sig_v**2
  sig2_A = sig2 @ R**2 /  np.sum(R)**2
  sig2_B = sig2 @ P**2 / (u    @ P)**2
  sig2_C = sig2 @ Q**2 / (u**2 @ Q)**2
  sig_AB = np.sum(sig2 * R * P) / (np.sum(R) * (u @ P))
  sig_BC = np.sum(sig2 * P * Q) / ((u @ P) * (u2 @ Q))
  sig_AC = np.sum(sig2 * R * Q) / (np.sum(R) * (u2 @ Q))
  return sig2_A, sig2_B, sig2_C, sig_AB, sig_BC, sig_AC


def errors_circle(a:float, b:float, R:float, epsilon:float,
                  u: np.array, sig_v: np.array):
  '''
  Calculates the errors of the circle parameters and the physical quantities

  - uses errors_parabola
  '''

  # First, calculate the erorrs of the parabola parameters in the (u,v) space:
  sig2_A, sig2_B, sig2_C, sig_AB, sig_BC, sig_AC = errors_parabola(u, sig_v)

  # Next, use them to estimate the errors for the parameters in the (x,y) space
  A = 1/(2*b)
  B = a/b
  C = epsilon * (R/b)**3
  oneA4 = 1/(4*A**4)
  sig2_a = oneA4 * (B**2 * sig2_A + A**2 * sig2_B - 2*A*B*sig_AB)
  sig2_b = oneA4 * sig2_A
  sig_ab = oneA4 * (-B * sig2_A + A*sig_AB)
  sig2_R = (1/R**2) * (b**2 * sig2_a + a**2 * sig2_b + 2*a*b*sig_ab)

  # Now estimate the errors for the physical quantities
  # relative momentum error (just sigma_P, without the denominator P)
  sig_P_rel = np.sqrt(sig2_R) / R
  # angular error
  sig_phi = np.sqrt(sig2_B) / (1+B**2)
  # impact parameter
  sig_eps = 1/(1+B**2)**3 * ( 9*(B*C)**2/(1+B**2)**2 * sig2_B + sig2_C
            - (6*B*C/(1+B)**2)*sig_BC )
  # returning the errors in the order corresponding to the order of this
  # function's arguments
  return sig2_a, sig2_b, sig2_R, sig_eps, sig_P_rel, sig_phi


def errors_test():
  # very naive testing of error calculation on senseless data, just sanity check:
  u = np.array([2.2, 8.1, 1.4, -2.4])
  sig_y = np.array([2.2, 8.1, 1.4, -2.4])
  x = u
  y = u
  sig_v = sig_y / (x**2 + y**2)
  #print(np.sum(u*u), u@u, )
  a, b, R, epsilon = 2, 3, 5.02, 1e-3
  #print(R)
  sig2_a, sig2_b, sig2_R, sig_eps, sig_P_rel, sig_phi = errors_circle(a, b, R, epsilon, u, sig_v)
  print(sig2_a, sig2_b, sig2_R, sig_eps, sig_P_rel, sig_phi )
