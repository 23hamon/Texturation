import numpy as np
import cv2

air_K_l = np.array([[
                    3291.181322249521,
                    0.0,
                    1510.08637472
                ],
                [
                    0.0,
                    3291.181322249521,
                    986.0160617
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]])

air_K_r = np.array([
                [
                    3302.023790071393,
                    0.0,
                    1493.22488246
                ],
                [
                    0.0,
                    3302.023790071393,
                    996.53835531
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ])
D_l = 40.0
D_r = 40.0

THICKNESS = 13
ETA_GLASS = 1.492
ETA_WATER = 1.34


if __name__ == "__main__" :
    x = np.array([-0.009622926, 0.1582513, -0.05655127, -216.6297, -6.304038, 20.39165, 0.006834929, -0.006016714, 0.9964889, -0.03711606, -0.006243035, 0.9757956])
    Rc = x[:3]
    Tc = x[3:6]
    n_l = x[6:9]
    n_r = x[9:]
    R_DG_calc, _ = cv2.Rodrigues(Rc)
    print("Rc = np.array(", np.array2string(Rc, separator=', '), ")", sep='')
    print("Tc = np.array(", np.array2string(Tc, separator=', '), ")", sep='')
    print("n_l = np.array(", np.array2string(n_l/np.linalg.norm(n_l), separator=', '), ")", sep='')
    print("n_r = np.array(", np.array2string(n_r/np.linalg.norm(n_r), separator=', '), ")", sep='')
    print("R_DG_calc = np.array(", np.array2string(R_DG_calc, separator=', ', precision=8), ")", sep='')

Rc = np.array([-0.00962293,  0.1582513 , -0.05655127])
Tc = np.array([-216.6297  ,   -6.304038,   20.39165 ])
n_l = np.array([ 0.00685873, -0.00603766,  0.99995825])
n_r = np.array([-0.03800845, -0.00639314,  0.99925697])
R_DG = np.array([[ 0.98591255,  0.05552497,  0.15777649],
 [-0.05704422,  0.99835856,  0.00511345],
 [-0.15723359, -0.01404165,  0.98746161]])
t_DG = -Tc

## troisieme calib jsp
# if __name__=="__main__":
#     x = np.array([  -0.00997663,    0.15734211,   -0.05691661, -214.3478,       -7.287022,
#     20.46281,       0.00426636,   -0.00005097,    0.9933951,    -0.03901951,
#     -0.0023926,     0.9871245 ])

#     Rc = x[:3]
#     Tc = x[3:6]
#     n_l = x[6:9]
#     n_r = x[9:]
#     R_DG_calc, _ = cv2.Rodrigues(Rc)
#     print("Rc = np.array(", np.array2string(Rc, separator=', '), ")", sep='')
#     print("Tc = np.array(", np.array2string(Tc, separator=', '), ")", sep='')
#     print("n_l = np.array(", np.array2string(n_l, separator=', '), ")", sep='')
#     print("n_r = np.array(", np.array2string(n_r, separator=', '), ")", sep='')
#     print("R_DG_calc = np.array(", np.array2string(R_DG_calc, separator=', ', precision=8), ")", sep='')

# Rc = np.array([-0.00997663,  0.15734211, -0.05691661])
# Tc = np.array([-214.3478  ,   -7.287022,   20.46281 ])
# n_l = np.array([ 4.266360e-03, -5.097000e-05,  9.933951e-01])
# n_r = np.array([-0.03901951, -0.0023926 ,  0.9871245 ])
# R_DG = np.array([[ 0.98603472,  0.05586743,  0.15688963],
#  [-0.0574335 ,  0.99833439,  0.00546276],
#  [-0.15632312, -0.0143972 ,  0.98760103]])
# t_DG = -Tc


# premiere calib
# n_l = np.array([
#                 0.03223987,
#                 0.0119772,
#                 0.99940839
#             ])

# n_r = np.array([
#                 -0.04773011,
#                 -0.02050776,
#                 0.99864972
#             ])
# RotationDroiteGauche = np.array([
#                 [
#                     0.9876747465511829,
#                     0.055341799355090704,
#                     0.14640997325728025
#                 ],
#                 [
#                     -0.05565357539749087,
#                     0.998448197144244,
#                     -0.0019690517729844233
#                 ],
#                 [
#                     -0.14629174471080872,
#                     -0.006203455774790089,
#                     0.989222039061968
#                 ]
#             ])

# TranslationDroiteGauche = np.array([
#                 210.29701361,
#                 7.24916971,
#                 -19.77955875
#             ])

# # nouvelle calib
# n_l_brut = np.array([0.01826953, -0.01171918, 0.9997644], dtype=np.float64)
# n_r_brut = np.array([-0.02498697, -0.00409035, 0.99967941], dtype=np.float64)

# n_r = n_r_brut / np.linalg.norm(n_r_brut)
# n_l = n_l_brut / np.linalg.norm(n_l_brut)

# Rc = np.array([-0.65741284, 9.19567167, -3.20664961], dtype=np.float64) # deg
# Tc = - np.array([-218.32875083, -5.95499246, 20.59005131], dtype=np.float64)

# # calcule avec le main :
# R_DG = np.array([[ 0.98558949,  0.05477774,  0.16003985],
#                  [-0.05661481,  0.99837199, 0.00693828],
#                  [-0.15939924, -0.01589892,  0.98708617]], dtype=np.float64)
# t_DG = Tc

# if __name__ == "__main__" :
#     Rc_rad = Rc * np.pi / 180
#     R_DG_calc, _ = cv2.Rodrigues(Rc_rad)
#     print(R_DG_calc)