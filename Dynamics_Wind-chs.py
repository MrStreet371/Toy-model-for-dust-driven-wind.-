
#CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
#                                                                      C
#          Toy model for dust driven wind                              C
#                                                                      C
#   Author:   Cayetano Hernández Sánchez                               C
#                                                                      C
#   Version:   06.10.26                                                C
#                                                                      C
#CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
os.makedirs("wind_output", exist_ok=True)
plt.style.use("default")


########## SETTING UP OF CONSTANTS ##########

G = 6.6743e-11          # gravitational constant
c = 2.99792458e8        # speed of light
sigma = 5.670374e-8     # Stefan-Boltzmann constant
M_sun = 1.98841e30      # kg
L_sun = 3.828e26        # W
R_sun = 6.957e8         # m
AU = 1.495978707e11     # m
yr = 3.15576e7          # s


########## STELLAR PARAMETERS AND QUANTITIES ##########
M_star = 1.5 * M_sun
L_star = 7000 * L_sun
T_star = 2600.0         # K

R_star = np.sqrt(L_star / (4 * np.pi * sigma * T_star**4))   # stellar radius
u_esc = np.sqrt(2 * G * M_star / R_star)                     # escape velocity at R*
t_ff = R_star / u_esc                                        # time unit R*/u_esc
g0 = G * M_star / R_star**2                                  # gravity at R*
R_AU = R_star / AU                                           # R* in AU

U0 = 0.8                # Initial velocity---


def to_yr(t): # To transform from dimensionless to physical
    
    return np.asarray(t) * t_ff / yr


def to_kms(u):
    
    return np.asarray(u) * u_esc / 1e3

print(f"  R*     = {R_star:.4e} m = {R_AU:.3f} AU = {R_star / R_sun:.0f} ")
print(f"  u_esc  = {u_esc / 1e3:.2f} km/s,  u0 = {U0 * u_esc / 1e3:.2f}")
print(f"  t_ff   = R*/u_esc = {t_ff:.3e} s = {t_ff / yr:.4f} ")


########## R_C AND GAMMA FOR GRAIN BASED MODEL ##########


T_c = 1500.0            # condensation temperature in K
p = 1.0                 
Q_over_a = 2.0e6        # 1/m
A_mon = 12.0            # monomer aw (C)
rho_grain = 1.85e3      # density of the grain ( kg/m^3)
eps_c = 3.3e-4          # carbon excess
eps_He = 0.1            # abundance of helium
f_c = 0.2               # degree of condensation



# Condensation distance (Eq 4):

Rc_grain = 0.5 * (T_c / T_star) ** (-(4 + p) / 2)

# Dust opacity (Eq 9):

kappa = 0.75 * (A_mon / rho_grain) * Q_over_a * eps_c / (1 + 4 * eps_He) * f_c

# Ratio of radiative to gravitational acceleration (Generalized Eddington??)

Gamma_grain = kappa * L_star / (4 * np.pi * c * G * M_star)


print(f"  Rc     = {Rc_grain:.4f} R* = {Rc_grain * R_AU:.3f} ")
print(f"  kappa  = {kappa:.4f} ")
print(f"  Gamma  = {Gamma_grain:.4f}")



########## NUMERICAL INTEGRATION ##########


def equations_of_motion(t, y, Gamma): # dimensionless eom
    r, u = y
    drdt = u
    dudt = -(1 - Gamma) / (2 * r**2)
    return [drdt, dudt]


def falls_back(t, y):#to stop the run if the element falls back
    return y[0] - 1.0
falls_back.terminal = True      # Stop the integration 
falls_back.direction = -1      


def solve_stage(Gamma, t_start, y_start, t_end, t_eval, events): # Integrate the equation using DOP853 (cte Gamma) 

    return solve_ivp(lambda t, y: equations_of_motion(t, y, Gamma),
                     (t_start, t_end), y_start,
                     method="DOP853", rtol=1e-10, atol=1e-12,
                     t_eval=t_eval, events=events,
                     dense_output=True)    # should keep an interpolant at any time


#TRAYECTORIES:

def integrate(Gamma, Rc, t_end=25, n_points=2000):

    t_eval = np.linspace(0, t_end, n_points) #create evaluation times      

    def reaches_Rc(t, y): # element reaches Rc
        return y[0] - Rc
    reaches_Rc.terminal = True
    reaches_Rc.direction = 1

    ######## DUST FREE GAS: GAMMA = 0 // Stage 1 ########
    
    stage1 = solve_stage(0.0, 0.0, [1.0, U0], t_end, t_eval, [falls_back, reaches_Rc])

    Rc_reached = len(stage1.t_events[1]) > 0
    
    if not Rc_reached:                              
        return {"t": stage1.t, "r": stage1.y[0], "u": stage1.y[1],
                "t_Rc": None, "stages": [stage1]}

    ######### DUSTY GAS: GAMMA CTE (Starting r = Rc) // Stage 2 #########
    
    t_Rc = stage1.t_events[1][0]
    y_Rc = stage1.y_events[1][0]                    # [r, u] at the crossing
    stage2 = solve_stage(Gamma, t_Rc, y_Rc, t_end, t_eval[t_eval > t_Rc], [falls_back])

    # Join both stages!!!
    t = np.concatenate([stage1.t, [t_Rc], stage2.t])
    r = np.concatenate([stage1.y[0], [y_Rc[0]], stage2.y[0]])
    u = np.concatenate([stage1.y[1], [y_Rc[1]], stage2.y[1]])
    return {"t": t, "r": r, "u": u, "t_Rc": t_Rc, "stages": [stage1, stage2]}


def state_at_time(traj, time): #give (r,u) at any time
    stage1 = traj["stages"][0]
    if traj["t_Rc"] is None or time <= traj["t_Rc"]:
        return stage1.sol(time)
    stage2 = traj["stages"][1]
    return stage2.sol(time)
    
    
    
########## ANALYTICAL SOLUTION ##########


def u_analytic(r, Gamma, Rc): #Outward velocity 

    u2_at_Rc = U0**2 + 1 / Rc - 1                          # from R* to Rc with Gamma = 0
    u2_dust_free = U0**2 + 1 / r - 1                       # r < Rc  (Gamma = 0)
    u2_dusty = u2_at_Rc + (1 - Gamma) * (1 / r - 1 / Rc)  # r >= Rc (Gamma = ccte)
    return np.sqrt(np.where(r < Rc, u2_dust_free, u2_dusty))


def terminal_velocity(Gamma, Rc): #Having taken the limit...

    u2_at_Rc = U0**2 + 1 / Rc - 1
    if u2_at_Rc <= 0:                                
        return np.nan
    u2_inf = u2_at_Rc + (Gamma - 1) / Rc
    return np.sqrt(u2_inf) if u2_inf > 0 else np.nan


def escape_radius(Gamma, Rc):
    return Gamma / terminal_velocity(Gamma, Rc) ** 2






r_max_freefall = 1 / (1 - U0**2)                       # max without dust, mirar Eq. 3 de las notas 
u_Rc = np.sqrt(U0**2 + 1 / Rc_grain - 1)               # velocity at Rc
u_inf = terminal_velocity(Gamma_grain, Rc_grain)       # terminal velocity
r_esc = escape_radius(Gamma_grain, Rc_grain)           




########## TEST CASES ########## 

#for comparison with fig 1 of the notes



title = r"$M_*=1.5\,M_\odot$, $L_*=7000\,L_\odot$, $T_*=2600$ K"
 
# Integrate the test cases: integrate(Gamma, Rc)
case_G5 = integrate(5.0, 2.5)     # Gamma = 5, Rc = 2.5 R*  (appears in both panels)
case_G1 = integrate(1.0, 2.5)     # Gamma = 1, Rc = 2.5 R*
case_G0 = integrate(0.0, 2.5)     # Gamma = 0, Rc = 2.5 R*
case_Rc2 = integrate(5.0, 2.0)    # Gamma = 5, Rc = 2.0 R*
case_Rc3 = integrate(5.0, 3.0)    # Gamma = 5, Rc = 3.0 R*
 
fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6.5, 9))
 
# Pannel for fixed Rc
ax_top.plot(to_yr(case_G5["t"]), case_G5["r"], color="tab:blue",
            label=r"$\Gamma = 5.0$, $R_c = 2.5\,R_*$")
ax_top.plot(to_yr(case_G1["t"]), case_G1["r"], color="tab:orange",
            label=r"$\Gamma = 1.0$, $R_c = 2.5\,R_*$")
ax_top.plot(to_yr(case_G0["t"]), case_G0["r"], color="tab:green",
            label=r"$\Gamma = 0.0$, $R_c = 2.5\,R_*$")
            
            
# Pannel for fixed Gamma

ax_bottom.plot(to_yr(case_Rc2["t"]), case_Rc2["r"], color="tab:blue",
               label=r"$R_c = 2.0$, $\Gamma = 5.0$")
ax_bottom.plot(to_yr(case_G5["t"]), case_G5["r"], color="tab:orange",
               label=r"$R_c = 2.5$, $\Gamma = 5.0$")
ax_bottom.plot(to_yr(case_Rc3["t"]), case_Rc3["r"], color="tab:green",
               label=r"$R_c = 3.0$, $\Gamma = 5.0$")
 

for ax in [ax_top, ax_bottom]:
    ax.set_xlim(0, 2.7)
    ax.set_ylim(1, 7)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(r"Radial distance [$R_*$]")
    ax.set_title(title)
    ax.legend(loc="upper left")
 
plt.tight_layout()
plt.savefig("wind_output/fig1_test_cases.png", dpi=200)
 
# Quick test:
#traj = integrate(0.0, 2.5, n_points=20000)
#print(f"\nCODE TEST (Gamma = 0): r_max numerical = {traj['r'].max():.5f} R*, "
     #f"analytical = {r_max_freefall:.5f} R*")
     
     
     
 
########## GRAIN MODEL VS TEST CASES ########## 

grain_label = rf"Grains: $\Gamma={Gamma_grain:.2f}$, $R_c={Rc_grain:.2f}\,R_*$"
 
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

# Fixed Rc = 2.5 R* 
ax_left.plot(to_yr(case_G5["t"]), case_G5["r"], "--", color="tab:blue", linewidth=2,
             label=r"Test: $\Gamma = 5.0$, $R_c = 2.5\,R_*$")
ax_left.plot(to_yr(case_G1["t"]), case_G1["r"], "--", color="tab:orange", linewidth=2,
             label=r"Test: $\Gamma = 1.0$, $R_c = 2.5\,R_*$")
ax_left.plot(to_yr(case_G0["t"]), case_G0["r"], "--", color="tab:green", linewidth=2,
             label=r"Test: $\Gamma = 0.0$, $R_c = 2.5\,R_*$")
ax_left.set_title(r"Test cases with fixed $R_c = 2.5\,R_*$ (varying $\Gamma$)")
 
# Fixed Gamma
ax_right.plot(to_yr(case_Rc2["t"]), case_Rc2["r"], "--", color="tab:blue", linewidth=2,
              label=r"Test: $R_c = 2.0$, $\Gamma = 5.0$")
ax_right.plot(to_yr(case_G5["t"]), case_G5["r"], "--", color="tab:orange", linewidth=2,
              label=r"Test: $R_c = 2.5$, $\Gamma = 5.0$")
ax_right.plot(to_yr(case_Rc3["t"]), case_Rc3["r"], "--", color="tab:green", linewidth=2,
              label=r"Test: $R_c = 3.0$, $\Gamma = 5.0$")
ax_right.set_title(r"Test cases with fixed $\Gamma = 5$ (varying $R_c$)")
 
# Plotting with grain model

grain = integrate(Gamma_grain, Rc_grain)

for ax in [ax_left, ax_right]:
    ax.plot(to_yr(grain["t"]), grain["r"], color="black", linewidth=3, zorder=1,
            label=grain_label)
    ax.set_xlim(0, 4)
    ax.set_ylim(1, 10)
    ax.set_xlabel("Time [yr]")
    ax.legend(loc="upper left")
ax_left.set_ylabel(r"Radial distance [$R_*$]")
 
plt.suptitle(title)
plt.tight_layout()
plt.savefig("wind_output/fig2_grain_model_vs_tests.png", dpi=200)
 
 



########## NUMERICAL VS ANALYTICAL ########## 

wind = integrate(Gamma_grain, Rc_grain, t_end=150, n_points=15000)
r_num, u_num = wind["r"], wind["u"]
 
 
def numerical_u_at(r_target): #Numerical (r, u) at r = r_target
    t_target = np.interp(r_target, r_num, wind["t"])
    return state_at_time(wind, t_target)
 
 
# Comparison table at selected radii: below Rc, at Rc and above Rc
check_radii = [1.25, 1.5, 1.9, Rc_grain, 2.1, 3.0, 5.0, 10.0, 20.0]


rows = []
for r_i in check_radii:
    r_found, u_n = numerical_u_at(r_i)
    u_a = u_analytic(r_found, Gamma_grain, Rc_grain)
    error = abs(u_n - u_a) / u_a
    print(f"{r_found:9.4f} {to_kms(u_n):14.5f} {to_kms(u_a):14.5f} {error:12.2e}")
    rows.append([r_found, to_kms(u_n), to_kms(u_a), error])
rows = np.array(rows)
np.savetxt("wind_output/velocity_check_table.csv", rows, delimiter=",",
           header="r_over_Rstar,u_numerical_kms,u_analytical_kms,relative_error")


#Important to calculate now the relative difference at any point: 

inside = (r_num >= 1) & (r_num <= 25)
u_ana_all = u_analytic(r_num[inside], Gamma_grain, Rc_grain)
residual = np.abs(u_num[inside] - u_ana_all) / u_ana_all
i_max = residual.argmax()

print(f"\nResiduals entre r 1 y 25, ptos {inside.sum()} ")
print(f"  maximum = {residual[i_max]:.1e} at r = {r_num[inside][i_max]:.2f} R*")
print(f"  median  = {np.median(residual):.1e}")
 
# Numerical solution between 1 and 25 R*
r_numerical = r_num[inside]
u_numerical = to_kms(u_num[inside])
 
# Analytical solution and local escape velocity on a fine grid


r_line = np.linspace(1, 25, 1000)
r_line = np.sort(np.append(r_line, Rc_grain))

u_analytical = to_kms(u_analytic(r_line, Gamma_grain, Rc_grain)) #ansol
u_esc_local = to_kms(1 / np.sqrt(r_line)) #Local escape velocity
 
# Value for the comparison table
r_table = rows[:, 0]
u_table = rows[:, 1]
error_table = rows[:, 3]
 


#TOP: Velocities
fig, (ax, ax_res) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
 

ax.plot(r_numerical, u_numerical, color="black", linewidth=5, label="Numerical (DOP853)")
ax.plot(r_line, u_analytical, "--", color="tab:orange", linewidth=2, label="Analytical")
ax.plot(r_table, u_table, "o", color="tab:green", markersize=8, markeredgecolor="black",
        zorder=5, label="Selected radii (comparison table)")        
ax.plot(r_line, u_esc_local, "--", color="tab:red", linewidth=1.6,
        label="Local escape velocity")
ax.axhline(to_kms(u_inf), color="tab:blue", linestyle="--", linewidth=1.8,
           label=rf"Terminal velocity $u_\infty={to_kms(u_inf):.1f}$ km/s")
ax.axvline(Rc_grain, color="grey", linestyle="--", linewidth=1.3,
           label=rf"$R_c={Rc_grain:.2f}\,R_*$")
ax.set_ylim(0, 1.35 * to_kms(1.0))
ax.set_ylabel("Velocity [km/s]")
ax.set_title(title + rf", $\Gamma={Gamma_grain:.2f}$")
ax.legend(loc="upper right")
 
#BOTTOM: Relative difference


ax_res.plot(r_numerical, np.maximum(residual, 1e-17), color="black", linewidth=1.6)
ax_res.plot(r_table, error_table, "o", color="tab:green", markersize=7,
            markeredgecolor="black", zorder=5)
ax_res.axvline(Rc_grain, color="grey", linestyle="--", linewidth=1.3)
ax_res.set_yscale("log")
ax_res.set_xlim(1, 25)
ax_res.set_ylim(1e-14, 1e-9)
ax_res.set_xlabel(r"Distance from the stellar centre [$R_*$]")
ax_res.set_ylabel(r"$|u_{num}-u_{ana}|\,/\,u_{ana}$")
 
plt.tight_layout()
plt.savefig("wind_output/fig3_velocity_vs_radius.png", dpi=200)
 
 




#Orbits of celestial bodies:

ORBITS = {"Mercury": 0.387, "Venus": 0.723, "Earth": 1.000, "Mars": 1.524,
          "Ceres": 2.767, "Jupiter": 5.203, "Saturn": 9.537,
          "Uranus": 19.19, "Neptune": 30.07, "Pluto": 39.48}
          
          
X_MAX = 45                                   # edge


#chango to physical 
Rc_AU = Rc_grain * R_AU                      
r_esc_AU = r_esc * R_AU                    
u_at_r_esc = to_kms(1 / np.sqrt(r_esc))      
u_pluto = to_kms(numerical_u_at(ORBITS["Pluto"] / R_AU)[1])
pluto_percent = 100 * u_pluto / to_kms(u_inf)
y_top = 1.55 * to_kms(1.0)                 
 
fig, ax = plt.subplots(figsize=(11, 6))
 
# Shade regions
ax.axvspan(0, R_AU,  color="#FFA54F", alpha=0.8, linewidth=0)
ax.axvspan(R_AU, Rc_AU, color="silver", alpha=0.20, linewidth=0)
ax.axvspan(Rc_AU, X_MAX, color="tab:blue", alpha=0.07, linewidth=0)
 
# Num solution
r_AU = r_num * R_AU
mask = (r_AU >= R_AU) & (r_AU <= X_MAX)
ax.plot(r_AU[mask], to_kms(u_num[mask]), color="black", linewidth=3, label="Wind velocity")
 
# Local escape velocity and terminal velocity

r_grid_AU = np.linspace(R_AU, X_MAX, 500)
u_esc_grid = to_kms(1.0) * np.sqrt(R_AU / r_grid_AU)
ax.plot(r_grid_AU, u_esc_grid, "--", color="tab:red", linewidth=1.8,
        label="Local escape velocity")
ax.axhline(to_kms(u_inf), color="tab:blue", linestyle="-.", linewidth=1.8,
           label=f"Terminal velocity {to_kms(u_inf):.1f} km/s")
 
# Points where the wind exceeds the local escape velocity and also where Plutois
ax.plot(r_esc_AU, u_at_r_esc, "o", color="tab:red", markersize=9, markeredgecolor="black",
        label=r"$u = u_{esc}(r)$" + f" at {r_esc_AU:.1f} AU")
ax.plot(ORBITS["Pluto"], u_pluto, "s", color="tab:green", markersize=9, markeredgecolor="black",
        label=f"Pluto's orbit: {u_pluto:.2f} km/s ({pluto_percent:.1f}% of terminal)")
 
#Planets

for name, a in ORBITS.items():
    if a < R_AU:
        continue
    ax.axvline(a, color="grey", linestyle=":", linewidth=1)
    ax.text(a, 0.98 * y_top, name, rotation=90, ha="right", va="top", fontsize=9)
ax.text(1.0, 0.98 * y_top, "Mercury, Venus, Earth, Mars", rotation=90, ha="center",
        va="top", fontsize=9)
 
# Names of the three regions
ax.text(R_AU / 2, 1.0, "star", rotation=90, ha="center", va="bottom", fontsize=9)
ax.text((R_AU + Rc_AU) / 2, 1.0, "dust-free", rotation=90, ha="center", va="bottom", fontsize=9)
ax.text(16, 1.5, "dusty wind", ha="center", fontsize=10)
 
ax.set_xlim(0, X_MAX)
ax.set_ylim(0, y_top)
ax.set_xlabel("Distance from the stellar centre [AU]")
ax.set_ylabel("Velocity [km/s]")
ax.set_title("Dust-driven wind in astronomical units")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)    # legend below the plot
 
plt.tight_layout()
plt.savefig("wind_output/fig4_velocity_AU.png", dpi=200, bbox_inches="tight")


########## 

r_pluto = ORBITS["Pluto"] / R_AU                                 # Pluto's orbit in R*

# 99%???
r99 = (Gamma_grain - 1) / ((1 - 0.99**2) * u_inf**2)             # 

#print(f"\n  r_max (Gamma = 0) = {r_max_freefall:.2f} R*")


print(f"  u(Rc)  = {to_kms(u_Rc):.2f}  = {100 * u_Rc * np.sqrt(Rc_grain):.0f}% of the local u_esc")
print(f"  u_inf  = {to_kms(u_inf):.2f} ")
print(f"  r_esc  = {r_esc:.2f} R* = {r_esc * R_AU:.1f} ")
print(f"  99% of u_inf at {r99 * R_AU:.0f} ")
print(f"  Pluto: r = {r_pluto:.1f} , u = {u_pluto:.2f} km/s = {pluto_percent:.1f}%")
print(f"         u / lu_esc = {u_pluto / to_kms(1 / np.sqrt(r_pluto)):.1f}")
print(f"         Ganancia= 1 - Rc/r = {1 - Rc_grain / r_pluto:.2f}")
print(f"  Tiempo alcance rc {to_yr(grain['t_Rc']):.2f} , "
      f"Gamma = 5,Rc = 2.0: {to_yr(case_Rc2['t_Rc']):.2f}")




# Gamma = 5, Rc = 2.5 overtaking
t_grid = np.linspace(case_G5["t_Rc"], 4 / (t_ff / yr), 5000)
r_test = np.interp(t_grid, case_G5["t"], case_G5["r"])
r_grain = np.interp(t_grid, grain["t"], grain["r"])
k = np.argmax(r_test > r_grain)                        
print(f"  Gamma = 5, Rc = 2.5 overtakes at t = {to_yr(t_grid[k]):.2f}")

