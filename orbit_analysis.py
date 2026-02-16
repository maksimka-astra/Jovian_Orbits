import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

name = "Ganymede"
dt = 600 # seconds
G = 6.67430e-20  # km^3 / (kg s^2)
orbit_periods = 100

# ==============================
# ARRAYS
# ==============================

a_real_list = []
T_real_list = []
epsilon_real_list = []

a_num_list = []
T_num_list = []
epsilon_num_list = []

trajectory = []

# ==============================
# PARCING
# ==============================

def load_horizons_csv(filename):
    data = []
    reading = False
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line == "$$SOE":
                reading = True
                continue
            if line == "$$EOE":
                break
            if not reading:
                continue

            parts = [p.strip() for p in line.split(",")]

            data.append({
                "x":  float(parts[2]),
                "y":  float(parts[3]),
                "z":  float(parts[4]),
                "vx": float(parts[5]),
                "vy": float(parts[6]),
                "vz": float(parts[7])
            })
    return data

# ==============================
# JUPITER-CENTRIC SYSTEM
# ==============================
def relative_orbit(
    satellite_csv,
    central_body_csv="jupiter.csv",
    output_csv=None
):
    sat_data = load_horizons_csv(satellite_csv)
    cen_data = load_horizons_csv(central_body_csv)

    rel_data = []
    for s, c in zip(sat_data, cen_data):
        rel_data.append({
            "x":  s["x"]  - c["x"],
            "y":  s["y"]  - c["y"],
            "z":  s["z"]  - c["z"],
            "vx": s["vx"] - c["vx"],
            "vy": s["vy"] - c["vy"],
            "vz": s["vz"] - c["vz"]
        })
    if output_csv is not None:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z", "vx", "vy", "vz"])
            for p in rel_data:
                writer.writerow([
                    p["x"], p["y"], p["z"],
                    p["vx"], p["vy"], p["vz"]
                ])
    return rel_data

# ==============================
# APOGEES & PEREGEES
# ==============================

def find_perigees_and_apogees(rel_data):

    x = np.array([p["x"] for p in rel_data])
    y = np.array([p["y"] for p in rel_data])
    z = np.array([p["z"] for p in rel_data])
    vx = np.array([p["vx"] for p in rel_data])
    vy = np.array([p["vy"] for p in rel_data])
    vz = np.array([p["vz"] for p in rel_data])

    r = np.sqrt(x**2 + y**2 + z**2)
    vr = (x*vx + y*vy + z*vz) / r
    apogees = np.where((vr[:-1] > 0) & (vr[1:] <= 0))[0] + 1
    perigees = np.where((vr[:-1] < 0) & (vr[1:] >= 0))[0] + 1
    return perigees.tolist(), apogees.tolist()


if __name__ == "__main__":
    satellite_csv = name+".csv"
    central_body_csv = "jupiter.csv"
    output_csv = "relative.csv"

    rel_data = relative_orbit(satellite_csv, central_body_csv, output_csv)
    perigees, apogees = find_perigees_and_apogees(rel_data)
    print(f"Apogees: {len(perigees)}, Perigees: {len(apogees)}")


# ==============================
# INITIAL ORBIT
# ==============================

p0 = rel_data[apogees[0]]
x0, y0, z0 = p0["x"], p0["y"], p0["z"]
vx0, vy0, vz0 = p0["vx"], p0["vy"], p0["vz"]

p1 = rel_data[perigees[0]]
x1, y1, z1 = p1["x"], p1["y"], p1["z"]
vx1, vy1, vz1 = p1["vx"], p1["vy"], p1["vz"]

v0 = math.sqrt(vx0**2 + vy0**2 + vz0**2)
v1 = math.sqrt(vx1**2 + vy1**2 + vz1**2)

r0 = math.sqrt(x0**2 + y0**2 + z0**2)
r1 = math.sqrt(x1**2 + y1**2 + z1**2)

mu = (v0**2 - v1**2)/(2*((1/r0)-(1/r1)))

epsilon = (v0**2)/2-(mu/r0)

a = -mu/(2*epsilon)

T = 2*math.pi*math.sqrt(a**3/mu)

M = mu/G

mu_num = mu

p3 = rel_data[apogees[1]]
x0, y0, z0 = p3["x"], p3["y"], p3["z"]
vx0, vy0, vz0 = p3["vx"], p3["vy"], p3["vz"]

a_num_list.append(a)
T_num_list.append(T)
epsilon_num_list.append(epsilon)

a_real_list.append(a)
T_real_list.append(T)
epsilon_real_list.append(epsilon)

for i in range(apogees[0], apogees[2] + 1):
    p = rel_data[i]
    x, y, z = [p["x"], p["y"], p["z"]]

    filename = "trajectory.csv"

    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(["satellite", "x", "y", "z"])

        writer.writerow([name, x, y, z])

# ==============================
# IDEAL SYSTEM
# ==============================

def derivatives(state, mu_num):
    x, y, z, vx, vy, vz = state
    r = math.sqrt(x**2 + y**2 + z**2)
    ax = -mu_num * x / r**3
    ay = -mu_num * y / r**3
    az = -mu_num * z / r**3
    return np.array([vx, vy, vz, ax, ay, az])


def rk4_step(state, dt, mu_num):
    k1 = derivatives(state, mu_num)
    k2 = derivatives(state + 0.5 * dt * k1, mu_num)
    k3 = derivatives(state + 0.5 * dt * k2, mu_num)
    k4 = derivatives(state + dt * k3, mu_num)

    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

state = np.array([x0, y0, z0, vx0, vy0, vz0])

trajectory_num = []

for orbit_index in range(orbit_periods):

    state_prev = state.copy()
    state = rk4_step(state, dt, mu_num)
    state_next = rk4_step(state, dt, mu_num)

    x0_i, y0_i, z0_i, vx0_i, vy0_i, vz0_i = state_prev
    x1_i, y1_i, z1_i, vx1_i, vy1_i, vz1_i = state
    x2_i, y2_i, z2_i, vx2_i, vy2_i, vz2_i = state_next

    r0 = math.sqrt(x0_i**2 + y0_i**2 + z0_i**2)
    r1 = math.sqrt(x1_i**2 + y1_i**2 + z1_i**2)
    r2 = math.sqrt(x2_i**2 + y2_i**2 + z2_i**2)

    perigee_found = False
    apogee_found = False

    while not (perigee_found and apogee_found):

        r0, r1 = r1, r2
        state_prev = state.copy()
        state = rk4_step(state, dt, mu_num)
        x, y, z, vx, vy, vz = state
        r2 = math.sqrt(x**2 + y**2 + z**2)

        trajectory_num.append([x, y, z])

        if not perigee_found and r1 < r0 and r1 < r2:
            perigee_r = r1
            perigee_v = math.sqrt(vx**2 + vy**2 + vz**2)
            perigee_found = True

        if not apogee_found and r1 > r0 and r1 > r2:
            apogee_r = r1
            apogee_v = math.sqrt(vx**2 + vy**2 + vz**2)
            apogee_found = True

    mu_num = (apogee_v**2 - perigee_v**2) / (2 * ((1/apogee_r) - (1/perigee_r)))
    epsilon_num = (apogee_v**2)/2 - mu_num/apogee_r
    a_num = -mu_num/(2*epsilon_num)
    T_num = 2 * math.pi * math.sqrt(a_num**3 / mu_num)

    a_num_list.append(a_num)
    T_num_list.append(T_num)
    epsilon_num_list.append(epsilon_num)

    mu_num = mu

# ==============================
# REAL SYSTEM
# ==============================

for orbit in range(len(epsilon_num_list)-1):

    p4 = rel_data[apogees[orbit+1]]
    x0, y0, z0 = p4["x"], p4["y"], p4["z"]
    vx0, vy0, vz0 = p4["vx"], p4["vy"], p4["vz"]

    p5 = rel_data[perigees[orbit+1]]
    x1, y1, z1 = p5["x"], p5["y"], p5["z"]
    vx1, vy1, vz1 = p5["vx"], p5["vy"], p5["vz"]

    v0 = math.sqrt(vx0**2 + vy0**2 + vz0**2)
    v1 = math.sqrt(vx1**2 + vy1**2 + vz1**2)

    r0 = math.sqrt(x0**2 + y0**2 + z0**2)
    r1 = math.sqrt(x1**2 + y1**2 + z1**2)

    mu_real = (v0**2 - v1**2)/(2*((1/r0)-(1/r1)))
    epsilon_real = (v0**2)/2-(mu_real/r0)
    a_real = -mu_real/(2*epsilon_real)
    T_real = 2*math.pi*math.sqrt(a_real**3/mu_real)
    MJ_real = mu_real/G

    a_real_list.append(a_real)
    T_real_list.append(T_real)
    epsilon_real_list.append(epsilon_real)

orbits = np.arange(len(a_num_list))

# ==============================
# EXPORT
# ==============================

def export_orbit_data_to_csv(
    csv_filename,
    satellite_id,
    a_real_list,
    T_real_list,
    epsilon_real_list,
    a_num_list,
    T_num_list,
    epsilon_num_list
):
    
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "satellite_id",
                "a_real", "T_real", "epsilon_real",
                "a_num",  "T_num",  "epsilon_num"
            ])

        n = orbit_periods
        for i in range(n):
            writer.writerow([
                satellite_id,
                a_real_list[i],
                T_real_list[i],
                epsilon_real_list[i],
                a_num_list[i],
                T_num_list[i],
                epsilon_num_list[i],
            ])

export_orbit_data_to_csv(
    csv_filename= "parametrs.csv",
    satellite_id=name,
    a_real_list=a_real_list,
    T_real_list=T_real_list,
    epsilon_real_list=epsilon_real_list,
    a_num_list=a_num_list,
    T_num_list=T_num_list,
    epsilon_num_list=epsilon_num_list
)

# ==============================
# VISUALISATION
# ==============================

plt.figure(figsize=(10, 6))
plt.plot(orbits, epsilon_num_list,
         label="Ideal system",
         linewidth=2,
         color="#15D9D9")

plt.plot(orbits, epsilon_real_list,
         label="Real system",
         linewidth=2,
         color= "#5015D9")

plt.axhline(epsilon,
            linestyle="--",
            label="Initial specific energy",
            linewidth=1.5,
            color="#FF3700FF")

plt.xlabel("Orbit number")
plt.ylabel("Specific orbital energy (J/kg)")
plt.title("Specific orbital energy vs orbit number")
plt.legend()
plt.grid()
plt.show()

a_num_cubed = np.array(a_num_list)**3
a_real_cubed = np.array(a_real_list)**3

T_num_squared = np.array(T_num_list)**2
T_real_squared = np.array(T_real_list)**2


plt.figure(figsize=(10, 6))

plt.scatter(a_real_cubed, T_real_squared,
            label="Real system",
            s=40,
            color="#8400FFFF")

plt.scatter(a_num_cubed, T_num_squared,
            label="Ideal system",
            s=40,
            color="#FF005DFF")

plt.xlabel("Semi-major axis³ (m³)")
plt.ylabel("Period² (s²)")
plt.title("Verification of Kepler's Third Law")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
