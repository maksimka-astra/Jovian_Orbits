import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

name = "Io"
dt = 3600 # 60 min
G = 6.67430e-20  # km^3 / (kg s^2)
orbit_periods = 100

# ==============================
# ARRAYS
# ==============================

a_real_list = []
T_real_list = []
MJ_real_list = []
epsilon_real_list = []

a_num_list = []
T_num_list = []
MJ_num_list = []
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

import numpy as np
import math

def find_perigees_and_apogees(rel_data):
    """
    Находит индексы перигеев и апогеев из массива rel_data.
    rel_data — список словарей с ключами: x, y, z, vx, vy, vz
    Возвращает два списка: perigees, apogees (индексы в rel_data)
    """
    # координаты и скорости
    x = np.array([p["x"] for p in rel_data])
    y = np.array([p["y"] for p in rel_data])
    z = np.array([p["z"] for p in rel_data])
    vx = np.array([p["vx"] for p in rel_data])
    vy = np.array([p["vy"] for p in rel_data])
    vz = np.array([p["vz"] for p in rel_data])

    # расстояние от центра
    r = np.sqrt(x**2 + y**2 + z**2)

    # радиальная скорость
    vr = (x*vx + y*vy + z*vz) / r

    # апогеи: vr меняется с + → −
    apogees = np.where((vr[:-1] > 0) & (vr[1:] <= 0))[0] + 1

    # перигеи: vr меняется с − → +
    perigees = np.where((vr[:-1] < 0) & (vr[1:] >= 0))[0] + 1

    return perigees.tolist(), apogees.tolist()


if __name__ == "__main__":
    satellite_csv = name+".csv"
    central_body_csv = "jupiter.csv"
    output_csv = "relative.csv"

    # rel_data должен возвращать список словарей с x,y,z,vx,vy,vz
    rel_data = relative_orbit(satellite_csv, central_body_csv, output_csv)

    # находим перигеи и апогеи
    perigees, apogees = find_perigees_and_apogees(rel_data)

    print(f"Найдено перигеев: {len(perigees)}, апогеев: {len(apogees)}")


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
MJ_num_list.append(M)
epsilon_num_list.append(epsilon)

a_real_list.append(a)
T_real_list.append(T)
MJ_real_list.append(M)
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

print(trajectory)

# ==============================
# IDEAL SYSTEM
# ==============================

state = np.array([x0, y0, z0, vx0, vy0, vz0])

def derivatives(state, mu_num):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    
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

print("Expected orbits:", orbit_periods)
print("Detected apogees:", len(apogees))

for orbit in range(orbit_periods):

    r0 = 0
    r1 = 0
    r2 = 0

    r0 = math.sqrt(x0**2 + y0**2 + z0**2)

    apogee1 = r0
    apogee2 = 0
    perigee1 = 0
    apogee_v = 0
    perigee_v = 0

    state = rk4_step(state, dt, mu_num)
    x, y, z, vx, vy, vz = state
    r1 = math.sqrt(x**2 + y**2 + z**2)

    state = rk4_step(state, dt, mu_num)
    x, y, z, vx, vy, vz = state
    r2 = math.sqrt(x**2 + y**2 + z**2)

    step_count = 0
    max_steps = 20000 

    while perigee1 == 0 or apogee2 == 0:
        r0 = r1
        r1 = r2

        step_count += 1

        state = rk4_step(state, dt, mu_num)
        x, y, z, vx, vy, vz = state
        r2 = math.sqrt(x**2 + y**2 + z**2)

        DR_THRESHOLD = 0

        if perigee1 == 0 and r1 < r0 - DR_THRESHOLD and r1 < r2 - DR_THRESHOLD:
            perigee1 = r1
            perigee_v = math.sqrt(vx**2 + vy**2 + vz**2)

        if apogee2 == 0 and r1 > r0 + DR_THRESHOLD and r1 > r2 + DR_THRESHOLD:
            apogee2 = r1
            apogee_v = math.sqrt(vx**2 + vy**2 + vz**2)

        if step_count % 100 == 0:
            print(f"Moon {name}: step {step_count}, r={r1:.2f}")

    mu_num = (apogee_v**2 - perigee_v**2) / (2 * ((1 / apogee2) - (1 / perigee1)))
    epsilon_num = (apogee_v**2) / 2 - (mu_num / apogee2)
    a_num = -mu_num / (2 * epsilon_num)
    T_num = 2 * math.pi * math.sqrt(a_num**3 / mu_num)
    MJ = mu_num / G

    a_num_list.append(a_num)
    T_num_list.append(T_num)
    MJ_num_list.append(MJ)
    epsilon_num_list.append(epsilon_num)

    mu_num = mu

# ==============================
# REAL SYSTEM
# ==============================

for orbit in range(orbit_periods):

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
    MJ_real_list.append(MJ_real)
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
    MJ_real_list,
    epsilon_real_list,
    a_num_list,
    T_num_list,
    MJ_num_list,
    epsilon_num_list
):
    
    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "satellite_id",
                "a_real", "T_real", "MJ_real", "epsilon_real",
                "a_num",  "T_num",  "MJ_num",  "epsilon_num"
            ])

        n = orbit_periods
        for i in range(n):
            writer.writerow([
                satellite_id,
                a_real_list[i],
                T_real_list[i],
                MJ_real_list[i],
                epsilon_real_list[i],
                a_num_list[i],
                T_num_list[i],
                MJ_num_list[i],
                epsilon_num_list[i],
            ])

export_orbit_data_to_csv(
    csv_filename="sattelites.csv",
    satellite_id= name,
    a_real_list=a_real_list,
    T_real_list=T_real_list,
    MJ_real_list=MJ_real_list,
    epsilon_real_list=epsilon_real_list,
    a_num_list=a_num_list,
    T_num_list=T_num_list,
    MJ_num_list=MJ_num_list,
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
