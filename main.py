from drone import Drone


def main():
    batteries = [0.1, 0.25, 0.5, 0.75, 1.0]   # kg (battery mass)
    flight_time = 600                          # s (10 min)

    print(f"{'m_payload (kg)':>13} {'P_aero (W)':>12} {'P_lin (W)':>12} {'E_lin (J)':>12}")
    for m in batteries:
        d = Drone(payload_mass=m)
        p_aero = d.power_aero()
        p_lin = d.power_linear()
        e_lin = d.energy_consumption(flight_time, model="aero")
        # e_lin = d.power_aero() * flight_time

        print(f"{m:>10.2f} {p_aero:>12.2f} {p_lin:>12.2f} {e_lin:>12.2f}")


if __name__ == "__main__":
    main()
