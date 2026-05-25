# Eq. (2): P = (W+m)^{3/2}\sqrt{g^3 / (2\rho\varsigma n)}
# Eq. (3): p(m) = \alpha m + \beta
# Fitted constants (p. 73): α = 46.7 W/kg, β = 26.9 W; n = 6, ρ = 1.204 kg/m³, ς = 0.2 m², W = 1.5 kg.

import math


class Drone:
    ALPHA = 46.7        # W/kg
    BETA = 26.9         # W
    N_ROTORS = 6
    RHO = 1.204         # kg/m^3
    VARSIGMA = 0.2      # m^2
    W_EMPTY = 1.5       # kg
    G = 9.81            # m/s^2

    def __init__(
        self,
        payload_mass: float = 0.0,
        *,
        alpha: float = ALPHA,
        beta: float = BETA,
        n_rotors: int = N_ROTORS,
        rho: float = RHO,
        varsigma: float = VARSIGMA,
        w_empty: float = W_EMPTY,
        g: float = G,
        battery_mass: float | None = None,
    ):
        # allow caller to pass battery_mass keyword (compatibility with existing code)
        if battery_mass is not None:
            payload_mass = battery_mass

        self.payload_mass = payload_mass
        self.alpha = alpha
        self.beta = beta
        self.n_rotors = n_rotors
        self.rho = rho
        self.varsigma = varsigma
        self.w_empty = w_empty
        self.g = g

# P = (W+m)^{3/2}\sqrt{g^3 / (2\rho\varsigma n)}
    def power_aero(self, mass: float | None = None) -> float:
        m = self.payload_mass if mass is None else mass
        total_mass = self.w_empty + m
        return (total_mass ** 1.5) * math.sqrt(
            self.g ** 3 / (2 * self.rho * self.varsigma * self.n_rotors)
        )

#  p(m) = \alpha m + \beta
    def power_linear(self, mass: float | None = None) -> float:
        m = self.payload_mass if mass is None else mass
        return self.alpha * m + self.beta

    def energy_consumption(self, flight_time_s: float, model: str = "linear") -> float:
        p = self.power_linear() if model == "linear" else self.power_aero()
        return p * flight_time_s

    def __repr__(self) -> str:
        return (
            f"Drone(payload_mass={self.payload_mass} kg, "
            f"w_empty={self.w_empty} kg, n={self.n_rotors}, "
            f"alpha={self.alpha}, beta={self.beta})"
        )
