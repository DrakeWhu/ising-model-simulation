import math

from scipy import special


def critical_temperature(coupling: float = 1.0) -> float:
    """Return exact 2D square-lattice Ising critical temperature for h=0, k_B=1."""
    if coupling <= 0.0:
        raise ValueError("coupling must be positive for the ferromagnetic exact solution.")

    return 2.0 * coupling / math.log(1.0 + math.sqrt(2.0))


def exact_spontaneous_magnetization_density(
    temperature: float,
    coupling: float = 1.0,
) -> float:
    """Return exact spontaneous magnetization per spin for the infinite 2D square lattice.

    Valid for the ferromagnetic, isotropic, zero-field Ising model.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if coupling <= 0.0:
        raise ValueError("coupling must be positive for the ferromagnetic exact solution.")

    if temperature >= critical_temperature(coupling):
        return 0.0

    reduced_coupling = coupling / temperature
    sinh_2k = math.sinh(2.0 * reduced_coupling)

    return float((1.0 - sinh_2k**-4.0) ** 0.125)


def exact_energy_density(temperature: float, coupling: float = 1.0) -> float:
    """Return exact internal energy per spin for the infinite 2D square lattice.

    Valid for the ferromagnetic, isotropic, zero-field Ising model.

    scipy.special.ellipk uses the elliptic parameter m = k**2, not the modulus k.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if coupling <= 0.0:
        raise ValueError("coupling must be positive for the ferromagnetic exact solution.")

    tc = critical_temperature(coupling)
    if math.isclose(temperature, tc, rel_tol=0.0, abs_tol=1e-12 * coupling):
        return -coupling * math.sqrt(2.0)

    reduced_coupling = coupling / temperature
    two_k = 2.0 * reduced_coupling

    tanh_2k = math.tanh(two_k)
    coth_2k = 1.0 / tanh_2k

    sinh_2k = math.sinh(two_k)
    cosh_2k = math.cosh(two_k)
    elliptic_modulus = 2.0 * sinh_2k / cosh_2k**2

    elliptic_parameter = elliptic_modulus**2
    elliptic_integral = special.ellipk(elliptic_parameter)

    bracket = 1.0 + (2.0 / math.pi) * (2.0 * tanh_2k**2 - 1.0) * elliptic_integral

    return float(-coupling * coth_2k * bracket)
