from enum import Enum

from imap_l3_processing.constants import (
    ALPHA_MASS_PER_CHARGE_M_P_PER_E,
    ALPHA_PARTICLE_MASS_KG,
    HE_PUI_PARTICLE_MASS_KG,
    HE_PUI_PARTICLE_MASS_PER_CHARGE_M_P_PER_E,
    PROTON_MASS_KG,
    PROTON_MASS_PER_CHARGE_M_P_PER_E,
)


class Species(Enum):
    mass_kg: float
    mass_per_charge_m_p_per_e: float

    PROTON = (PROTON_MASS_KG, PROTON_MASS_PER_CHARGE_M_P_PER_E)
    ALPHA = (ALPHA_PARTICLE_MASS_KG, ALPHA_MASS_PER_CHARGE_M_P_PER_E)
    HELIUM_PLUS = (
        HE_PUI_PARTICLE_MASS_KG,
        HE_PUI_PARTICLE_MASS_PER_CHARGE_M_P_PER_E,
    )

    def __init__(
        self,
        mass_kg: float,
        mass_per_charge_m_p_per_e: float,
    ) -> None:
        self.mass_kg = mass_kg
        self.mass_per_charge_m_p_per_e = mass_per_charge_m_p_per_e
