# Revised SWAPI ESA k-factor from high-resolution SIMION simulations (Q ≈ 1.89 eV/V at θ = 0).
# Used internally by L3 for passband normalization and central-speed conversions.
SWAPI_K_FACTOR = 1.89

# Pre-launch k-factor used by the L2 product to label its `esa_energy` field as
# `esa_energy = SWAPI_L2_K_FACTOR × |voltage|`. Different from SWAPI_K_FACTOR — divide L2's
# `esa_energy` by this to recover true ESA voltage before any L3 processing.
SWAPI_L2_K_FACTOR = 1.93

# SWAPI ESA sweep bin layout (72 bins total, indices 0–71):
#   Index 0       : always discarded (hardware artifact, never science data)
#   Indices 1–62  : coarse sweep passbands (62 bins, uniform energy steps)
#   Indices 63–71 : fine sweep passbands (9 bins, higher resolution near the proton peak)
SWAPI_DISCARDED_BIN = 0
SWAPI_COARSE_SWEEP_BINS = slice(1, 63)  # indices 1–62
SWAPI_FINE_SWEEP_BINS = slice(63, 72)  # indices 63–71
SWAPI_SCIENCE_BINS = slice(1, 72)  # indices 1–71, all usable bins (coarse + fine)

SWAPI_LIVETIME_S = 0.145

SWAPI_BIN_PERIOD_S = 12 / 72  # duration of one ESA step in the 12 s, 72-bin sweep
# Counts accumulate over the 0.145 s livetime that starts after the voltage ramp-up;
# sample geometry at the center of that window rather than the bin's start time.
SWAPI_LIVETIME_CENTER_OFFSET_S = (SWAPI_BIN_PERIOD_S - SWAPI_LIVETIME_S) + SWAPI_LIVETIME_S / 2

# baseline instrument background is approximately 0.1 Hz
SWAPI_BACKGROUND_RATE = 0.1

# assumes adiabatic cooling
SWAPI_PUI_COOLING_INDEX = 1.5
