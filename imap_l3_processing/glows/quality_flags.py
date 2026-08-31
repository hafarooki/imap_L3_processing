from imap_processing.quality_flags import CommonFlags, FlagNameMixin


NOMINAL_ALPHA_PROTON_RATIO_VALUE: float = 0.04


class GlowsL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    NOMINAL_ALPHA_PROTON_RATIO = 2 ** 2
    PERSISTED_LAST_POINT = 2 ** 3
    PREDICTIVE_EPHEMERIS = 2 ** 15
