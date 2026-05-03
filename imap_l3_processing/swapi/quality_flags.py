from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SwapiL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    SWP_SW_ANGLES_ESTIMATED = 2 ** 2
    BAD_FIT = 2 ** 3
    PUI_FIT_MISSING_UNCERTAINTY = 2 ** 4
    STALE_PROTON = 2 ** 5
    ALPHA_MAG_DATA_FALLBACK = 2 ** 6
