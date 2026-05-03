from imap_processing.quality_flags import FlagNameMixin, CommonFlags


class SwapiL3Flags(FlagNameMixin):
    NONE = CommonFlags.NONE
    SWP_SW_ANGLES_ESTIMATED = 2 ** 2
    BAD_FIT = 2 ** 3
    PUI_FIT_MISSING_UNCERTAINTY = 2 ** 4
    STALE_PROTON = 2 ** 5
    # Set on every alpha-sw chunk in a run when MAG L1D was used as the source
    # (L2 was unavailable). See issue #13 / #70. Indicates the MAG inputs are
    # preliminary and the run is a candidate for reprocessing once L2 is available.
    PRELIMINARY_MAG = 2 ** 6
