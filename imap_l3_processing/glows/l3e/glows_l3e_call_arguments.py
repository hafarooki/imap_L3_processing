from dataclasses import dataclass


@dataclass
class GlowsL3eSpacecraftInfo:
    spacecraft_radius: float
    spacecraft_longitude: float
    spacecraft_latitude: float
    spacecraft_velocity_x: float
    spacecraft_velocity_y: float
    spacecraft_velocity_z: float
    spin_axis_longitude: float
    spin_axis_latitude: float

    def to_argument_list(self):
        return [
            str(self.spacecraft_radius),
            f"{self.spacecraft_longitude:.4f}",
            f"{self.spacecraft_latitude:.4f}",
            f"{self.spacecraft_velocity_x:.4f}",
            f"{self.spacecraft_velocity_y:.4f}",
            f"{self.spacecraft_velocity_z}",
            f"{self.spin_axis_longitude:.4f}",
            f"{self.spin_axis_latitude:.4f}",
        ]

@dataclass
class GlowsL3eCallArguments:
    formatted_date: str
    decimal_date: str
    spacecraft_info: GlowsL3eSpacecraftInfo
    elongation: float

    def to_argument_list(self):
        return [
            self.formatted_date,
            self.decimal_date,
            *self.spacecraft_info.to_argument_list(),
            f"{self.elongation:.3f}"
        ]