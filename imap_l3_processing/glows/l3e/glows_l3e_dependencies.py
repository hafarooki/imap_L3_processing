import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import imap_data_access
from imap_data_access.processing_input import ProcessingInputCollection, RepointInput

from imap_l3_processing.utils import furnish_spice_metakernel, SpiceKernelTypes

logger = logging.getLogger(__name__)

GLOWS_L3E_REQUIRED_SPICE_KERNELS: list[SpiceKernelTypes] = [
    SpiceKernelTypes.ScienceFrames,
    SpiceKernelTypes.EphemerisPredicted,
    SpiceKernelTypes.EphemerisReconstructed,
    SpiceKernelTypes.AttitudeHistory,
    SpiceKernelTypes.PointingAttitude,
    SpiceKernelTypes.PlanetaryEphemeris,
    SpiceKernelTypes.Leapseconds,
    SpiceKernelTypes.SpacecraftClock,
]

@dataclass
class GlowsL3EDependencies:
    energy_grid_lo: Path
    energy_grid_hi: Path
    energy_grid_ultra: Path
    tess_xyz_8: Path
    tess_ang16: Path
    lya_series: Path
    solar_uv_anisotropy: Path
    speed_3d_sw: Path
    density_3d_sw: Path
    phion_hydrogen: Path
    sw_eqtr_electrons: Path
    pipeline_settings: dict
    pipeline_settings_file: Path
    repointing_file: Path

    @classmethod
    def fetch_dependencies(cls, dependencies: ProcessingInputCollection):
        lya_series_dependency = dependencies.get_file_paths(source='glows', descriptor='lya')
        solar_uv_anisotropy_dependency = dependencies.get_file_paths(source='glows', descriptor='uv-anis')
        speed_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='speed')
        density_3d_dependency = dependencies.get_file_paths(source='glows', descriptor='p-dens')
        phion_hydrogen_dependency = dependencies.get_file_paths(source='glows', descriptor='phion')
        sw_eqtr_electrons_dependency = dependencies.get_file_paths(source='glows', descriptor='e-dens')

        pipeline_settings_dependency = dependencies.get_file_paths(source='glows', descriptor='pipeline-settings-l3bcde')

        tess_xyz_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-xyz-8')
        tess_ang_dependency = dependencies.get_file_paths(source='glows', descriptor='tess-ang-16')

        energy_grid_lo_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-lo')
        energy_grid_hi_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-hi')
        energy_grid_ultra_dependency = dependencies.get_file_paths(source='glows', descriptor='energy-grid-ultra')

        lya_series_path = imap_data_access.download(lya_series_dependency[0])
        solar_uv_anisotropy_path = imap_data_access.download(solar_uv_anisotropy_dependency[0])
        speed_3d_path = imap_data_access.download(speed_3d_dependency[0])
        density_3d_path = imap_data_access.download(density_3d_dependency[0])
        phion_hydrogen_path = imap_data_access.download(phion_hydrogen_dependency[0])
        sw_eqtr_electrons_path = imap_data_access.download(sw_eqtr_electrons_dependency[0])

        pipeline_settings_path = imap_data_access.download(pipeline_settings_dependency[0])

        tess_xyz_path = imap_data_access.download(tess_xyz_dependency[0])
        tess_ang_path = imap_data_access.download(tess_ang_dependency[0])

        energy_grid_lo_path = imap_data_access.download(energy_grid_lo_dependency[0])
        energy_grid_hi_path = imap_data_access.download(energy_grid_hi_dependency[0])
        energy_grid_ultra_path = imap_data_access.download(energy_grid_ultra_dependency[0])

        with open(pipeline_settings_path) as f:
            pipeline_settings = json.load(f)

        repoint_file_dependency = dependencies.get_file_paths(data_type=RepointInput.data_type)
        repoint_file_path = imap_data_access.download(repoint_file_dependency[0])

        return cls(
            energy_grid_lo=energy_grid_lo_path,
            energy_grid_hi=energy_grid_hi_path,
            energy_grid_ultra=energy_grid_ultra_path,
            tess_xyz_8=tess_xyz_path,
            tess_ang16=tess_ang_path,
            lya_series=lya_series_path,
            solar_uv_anisotropy=solar_uv_anisotropy_path,
            speed_3d_sw=speed_3d_path,
            density_3d_sw=density_3d_path,
            phion_hydrogen=phion_hydrogen_path,
            sw_eqtr_electrons=sw_eqtr_electrons_path,
            pipeline_settings=pipeline_settings,
            pipeline_settings_file=pipeline_settings_path,
            repointing_file=repoint_file_path,
        )

    @staticmethod
    def collect_spice_dependencies(start_date: datetime, end_date: datetime):
        logger.info(f"Querying for SPICE data over the range: {start_date} to {end_date}")

        kernel_types_with_predicted = GLOWS_L3E_REQUIRED_SPICE_KERNELS
        kernel_types_without_predicted = [kernel for kernel in GLOWS_L3E_REQUIRED_SPICE_KERNELS if kernel != SpiceKernelTypes.EphemerisPredicted]

        return (furnish_spice_metakernel(start_date=start_date, end_date=end_date, kernel_types=kernel_types_with_predicted, metakernel_file_name="metakernel_with_predict_ephem.txt"),
            furnish_spice_metakernel(start_date=start_date, end_date=end_date, kernel_types=kernel_types_without_predicted, metakernel_file_name="metakernel_without_predict_ephem.txt"))

    def copy_dependencies(self):
        if self.energy_grid_lo is not None:
            shutil.copy(self.energy_grid_lo, self.pipeline_settings['executable_dependency_paths']['energy-grid-lo'])
        if self.energy_grid_hi is not None:
            shutil.copy(self.energy_grid_hi, self.pipeline_settings['executable_dependency_paths']['energy-grid-hi'])
        if self.energy_grid_ultra is not None:
            shutil.copy(self.energy_grid_ultra,
                        self.pipeline_settings['executable_dependency_paths']['energy-grid-ultra'])
        if self.tess_xyz_8 is not None:
            shutil.copy(self.tess_xyz_8, self.pipeline_settings['executable_dependency_paths']['tess-xyz-8'])
        if self.tess_ang16 is not None:
            shutil.copy(self.tess_ang16, self.pipeline_settings['executable_dependency_paths']['tess-ang-16'])
        shutil.copy(self.lya_series, self.lya_series.name)
        shutil.copy(self.solar_uv_anisotropy, self.solar_uv_anisotropy.name)
        shutil.copy(self.speed_3d_sw, self.speed_3d_sw.name)
        shutil.copy(self.density_3d_sw, self.density_3d_sw.name)
        shutil.copy(self.phion_hydrogen, self.phion_hydrogen.name)
        shutil.copy(self.sw_eqtr_electrons, self.sw_eqtr_electrons.name)

        with open("ionization.files.dat", 'w') as ionization_file:
            ionization_file.write("\n".join([
                self.lya_series.name,
                self.solar_uv_anisotropy.name,
                self.speed_3d_sw.name,
                self.density_3d_sw.name,
                self.phion_hydrogen.name,
                self.sw_eqtr_electrons.name,
            ]) + "\n")

    def get_hi_parents(self):
        return [
            self.energy_grid_hi.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
        ]

    def get_lo_parents(self):
        return [
            self.energy_grid_lo.name,
            self.tess_xyz_8.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
        ]

    def get_ul_parents(self):
        return [
            self.energy_grid_ultra.name,
            self.tess_ang16.name,
            self.lya_series.name,
            self.solar_uv_anisotropy.name,
            self.speed_3d_sw.name,
            self.density_3d_sw.name,
            self.phion_hydrogen.name,
            self.sw_eqtr_electrons.name,
            self.pipeline_settings_file.name,
            self.repointing_file.name,
        ]
