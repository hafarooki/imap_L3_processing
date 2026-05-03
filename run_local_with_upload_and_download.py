import os
import shutil
import subprocess
import sys

from imap_l3_processing.glows.l3d.utils import PATH_TO_L3D_TOOLKIT

if __name__ == "__main__":
    target = sys.argv[1]
    subtarget = sys.argv[2] if len(sys.argv) > 2 else None
    match target, subtarget:
        case "hi", "combine-l2":
            version = "v004"
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "hi",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20250415",
                    "--version",
                    version,
                    "--descriptor",
                    "hic-ena-h-hf-nsp-full-hae-4deg-1yr",
                    "--dependency",
                    f"""[{{"type": "science", "files": ["imap_hi_l2_h45-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h45-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h90-ena-h-hf-nsp-anti-hae-4deg-1yr_20250415_{version}.cdf"]}},
            {{"type": "science", "files": ["imap_hi_l2_h90-ena-h-hf-nsp-ram-hae-4deg-1yr_20250415_{version}.cdf"]}}]""",
                ]
            )
        case "lo", "spx":
            version = "v902"
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "lo",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20260101",
                    "--version",
                    version,
                    "--descriptor",
                    "l090-spx-h-sf-sp-ram-hae-6deg-1yr",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_lo_l3_l090-ena-h-sf-sp-ram-hae-6deg-1yr_20260101_v001.cdf"]}]',
                ]
            )
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "lo",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20260101",
                    "--version",
                    version,
                    "--descriptor",
                    "l090-spx-h-sf-nsp-ram-hae-6deg-1yr",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_lo_l2_l090-ena-h-sf-nsp-ram-hae-6deg-1yr_20260101_v902.cdf"]}]',
                ]
            )
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "lo",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20260101",
                    "--version",
                    version,
                    "--descriptor",
                    "l090-spxnbs-h-sf-nsp-ram-hae-6deg-1yr",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_lo_l2_l090-enanbs-h-sf-nsp-ram-hae-6deg-1yr_20260101_v900.cdf"]}]',
                ]
            )
        case "hi", "spx":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "hi",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20250415",
                    "--version",
                    "v002",
                    "--descriptor",
                    "h90-spx-h-hf-sp-full-hae-4deg-6mo",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v002.cdf"]}]',
                ]
            )
        case "hi", "survival":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "hi",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20250415",
                    "--version",
                    "v002",
                    "--descriptor",
                    "h90-ena-h-sf-sp-full-hae-4deg-6mo",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-ram-hae-4deg-6mo_20250415_v001.cdf"]},\n{"type": "science", "files": ["imap_hi_l2_h90-ena-h-sf-nsp-anti-hae-4deg-6mo_20250415_v001.cdf"]}]',
                ]
            )
        case "hit", "direct":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "hit",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20100106",
                    "--version",
                    "v001",
                    "--descriptor",
                    "direct-events",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_hit_l1a_direct-events_20100106_v001.cdf"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-2A-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-2B-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-3A-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-3B-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-4A-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-4B-cosine-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_hi-gain-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_lo-gain-lookup_20250203_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_hit-event-type-lookup_20250228_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-2A-charge-fit-lookup_20250319_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-3A-charge-fit-lookup_20250319_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-4A-charge-fit-lookup_20250319_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-2B-charge-fit-lookup_20250319_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-3B-charge-fit-lookup_20250319_v000.csv"]},'
                    + '{"type": "ancillary", "files": ["imap_hit_range-4B-charge-fit-lookup_20250319_v000.csv"]}]',
                ]
            )
        case "hit", "macropixel":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "hit",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20100106",
                    "--version",
                    "v001",
                    "--descriptor",
                    "macropixel",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_hit_l2_macropixel-intensity_20100106_v001.cdf"]},'
                    + '{"type": "science", "files": ["imap_mag_l1d_norm-dsrf_20100106_v001.cdf"]}]',
                ]
            )
        case "ultra", "survival":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "ultra",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20250415",
                    "--version",
                    "v010",
                    "--descriptor",
                    "u90-ena-h-sf-sp-full-hae-4deg-3mo",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_ultra_l2_u90-ena-h-sf-nsp-full-hae-4deg-3mo_20250415_v010.cdf"]},'
                    + '{"type": "spice", "files": ["imap_sclk_0000.tsc"]},'
                    + '{"type": "spice", "files": ["naif0012.tls"]},'
                    + '{"type": "spice", "files": ["imap_science_0001.tf"]},'
                    + '{"type": "spice", "files": ["imap_dps_2025_105_2026_105_01.ah.bc"]}]',
                ]
            )
        case "ultra", "spx":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "ultra",
                    "--data-level",
                    "l3",
                    "--start-date",
                    "20250415",
                    "--version",
                    "v010",
                    "--descriptor",
                    "u90-spx-h-sf-sp-full-hae-4deg-3mo",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_ultra_l3_u90-ena-h-sf-sp-full-hae-4deg-3mo_20250415_v010.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_ultra_ulc-spx-energy-ranges_20250407_v000.dat"]}'
                    "]",
                ]
            )
        case "glows", "l3b":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "glows",
                    "--data-level",
                    "l3b",
                    "--start-date",
                    "20250415",
                    "--version",
                    "v025",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_glows_l3a_hist_20250415-repoint01000_v026.cdf"]},'
                    '{"type": "ancillary", "files":["imap_glows_bad-days-list_20100101_v001.dat"]},'
                    '{"type": "ancillary", "files":["imap_glows_WawHelioIonMP_20100101_v002.json"]},'
                    '{"type": "ancillary", "files":["imap_glows_uv-anisotropy-1CR_20100101_v001.json"]},'
                    '{"type": "ancillary", "files":["imap_glows_pipeline-settings-l3bcde_19470303_v008.json"]},'
                    '{"type": "repoint", "files":["imap_2026_105_01.repoint.csv"]}]',
                ]
            )
        case "glows", "l3d":
            if os.path.exists(PATH_TO_L3D_TOOLKIT / "data_l3b"):
                shutil.rmtree(PATH_TO_L3D_TOOLKIT / "data_l3b")
            if os.path.exists(PATH_TO_L3D_TOOLKIT / "data_l3c"):
                shutil.rmtree(PATH_TO_L3D_TOOLKIT / "data_l3c")
            if os.path.exists(PATH_TO_L3D_TOOLKIT / "data_l3d"):
                shutil.rmtree(PATH_TO_L3D_TOOLKIT / "data_l3d")
            if os.path.exists(PATH_TO_L3D_TOOLKIT / "data_l3d_txt"):
                shutil.rmtree(PATH_TO_L3D_TOOLKIT / "data_l3d_txt")

            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "glows",
                    "--data-level",
                    "l3d",
                    "--start-date",
                    "20100101",
                    "--version",
                    "v010",
                    "--descriptor",
                    "solar-hist",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_glows_l3b_ion-rate-profile_20100422_v013.cdf"]},'
                    '{"type": "science", "files": ["imap_glows_l3c_sw-profile_20100422_v012.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_glows_plasma-speed-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_proton-density-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_uv-anisotropy-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_photoion-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_electron-density-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_lya-2010a_20100101_v003.dat"]},'
                    '{"type": "ancillary", "files": ["imap_glows_l3b-archive_20100422_v011.zip"]},'
                    '{"type": "ancillary", "files": ["imap_glows_pipeline-settings-l3bcde_20100101_v006.json"]}]',
                ]
            )
        case "swapi", "proton":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swapi",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "proton-sw",
                    "--start-date",
                    "20250606",
                    "--version",
                    "v000",
                    "--dependency",
                    '[{"type":"science","files":["imap_swapi_l2_sci_20251208_v006.cdf"]},'
                    '{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]},'
                    '{"type": "spice", "files": ["imap_sclk_0000.tsc"]},'
                    + '{"type": "spice", "files": ["naif0012.tls"]},'
                    + '{"type": "spice", "files": ["imap_science_108.tf"]},'
                    + '{"type": "spice", "files": ["imap_001.tf"]},'
                    + '{"type": "spice", "files": ["imap_2025_105_2026_105_01.ah.bc"]},'
                    + '{"type": "spice", "files": ["imap_dps_2025_105_2026_105_01.ah.bc"]}'
                    "]",
                ]
            )
        case "swapi", "alpha":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swapi",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "alpha-sw",
                    "--start-date",
                    "20250606",
                    "--version",
                    "v000",
                    "--dependency",
                    '[{"type":"science","files":["imap_swapi_l2_sci_20250606_v001.cdf"]},'
                    '{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]}'
                    "]",
                ]
            )
        case "swapi", "pui":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swapi",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "pui-he",
                    "--start-date",
                    "20250606",
                    "--version",
                    "v000",
                    "--dependency",
                    '[{"type":"science","files":["imap_swapi_l2_sci_20250606_v001.cdf"]},'
                    '{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_density-of-neutral-helium-lut_20241023_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]},'
                    '{"type": "spice", "files": ["imap_sclk_0000.tsc"]},'
                    + '{"type": "spice", "files": ["naif0012.tls"]},'
                    + '{"type": "spice", "files": ["imap_science_0001.tf"]},'
                    + '{"type": "spice", "files": ["imap_001.tf"]},'
                    + '{"type": "spice", "files": ["imap_2025_105_2026_105_01.ah.bc"]},'
                    + '{"type": "spice", "files": ["de440.bsp"]},'
                    + '{"type": "spice", "files": ["imap_recon_20250415_20260415_v01.bsp"]},'
                    + '{"type": "spice", "files": ["imap_dps_2025_105_2026_105_01.ah.bc"]}'
                    "]",
                ]
            )
        case "swapi", "l3b":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "swapi",
                    "--data-level",
                    "l3b",
                    "--start-date",
                    "20250606",
                    "--version",
                    "v003",
                    "--dependency",
                    '[{"type": "science", "files": ["imap_swapi_l2_sci_20250606_v001.cdf"]},'
                    '{"type":"ancillary","files":["imap_swapi_efficiency-lut_20241020_v000.dat"]},'
                    '{"type":"ancillary","files":["imap_swapi_azimuthal-transmission_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_central-effective-area_20260425_v001.csv"]},'
                    '{"type":"ancillary","files":["imap_swapi_passband-fit-coefficients_20260425_v001.csv"]}]',
                ]
            )
        case "codice", "lo-direct-events":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-direct-events",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l1a_lo-sw-priority_20241110_v002.cdf"]},'
                    '{"type": "science", "files": ["imap_codice_l1a_lo-nsw-priority_20241110_v003.cdf"]},'
                    '{"type": "science", "files": ["imap_codice_l2_lo-direct-events_20241110_v006.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_codice_mass-coefficient-lookup_20241110_v002.csv"]},'
                    '{"type": "ancillary", "files": ["imap_codice_lo-energy-per-charge_20241110_v001.csv"]}]',
                ]
            )
        case "codice", ("lo-partial-densities" | "partial-densities"):
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-partial-densities",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l2_lo-sw-species_20241110_v002.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_codice_mass-per-charge_20241110_v002.csv"]}]',
                ]
            )
        case "codice", ("lo-sw-ratios" | "ratios"):
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-sw-ratios",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l3a_lo-partial-densities_20241110_v000.cdf"]}'
                    "]",
                ]
            )
        case "codice", ("lo-sw-charge-state-distributions"):
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "lo-sw-charge-state-distributions",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l3a_lo-partial-densities_20241110_v000.cdf"]}'
                    "]",
                ]
            )
        case "codice", "hi-direct-events":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    "hi-direct-events",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l2_hi-direct-events_20241110_v002.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_codice_tof-lookup_20241110_v002.csv"]}]',
                ]
            )
        case "codice", "hi-pitch-angle":
            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3b",
                    "--descriptor",
                    "hi-pitch-angle",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l2_hi-sectored_20241110_v002.cdf"]},'
                    '{"type": "science", "files": ["imap_mag_l1d_norm-dsrf_20250630_v001.cdf"]}'
                    "]",
                ]
            )
        case "codice", descriptor if "3d-distribution" in descriptor:
            species = descriptor.split("-")[1]
            assert species in ["hplus", "heplus", "heplus2", "oplus6"], (
                NotImplementedError("Target instrument or product not implemented")
            )

            subprocess.run(
                [
                    sys.executable,
                    "imap_l3_data_processor.py",
                    "--instrument",
                    "codice",
                    "--data-level",
                    "l3a",
                    "--descriptor",
                    f"lo-{species}-3d-distribution",
                    "--start-date",
                    "20241110",
                    "--version",
                    "v000",
                    "--dependency",
                    "["
                    '{"type": "science", "files": ["imap_codice_l1a_lo-sw-priority_20241110_v002.cdf"]},'
                    '{"type": "science", "files": ["imap_codice_l1a_lo-nsw-priority_20241110_v003.cdf"]},'
                    '{"type": "science", "files": ["imap_codice_l3a_lo-direct-events_20241110_v005.cdf"]},'
                    '{"type": "ancillary", "files": ["imap_codice_lo-mass-species-bin-lookup_20241110_v001.csv"]},'
                    '{"type": "ancillary", "files": ["imap_codice_lo-geometric-factors_20241110_v001.csv"]},'
                    '{"type": "ancillary", "files": ["imap_codice_lo-energy-per-charge_20241110_v001.csv"]},'
                    f'{{"type": "ancillary", "files": ["imap_codice_lo-{species}-efficiency_20241110_v001.csv"]}}'
                    "]",
                ]
            )
        case _:
            raise NotImplementedError("Target instrument or product not implemented")
