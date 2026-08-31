import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel, Mock, MagicMock, patch

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo
from imap_l3_processing.glows.l3e.glows_l3e_lo_model import GlowsL3ELoData, SPIN_AXIS_LONGITUDE_VAR_NAME, \
    SPIN_AXIS_LATITUDE_VAR_NAME, EPOCH_CDF_VAR_NAME, ENERGY_VAR_NAME, SPIN_ANGLE_VAR_NAME, \
    PROBABILITY_OF_SURVIVAL_VAR_NAME, ENERGY_LABEL_VAR_NAME, SPIN_ANGLE_LABEL_VAR_NAME, ELONGATION_VAR_NAME, \
    PROGRAM_VERSION_VAR_NAME, SPACECRAFT_RADIUS_VAR_NAME, SPACECRAFT_LONGITUDE_VAR_NAME, SPACECRAFT_LATITUDE_VAR_NAME, \
    SPACECRAFT_VELOCITY_X_VAR_NAME, SPACECRAFT_VELOCITY_Y_VAR_NAME, SPACECRAFT_VELOCITY_Z_VAR_NAME, \
    GLOWS_FLAGS_VAR_NAME, ENERGY_DELTA_PLUS_VAR_NAME, ENERGY_DELTA_MINUS_VAR_NAME, EPOCH_DELTA_CDF_VAR_NAME
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import get_test_instrument_team_data_path, NumpyArrayMatcher


class TestL3eLoModel(unittest.TestCase):
    def test_l3e_lo_model_to_data_product_variables(self):
        test_cases = {
            'case 1':(
                [1.2234,2,4.51,66.7666],
                [1,41650233.0,4.22,9.5],
                ['1.22','2.00','4.51','66.77'],
                ['1','41650233','4','10']
            ),
            'case 2':(
                [4.536,48.193,4253.1],
                [13,14.0,34.2,19.5],
                ['4.54','48.19','4253.10'],
                ['13','14','34','20']
            )
        }

        for name, (energy_array, spin_angle_array, expected_energy_labels, expected_spin_angle_labels) in test_cases.items():
            with self.subTest(name):
                l3e_lo: GlowsL3ELoData = GlowsL3ELoData(
                        Mock(),
                        sentinel.epoch,
                        sentinel.epoch_delta,
                        energy_array,
                        sentinel.energy_delta_plus,
                        sentinel.energy_delta_minus,
                        spin_angle_array,
                        sentinel.probability_of_survival,
                        sentinel.elongation,
                        sentinel.spin_axis_latitude,
                        sentinel.spin_axis_longitude,
                        sentinel.program_version,
                        sentinel.spacecraft_radius,
                        sentinel.spacecraft_latitude,
                        sentinel.spacecraft_longitude,
                        sentinel.spacecraft_velocity_x,
                        sentinel.spacecraft_velocity_y,
                        sentinel.spacecraft_velocity_z,
                        sentinel.glows_flags,
                    )
                data_products = l3e_lo.to_data_product_variables()


                expected_data_products = [
                        DataProductVariable(EPOCH_CDF_VAR_NAME, sentinel.epoch),
                        DataProductVariable(EPOCH_DELTA_CDF_VAR_NAME, sentinel.epoch_delta),
                        DataProductVariable(ENERGY_VAR_NAME, energy_array),
                        DataProductVariable(ENERGY_DELTA_PLUS_VAR_NAME, sentinel.energy_delta_plus),
                        DataProductVariable(ENERGY_DELTA_MINUS_VAR_NAME, sentinel.energy_delta_minus),
                        DataProductVariable(SPIN_ANGLE_VAR_NAME, spin_angle_array),
                        DataProductVariable(PROBABILITY_OF_SURVIVAL_VAR_NAME, sentinel.probability_of_survival),
                        DataProductVariable(ENERGY_LABEL_VAR_NAME, expected_energy_labels),
                        DataProductVariable(SPIN_ANGLE_LABEL_VAR_NAME, expected_spin_angle_labels),
                        DataProductVariable(ELONGATION_VAR_NAME, sentinel.elongation),
                        DataProductVariable(SPIN_AXIS_LATITUDE_VAR_NAME, np.array([sentinel.spin_axis_latitude])),
                        DataProductVariable(SPIN_AXIS_LONGITUDE_VAR_NAME, np.array([sentinel.spin_axis_longitude])),
                        DataProductVariable(PROGRAM_VERSION_VAR_NAME, np.array([sentinel.program_version])),
                        DataProductVariable(SPACECRAFT_RADIUS_VAR_NAME, np.array([sentinel.spacecraft_radius])),
                        DataProductVariable(SPACECRAFT_LATITUDE_VAR_NAME, np.array([sentinel.spacecraft_latitude])),
                        DataProductVariable(SPACECRAFT_LONGITUDE_VAR_NAME, np.array([sentinel.spacecraft_longitude])),
                        DataProductVariable(SPACECRAFT_VELOCITY_X_VAR_NAME, np.array([sentinel.spacecraft_velocity_x])),
                        DataProductVariable(SPACECRAFT_VELOCITY_Y_VAR_NAME, np.array([sentinel.spacecraft_velocity_y])),
                        DataProductVariable(SPACECRAFT_VELOCITY_Z_VAR_NAME, np.array([sentinel.spacecraft_velocity_z])),
                        DataProductVariable(GLOWS_FLAGS_VAR_NAME, sentinel.glows_flags),
                ]
                self.assertEqual(expected_data_products, data_products)

    @patch("imap_l3_processing.glows.l3e.glows_l3e_lo_model.calculate_energy_deltas")
    def test_convert_dat_to_glows_l3e_lo_product(self, mock_calculate_energy_deltas):
        lo_file_path = get_test_instrument_team_data_path("glows/probSur.Imap.Lo_20090101_010101_2009.000_60.00.txt")
        epoch = datetime(year=2009, month=1, day=1)
        epoch_delta = timedelta(hours=3)
        expected_epoch_delta_in_nanoseconds = 3*3600*1e9
        expected_energy = [0.1700000, 0.2212954, 0.2880685, 0.3749897, 0.4881381, 0.6354278, 0.8271602, 1.0767456,
                           1.4016403, 1.8245678, 2.3751086, 3.0917682, 4.0246710]

        mock_calculate_energy_deltas.return_value = sentinel.energy_delta_plus, sentinel.energy_delta_minus

        expected_prob_of_survival_first_col_1 = [
            0.48543928E+00, 0.48415770E+00, 0.48286930E+00, 0.48155365E+00, 0.48022886E+00, 0.47885535E+00,
            0.47745419E+00, 0.47602527E+00, 0.47456880E+00, 0.47308723E+00, 0.47161934E+00, 0.47007995E+00,
            0.46855039E+00, 0.46696817E+00, 0.46535270E+00, 0.46368022E+00, 0.46201160E+00, 0.46026144E+00,
            0.45852743E+00, 0.45670696E+00, 0.45486823E+00, 0.45294031E+00, 0.45103086E+00, 0.44905075E+00,
            0.44699322E+00, 0.44493658E+00, 0.44274056E+00, 0.44049151E+00, 0.43813752E+00, 0.43578356E+00,
            0.43334287E+00, 0.43079981E+00, 0.42816069E+00, 0.42550222E+00, 0.42269926E+00, 0.41984889E+00,
            0.41688223E+00, 0.41387416E+00, 0.41075224E+00, 0.40761023E+00, 0.40432881E+00, 0.40096472E+00,
            0.39747450E+00, 0.39382089E+00, 0.39011010E+00, 0.38626253E+00, 0.38238974E+00, 0.37847099E+00,
            0.37447377E+00, 0.37034872E+00, 0.36611061E+00, 0.36168805E+00, 0.35708535E+00, 0.35234758E+00,
            0.34755946E+00, 0.34276461E+00, 0.33796280E+00, 0.33305954E+00, 0.32807347E+00, 0.32294017E+00,
            0.31786557E+00, 0.31276043E+00, 0.30773144E+00, 0.30281685E+00, 0.29797800E+00, 0.29321404E+00,
            0.28846079E+00, 0.28371704E+00, 0.27912920E+00, 0.27468323E+00, 0.27045568E+00, 0.26630037E+00,
            0.26237361E+00, 0.25859967E+00, 0.25499865E+00, 0.25157148E+00, 0.24843419E+00, 0.24562329E+00,
            0.24332850E+00, 0.24163423E+00, 0.24043665E+00, 0.23972507E+00, 0.23958627E+00, 0.23984571E+00,
            0.24038628E+00, 0.24120584E+00, 0.24238245E+00, 0.24409800E+00, 0.24644643E+00, 0.24933851E+00,
            0.25273734E+00, 0.25673467E+00, 0.26113737E+00, 0.26578085E+00, 0.27063217E+00, 0.27563385E+00,
            0.28065683E+00, 0.28568105E+00, 0.29082592E+00, 0.29609185E+00, 0.30147878E+00, 0.30707066E+00,
            0.31284106E+00, 0.31876025E+00, 0.32458878E+00, 0.32999671E+00, 0.33517199E+00, 0.34027166E+00,
            0.34522320E+00, 0.35023574E+00, 0.35525591E+00, 0.36033213E+00, 0.36532967E+00, 0.37024320E+00,
            0.37475598E+00, 0.37896325E+00, 0.38299882E+00, 0.38699896E+00, 0.39085028E+00, 0.39468049E+00,
            0.39842597E+00, 0.40207629E+00, 0.40552207E+00, 0.40883026E+00, 0.41207903E+00, 0.41518970E+00,
            0.41827498E+00, 0.42130951E+00, 0.42430284E+00, 0.42722657E+00, 0.43005367E+00, 0.43275769E+00,
            0.43536314E+00, 0.43791211E+00, 0.44040868E+00, 0.44286096E+00, 0.44523738E+00, 0.44760685E+00,
            0.44986863E+00, 0.45215006E+00, 0.45433928E+00, 0.45643657E+00, 0.45854771E+00, 0.46059383E+00,
            0.46257694E+00, 0.46460808E+00, 0.46651408E+00, 0.46842937E+00, 0.47031930E+00, 0.47214870E+00,
            0.47394193E+00, 0.47575376E+00, 0.47747515E+00, 0.47919249E+00, 0.48087054E+00, 0.48248460E+00,
            0.48411980E+00, 0.48572180E+00, 0.48729084E+00, 0.48882613E+00, 0.49029764E+00, 0.49173477E+00,
            0.49316337E+00, 0.49455725E+00, 0.49591591E+00, 0.49723950E+00, 0.49855476E+00, 0.49983531E+00,
            0.50105632E+00, 0.50224193E+00, 0.50338975E+00, 0.50452839E+00, 0.50568552E+00, 0.50673081E+00,
            0.50774070E+00, 0.50873875E+00, 0.50974595E+00, 0.51063701E+00, 0.51151521E+00, 0.51238110E+00,
            0.51321059E+00, 0.51397275E+00, 0.51472495E+00, 0.51540664E+00, 0.51610368E+00, 0.51667264E+00,
            0.51730154E+00, 0.51776070E+00, 0.51827571E+00, 0.51868642E+00, 0.51905088E+00, 0.51938042E+00,
            0.51966850E+00, 0.51988677E+00, 0.52010423E+00, 0.52020368E+00, 0.52028969E+00, 0.52027278E+00,
            0.52016485E+00, 0.52001351E+00, 0.51981105E+00, 0.51959358E+00, 0.51926552E+00, 0.51884032E+00,
            0.51847414E+00, 0.51792884E+00, 0.51731622E+00, 0.51669638E+00, 0.51593604E+00, 0.51513940E+00,
            0.51427166E+00, 0.51335442E+00, 0.51238909E+00, 0.51129305E+00, 0.51014602E+00, 0.50890185E+00,
            0.50758409E+00, 0.50615044E+00, 0.50464683E+00, 0.50296036E+00, 0.50118289E+00, 0.49930168E+00,
            0.49730277E+00, 0.49530416E+00, 0.49314260E+00, 0.49083236E+00, 0.48844397E+00, 0.48583154E+00,
            0.48323435E+00, 0.48058649E+00, 0.47780153E+00, 0.47506716E+00, 0.47209482E+00, 0.46907585E+00,
            0.46597907E+00, 0.46291516E+00, 0.45981653E+00, 0.45665363E+00, 0.45351121E+00, 0.45044243E+00,
            0.44730392E+00, 0.44413627E+00, 0.44106236E+00, 0.43794686E+00, 0.43485646E+00, 0.43175941E+00,
            0.42864794E+00, 0.42564675E+00, 0.42266427E+00, 0.41976102E+00, 0.41687658E+00, 0.41406969E+00,
            0.41130938E+00, 0.40859441E+00, 0.40588626E+00, 0.40343902E+00, 0.40125003E+00, 0.39935862E+00,
            0.39784753E+00, 0.39670361E+00, 0.39585956E+00, 0.39523582E+00, 0.39476103E+00, 0.39439905E+00,
            0.39423968E+00, 0.39435100E+00, 0.39473203E+00, 0.39554310E+00, 0.39666286E+00, 0.39808096E+00,
            0.39972118E+00, 0.40153794E+00, 0.40352131E+00, 0.40548763E+00, 0.40751478E+00, 0.40963578E+00,
            0.41174409E+00, 0.41388333E+00, 0.41612846E+00, 0.41834220E+00, 0.42056149E+00, 0.42288866E+00,
            0.42519516E+00, 0.42763343E+00, 0.43005765E+00, 0.43251298E+00, 0.43502684E+00, 0.43754215E+00,
            0.44011421E+00, 0.44269314E+00, 0.44537654E+00, 0.44802922E+00, 0.45079502E+00, 0.45340240E+00,
            0.45589495E+00, 0.45836848E+00, 0.46074029E+00, 0.46320577E+00, 0.46557171E+00, 0.46804433E+00,
            0.47046801E+00, 0.47291016E+00, 0.47527772E+00, 0.47750468E+00, 0.47952019E+00, 0.48141970E+00,
            0.48317952E+00, 0.48487642E+00, 0.48650926E+00, 0.48807554E+00, 0.48963920E+00, 0.49111387E+00,
            0.49259059E+00, 0.49402593E+00, 0.49539224E+00, 0.49660220E+00, 0.49767488E+00, 0.49868284E+00,
            0.49962355E+00, 0.50044291E+00, 0.50121762E+00, 0.50196273E+00, 0.50261243E+00, 0.50326871E+00,
            0.50387423E+00, 0.50434609E+00, 0.50480509E+00, 0.50515270E+00, 0.50543081E+00, 0.50566424E+00,
            0.50580062E+00, 0.50590890E+00, 0.50589201E+00, 0.50588645E+00, 0.50578122E+00, 0.50561852E+00,
            0.50539304E+00, 0.50510737E+00, 0.50474719E+00, 0.50435977E+00, 0.50384726E+00, 0.50336489E+00,
            0.50279180E+00, 0.50216504E+00, 0.50148265E+00, 0.50078978E+00, 0.50000804E+00, 0.49918459E+00,
            0.49836367E+00, 0.49744147E+00, 0.49651714E+00, 0.49557195E+00, 0.49456744E+00, 0.49353034E+00,
            0.49245761E+00, 0.49135232E+00, 0.49023631E+00, 0.48909069E+00, 0.48791635E+00, 0.48669295E+00
        ]

        expected_spin_angle = np.arange(1, 361, 1, dtype=np.float64)
        elongation_value = 75
        expected_survival_probability_shape = (1, 13, 360)

        mock_metadata = Mock()

        spin_axis_lat = 45.0
        spin_axis_lon = 90.0

        args = MagicMock(spec=GlowsL3eCallArguments)
        expected_program_version = 'Lo.v00.01'

        spacecraft_info = MagicMock(spec=GlowsL3eSpacecraftInfo)
        spacecraft_info.spin_axis_latitude = spin_axis_lat
        spacecraft_info.spin_axis_longitude = spin_axis_lon
        spacecraft_info.spacecraft_radius = .5
        spacecraft_info.spacecraft_longitude = 85.4
        spacecraft_info.spacecraft_latitude = 45.1

        spacecraft_info.spacecraft_velocity_x = 2.1
        spacecraft_info.spacecraft_velocity_y = 2.2
        spacecraft_info.spacecraft_velocity_z = 2.3

        args.spacecraft_info = spacecraft_info

        l3e_lo_product: GlowsL3ELoData = GlowsL3ELoData.convert_dat_to_glows_l3e_lo_product(mock_metadata, lo_file_path,
                                                                                            epoch,
                                                                                            epoch_delta,
                                                                                            elongation_value,
                                                                                            args)

        mock_calculate_energy_deltas.assert_called_once_with(NumpyArrayMatcher(l3e_lo_product.energy))

        np.testing.assert_equal([epoch], l3e_lo_product.epoch, strict=True)
        np.testing.assert_equal([expected_epoch_delta_in_nanoseconds], l3e_lo_product.epoch_delta, strict=True)
        np.testing.assert_equal(l3e_lo_product.energy, expected_energy, strict=True)
        np.testing.assert_equal(l3e_lo_product.energy_delta_plus, sentinel.energy_delta_plus)
        np.testing.assert_equal(l3e_lo_product.energy_delta_minus, sentinel.energy_delta_minus,)

        np.testing.assert_equal(l3e_lo_product.spin_angle, expected_spin_angle, strict=True)
        np.testing.assert_equal(l3e_lo_product.probability_of_survival.shape, expected_survival_probability_shape,
                                strict=True)
        np.testing.assert_equal(l3e_lo_product.probability_of_survival[0][0],
                                expected_prob_of_survival_first_col_1, strict=True)
        np.testing.assert_equal(l3e_lo_product.elongation, np.array([elongation_value]), strict=True)

        np.testing.assert_equal(np.array([spin_axis_lat]), l3e_lo_product.spin_axis_lat, strict=True)
        np.testing.assert_equal(np.array([spin_axis_lon]), l3e_lo_product.spin_axis_lon, strict=True)

        np.testing.assert_equal([expected_program_version], l3e_lo_product.program_version, strict=True)

        np.testing.assert_equal(l3e_lo_product.spacecraft_radius, np.array([.5]), strict=True)
        np.testing.assert_equal(l3e_lo_product.spacecraft_longitude, np.array([85.4]), strict=True)
        np.testing.assert_equal(l3e_lo_product.spacecraft_latitude, np.array([45.1]), strict=True)

        np.testing.assert_equal(l3e_lo_product.spacecraft_velocity_x, np.array([2.1]), strict=True)
        np.testing.assert_equal(l3e_lo_product.spacecraft_velocity_y, np.array([2.2]), strict=True)
        np.testing.assert_equal(l3e_lo_product.spacecraft_velocity_z, np.array([2.3]), strict=True)

