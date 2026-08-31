import unittest
from datetime import datetime, timedelta
from unittest.mock import sentinel, Mock, MagicMock, patch

import numpy as np

from imap_l3_processing.glows.l3e.glows_l3e_call_arguments import GlowsL3eCallArguments, GlowsL3eSpacecraftInfo
from imap_l3_processing.glows.l3e.glows_l3e_hi_model import GlowsL3EHiData
from imap_l3_processing.models import DataProductVariable
from tests.test_helpers import get_test_instrument_team_data_path


class TestL3eHiModel(unittest.TestCase):
    def test_l3e_hi_model_to_data_product_variables(self):

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
                l3e_hi: GlowsL3EHiData = GlowsL3EHiData(
                    Mock(),
                    sentinel.epoch,
                    sentinel.epoch_delta,
                    energy_array,
                    sentinel.energy_delta_plus,
                    sentinel.energy_delta_minus,
                    spin_angle_array,
                    sentinel.probability_of_survival,
                    sentinel.spin_axis_latitude,
                    sentinel.spin_axis_longitude,
                    sentinel.program_version,
                    sentinel.spacecraft_radius,
                    sentinel.spacecraft_latitude,
                    sentinel.spacecraft_longitude,
                    sentinel.spacecraft_velocity_x,
                    sentinel.spacecraft_velocity_y,
                    sentinel.spacecraft_velocity_z,
                    sentinel.elongation,
                    sentinel.glows_flags,
                )

                data_products = l3e_hi.to_data_product_variables()

                expected_data_products = [
                    DataProductVariable("epoch", sentinel.epoch),
                    DataProductVariable("epoch_delta", sentinel.epoch_delta),
                    DataProductVariable("energy_grid", energy_array),
                    DataProductVariable("energy_delta_plus", sentinel.energy_delta_plus),
                    DataProductVariable("energy_delta_minus", sentinel.energy_delta_minus),
                    DataProductVariable("spin_angle", spin_angle_array),
                    DataProductVariable("surv_prob", sentinel.probability_of_survival),
                    DataProductVariable("energy_label", expected_energy_labels),
                    DataProductVariable("spin_angle_label", expected_spin_angle_labels),
                    DataProductVariable("spin_axis_latitude", sentinel.spin_axis_latitude),
                    DataProductVariable("spin_axis_longitude", sentinel.spin_axis_longitude),
                    DataProductVariable("program_version", sentinel.program_version),
                    DataProductVariable("spacecraft_radius", sentinel.spacecraft_radius),
                    DataProductVariable("spacecraft_latitude", sentinel.spacecraft_latitude),
                    DataProductVariable("spacecraft_longitude", sentinel.spacecraft_longitude),
                    DataProductVariable("spacecraft_velocity_x", sentinel.spacecraft_velocity_x),
                    DataProductVariable("spacecraft_velocity_y", sentinel.spacecraft_velocity_y),
                    DataProductVariable("spacecraft_velocity_z", sentinel.spacecraft_velocity_z),
                    DataProductVariable("elongation", sentinel.elongation),
                    DataProductVariable("glows_flags", sentinel.glows_flags),
                ]

                self.assertEqual(expected_data_products, data_products)

    @patch('imap_l3_processing.glows.l3e.glows_l3e_hi_model.calculate_energy_deltas')
    def test_convert_dat_to_glows_l3e_hi_product(self, mock_calculate_energy_deltas):
        hi_file_path = get_test_instrument_team_data_path("glows/probSur.Imap.Hi_2009.000_90.00.txt")
        expected_epoch = datetime(year=2009, month=8, day=16)
        epoch_delta = timedelta(hours=8)
        expected_epoch_delta_in_nanoseconds = 8*3600*1e9
        expected_energy = [0.3749896542976582, 0.4881381387720609, 0.6354277772546533, 0.8271602401781013,
                           1.076745599456691, 1.401640250140305, 1.824567838312671, 2.375108588863466,
                           3.091768193234094, 4.024670958420553, 5.239065580337116, 6.819888740878518,
                           8.877705713882047, 11.55644347537280, 15.04345718406192, 19.58263409767632]

        mock_calculate_energy_deltas.return_value = (sentinel.expected_energy_delta_plus, sentinel.expected_energy_delta_minus)

        expected_spin_angle = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                               47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
                               69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                               91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                               110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
                               128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
                               146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                               164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                               182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
                               200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217,
                               218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
                               236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253,
                               254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
                               272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289,
                               290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
                               308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325,
                               326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343,
             344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360], dtype=np.float64)

        expected_survival_probability_index_8 = [
            0.8589063329371907, 0.8587603216362375, 0.8585838125331493, 0.8583938671767950, 0.8581975494622297,
            0.8579940886919561, 0.8577658336186133, 0.8575214109067385, 0.8572599641959735, 0.8569812266143973,
            0.8566932795409066, 0.8563886260499640, 0.8560556586637402, 0.8557098877131397, 0.8553438616744717,
            0.8549578606822365, 0.8545510333975665, 0.8541212480707067, 0.8536681161158036, 0.8531976189241352,
            0.8526922669152210, 0.8521695594671225, 0.8516090453648694, 0.8510264168109199, 0.8504115338966615,
            0.8497787446956853, 0.8491284768090441, 0.8484307529868772, 0.8477197767662112, 0.8469592190599952,
            0.8461609726558839, 0.8453310589628533, 0.8444869719908985, 0.8436110574283868, 0.8426901005437170,
            0.8417444019589297, 0.8407741696303948, 0.8397702695719742, 0.8387397217249644, 0.8376368056061494,
            0.8365059181016049, 0.8353506712749569, 0.8341197401096629, 0.8328807386203723, 0.8316151236695234,
            0.8302970047784992, 0.8289519503384518, 0.8275561150432003, 0.8261200965047648, 0.8246070354411250,
            0.8230668102146183, 0.8214856128640838, 0.8198277050221140, 0.8181845942358368, 0.8164978928432362,
            0.8148309289431280, 0.8131205906925642, 0.8113809258453120, 0.8096509045157287, 0.8079093484999044,
            0.8061612273184158, 0.8044016098272827, 0.8026764914872416, 0.8009798800464115, 0.7992804912990893,
            0.7975942906910145, 0.7959642742484756, 0.7942911491765003, 0.7926979440533350, 0.7911359974877107,
            0.7895681692802368, 0.7880552693505788, 0.7865820633788240, 0.7851433182389695, 0.7838179439870212,
            0.7825923824434343, 0.7815656413769618, 0.7808147672716648, 0.7804437518138578, 0.7803130526774916,
            0.7805042213127741, 0.7809074064230767, 0.7813841009090794, 0.7819760778000449, 0.7827500216229394,
            0.7838103093482217, 0.7852133370825548, 0.7868617457922777, 0.7887631522950826, 0.7908012747902750,
            0.7928705592892216, 0.7949576084461400, 0.7970472829205981, 0.7991794478975991, 0.8013141332087443,
            0.8034770881143130, 0.8056414449314634, 0.8078210292202376, 0.8099514597952638, 0.8120972717505870,
            0.8141922936155517, 0.8162799495859382, 0.8183661997086748, 0.8204529303103028, 0.8225442185915449,
            0.8246255008053819, 0.8266809204723378, 0.8286882267316884, 0.8305683903957821, 0.8324033176411660,
            0.8341171708051789, 0.8358134217032417, 0.8374520711185758, 0.8390655387793522, 0.8406882235809617,
            0.8422533842982793, 0.8437133249342935, 0.8451236245850865, 0.8464408939197524, 0.8476841636780257,
            0.8488764128329585, 0.8500031695041402, 0.8511294318653039, 0.8522065671879705, 0.8532635205533413,
            0.8542850871415247, 0.8552602056521883, 0.8562005567721789, 0.8570951569076538, 0.8579399483931489,
            0.8587430520841162, 0.8595486843278016, 0.8603074584511685, 0.8610408869701144, 0.8617638553578701,
            0.8624663050361265, 0.8631193864034373, 0.8637317777523140, 0.8643221475784872, 0.8648832636500384,
            0.8654137915804599, 0.8659313206452271, 0.8664381867080074, 0.8669254461854356, 0.8673860980091474,
            0.8678333557993616, 0.8682417934320824, 0.8686624857971838, 0.8690300296220324, 0.8693965768813886,
            0.8697486392127265, 0.8700837888261288, 0.8704044462925420, 0.8707110890085822, 0.8710108193918249,
            0.8712891065199926, 0.8715586093104385, 0.8718059308003806, 0.8720371638380639, 0.8722689042395361,
            0.8724713677077062, 0.8726762347681811, 0.8728679808089833, 0.8730318743361172, 0.8731905734470720,
            0.8733435122176896, 0.8734819340852058, 0.8735987818461556, 0.8737025069718742, 0.8738009111413396,
            0.8738940497203151, 0.8739674128793775, 0.8740283659839079, 0.8740767944974310, 0.8741198280035212,
            0.8741419468495819, 0.8741503018130508, 0.8741524268548572, 0.8741406243503601, 0.8741196786892612,
            0.8740871319460586, 0.8740299021694693, 0.8739595970117827, 0.8738911754598945, 0.8738085706855024,
            0.8737121351399008, 0.8735851165165418, 0.8734681027336583, 0.8733205433096817, 0.8731680229742780,
            0.8730005251022243, 0.8728193547081913, 0.8726222591204129, 0.8724091878053493, 0.8721881990724981,
            0.8719404959958943, 0.8716726564365842, 0.8713956164937880, 0.8710970955594298, 0.8707809341848330,
            0.8704483245506168, 0.8700869953326894, 0.8697240184707551, 0.8693228177240478, 0.8689058188918802,
            0.8684725481212315, 0.8679864074740811, 0.8674846479170634, 0.8669565386658245, 0.8663987931741293,
            0.8658292976477142, 0.8652400888199558, 0.8646159344170304, 0.8639645675796087, 0.8632894095541491,
            0.8625775450692091, 0.8618508940390988, 0.8610662587207843, 0.8602420947979763, 0.8593876584994568,
            0.8585135191949332, 0.8576052938566121, 0.8566647318199154, 0.8557040506174665, 0.8546950723577439,
            0.8536288094574640, 0.8525140899403636, 0.8513132917711719, 0.8500547007246933, 0.8487031312551817,
            0.8473557396340265, 0.8459512894425343, 0.8445566831823717, 0.8431045795291137, 0.8416159404817778,
            0.8400524806016000, 0.8384689895980637, 0.8367684445784317, 0.8350331558210311, 0.8332845367962914,
            0.8315062693986432, 0.8297374174385194, 0.8279583012690772, 0.8261865084237222, 0.8243942252989166,
            0.8225734619575098, 0.8207549333753910, 0.8189216746886322, 0.8170647071344279, 0.8152032410605699,
            0.8133635963914283, 0.8115323792261186, 0.8097108939706253, 0.8079113822965517, 0.8061213846923494,
            0.8043408988724510, 0.8025426088683201, 0.8007814521258226, 0.7990987049136656, 0.7976307399916195,
            0.7964515215938459, 0.7955921535640210, 0.7949442384447885, 0.7944722659231427, 0.7940835322381860,
            0.7937574117930607, 0.7935319301733967, 0.7935410410847828, 0.7938430647730597, 0.7944549685472722,
            0.7953201271372370, 0.7963353399742404, 0.7974666717653967, 0.7986284092599631, 0.7998758861147037,
            0.8011160184065348, 0.8023966457837027, 0.8037516415367600, 0.8050684075928743, 0.8064302119915984,
            0.8078016416648872, 0.8091911716432389, 0.8105695706967822, 0.8120156466367995, 0.8134179619097438,
            0.8148718795368090, 0.8163047327580164, 0.8177597708329662, 0.8192293148111823, 0.8207074925156855,
            0.8222059167731761, 0.8236269008644003, 0.8250637062532796, 0.8264471756096386, 0.8278214795110156,
            0.8291718933569523, 0.8305216689363951, 0.8318736481229983, 0.8331525651945888, 0.8344452634309963,
            0.8356818249509860, 0.8368599136784342, 0.8379673523433857, 0.8390344990656603, 0.8400886238827425,
            0.8411180665764300, 0.8421142442355391, 0.8431022785867203, 0.8440249657050137, 0.8449432095397274,
            0.8458194063508257, 0.8466739192575265, 0.8474814859179764, 0.8482437338193161, 0.8489939765878994,
            0.8497227079261302, 0.8503990404728226, 0.8510604321333799, 0.8516904214587106, 0.8522976961031688,
            0.8528845012461620, 0.8534308899160006, 0.8539713965861790, 0.8544590509167357, 0.8549124471041260,
            0.8553675524645131, 0.8557728466303356, 0.8561569562069312, 0.8565541426261458, 0.8568977169319288,
            0.8572347416962208, 0.8575440542473425, 0.8578267049055189, 0.8580766194405365, 0.8583205882562980,
            0.8585270608521915, 0.8587229325896764, 0.8589085011920993, 0.8590676557001768, 0.8592175194277411,
            0.8593425048092776, 0.8594494587601790, 0.8595467856580804, 0.8596372067196015, 0.8596791152243492,
            0.8597382969327882, 0.8597823055545957, 0.8597949778619707, 0.8598094662503897, 0.8598097898645332,
            0.8597880908859125, 0.8597611602425049, 0.8597126790735709, 0.8596589797562888, 0.8595919584381491,
            0.8595034812547143, 0.8594095601838275, 0.8593104306912213, 0.8591896577901775, 0.8590583978048111
        ]
        expected_survival_probability_shape = (1, 16, 360)

        mock_metadata = Mock()

        spin_axis_lat = 45.0
        spin_axis_lon = 90.0

        args = MagicMock(spec=GlowsL3eCallArguments)
        
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

        args.elongation = 135.0

        expected_program_version = 'Hi.v00.01'

        l3e_hi_product: GlowsL3EHiData = GlowsL3EHiData.convert_dat_to_glows_l3e_hi_product(mock_metadata, hi_file_path,
                                                                                            expected_epoch,
                                                                                            epoch_delta,
                                                                                            args)

        np.testing.assert_equal(np.array([expected_epoch]), l3e_hi_product.epoch, strict=True)
        np.testing.assert_equal(np.array([expected_epoch_delta_in_nanoseconds]), l3e_hi_product.epoch_delta, strict=True)
        np.testing.assert_equal(l3e_hi_product.energy, expected_energy, strict=True)
        np.testing.assert_equal(l3e_hi_product.spin_angle, expected_spin_angle, strict=True)
        np.testing.assert_equal(l3e_hi_product.probability_of_survival.shape, expected_survival_probability_shape,
                                strict=True)
        np.testing.assert_equal(l3e_hi_product.probability_of_survival[0][7],
                                expected_survival_probability_index_8, strict=True)

        np.testing.assert_equal(np.array([spin_axis_lat]), l3e_hi_product.spin_axis_lat, strict=True)
        np.testing.assert_equal(np.array([spin_axis_lon]), l3e_hi_product.spin_axis_lon, strict=True)
        np.testing.assert_equal(np.array([expected_program_version]), l3e_hi_product.program_version, strict=True)

        np.testing.assert_equal(np.array([.5]), l3e_hi_product.spacecraft_radius, strict=True)
        np.testing.assert_equal(np.array([85.4]), l3e_hi_product.spacecraft_longitude, strict=True)
        np.testing.assert_equal(np.array([45.1]), l3e_hi_product.spacecraft_latitude, strict=True)

        np.testing.assert_equal(np.array([2.1]), l3e_hi_product.spacecraft_velocity_x, strict=True)
        np.testing.assert_equal(np.array([2.2]), l3e_hi_product.spacecraft_velocity_y, strict=True)
        np.testing.assert_equal(np.array([2.3]), l3e_hi_product.spacecraft_velocity_z, strict=True)

        np.testing.assert_equal([135.0], l3e_hi_product.elongation)
        np.testing.assert_equal(sentinel.expected_energy_delta_plus, l3e_hi_product.energy_delta_plus)
        np.testing.assert_equal(sentinel.expected_energy_delta_minus, l3e_hi_product.energy_delta_minus)

        mock_calculate_energy_deltas.assert_called_once_with(l3e_hi_product.energy)
