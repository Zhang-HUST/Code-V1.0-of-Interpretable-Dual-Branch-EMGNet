from utils.tools import is_string_in_list
from utils.tools import data_normalize
from utils.feature_extraction_tools.feature_extraction_utils import *

"""Feature extraction of emg data with multiple samples and multiple channels"""
fs_global = 1000


def emg_feature_extraction(emg_sample, emg_feature_type, fea_normalize_method, fea_normalize_level):
    """
    1. 支持的emg特征列表
        1.1. 15个时域特征['方差VAR', '均方根值RMS', '肌电积分值IEMG', '绝对值均值MAV', '对数探测器LOG', '波形长度WL', '平均振幅变化AAC','差值绝对标准差值DASDV',
                        '过零率ZC', 'Willison幅值WAMP', '脉冲百分率MYOP', 斜率符号变化SSC, 简单平方积分SSI, 峭度因子KF, 第三时间矩TM3]
        1.2. 9个频域特征['频数比FR', '平均功率MNP', '总功率TOP', '平均频率MNF', '中值频率MDF', '峰值频率PKF', '谱矩1SM1', '谱矩2SM2', '谱矩3SM3']
        1.3. 1个时频域特征['小波能量WENT']
        1.4. 3个信息熵特征['近似熵AE', '样本熵SE’, '模糊熵FE']
    """
    all_emg_feas = []
    for i in range(emg_sample.shape[0]):
        temp1 = []
        for j in range(emg_sample.shape[2]):
            sub_emg_data = emg_sample[i, :, j]
            sub_emg_feas = emg_feature_extraction_alone(sub_emg_data, emg_feature_type)
            temp1.extend(np.array(sub_emg_feas))
        all_emg_feas.append(temp1)
    all_emg_feas = np.array(all_emg_feas)  # shape, num*[len(emg_channels)*len(feature_type)]

    emg_feas_normalize = data_normalize(all_emg_feas, fea_normalize_method, fea_normalize_level)

    all_emg_feas_pre = np.reshape(emg_feas_normalize,
                                  (emg_feas_normalize.shape[0], -1))
    print('emg_feas.shape: ', all_emg_feas_pre.shape)
    return all_emg_feas_pre


def emg_feature_extraction_alone(x, feature_type):
    emg_feas = []
    th = np.mean(x) + 3 * np.std(x)
    ssc_threshold = 0.000001

    # 16 time-domain features
    if is_string_in_list(feature_type, 'VAR'):
        fea_var = np.var(x)
        emg_feas.append(fea_var)
    if is_string_in_list(feature_type, 'RMS'):
        fea_rms = np.sqrt(np.mean(x ** 2))
        emg_feas.append(fea_rms)
    if is_string_in_list(feature_type, 'IEMG'):
        fea_iemg = np.sum(abs(x))
        emg_feas.append(fea_iemg)
    if is_string_in_list(feature_type, 'MAV'):
        fea_mav = np.sum(np.absolute(x)) / len(x)
        emg_feas.append(fea_mav)
    if is_string_in_list(feature_type, 'LOG'):
        fea_log = np.exp(np.sum(np.log10(np.absolute(x))) / len(x))
        emg_feas.append(fea_log)
    if is_string_in_list(feature_type, 'WL'):
        fea_wl = np.sum(abs(np.diff(x)))
        emg_feas.append(fea_wl)
    if is_string_in_list(feature_type, 'AAC'):
        fea_aac = np.sum(abs(np.diff(x))) / len(x)
        emg_feas.append(fea_aac)
    if is_string_in_list(feature_type, 'DASDV'):
        fea_dasdv = math.sqrt((1 / (len(x) - 1)) * np.sum((np.diff(x)) ** 2))
        emg_feas.append(fea_dasdv)
    if is_string_in_list(feature_type, 'ZC'):
        fea_zc = get_emg_feature_zc(x, th)
        emg_feas.append(fea_zc)
    if is_string_in_list(feature_type, 'WAMP'):
        fea_wamp = get_emg_feature_wamp(x, th)
        emg_feas.append(fea_wamp)
    if is_string_in_list(feature_type, 'MYOP'):
        fea_myop = get_emg_feature_myop(x, th)
        emg_feas.append(fea_myop)
    if is_string_in_list(feature_type, 'SSC'):
        fea_ssc = get_emg_feature_ssc(x, threshold=ssc_threshold)
        emg_feas.append(fea_ssc)
    if is_string_in_list(feature_type, 'SSI'):
        fea_ssi = get_emg_feature_ssi(x)
        emg_feas.append(fea_ssi)
    if is_string_in_list(feature_type, 'KF'):
        fea_kf = get_emg_feature_kf(x)
        emg_feas.append(fea_kf)
    if is_string_in_list(feature_type, 'TM3'):
        fea_tm3 = get_emg_feature_tm3(x)
        emg_feas.append(fea_tm3)

    # 9 frequency-domain features
    frequency, power = get_signal_spectrum(x, fs_global)
    if is_string_in_list(feature_type, 'FR'):
        fea_fr = get_emg_feature_fr(frequency, power)  # Frequency ratio
        emg_feas.append(fea_fr)
    if is_string_in_list(feature_type, 'MNP'):
        fea_mnp = np.sum(power) / len(power)  # Mean power
        emg_feas.append(fea_mnp)
    if is_string_in_list(feature_type, 'TOP'):
        fea_top = np.sum(power)  # Total power
        emg_feas.append(fea_top)
    if is_string_in_list(feature_type, 'MNF'):
        fea_mnf = get_emg_feature_mnf(frequency, power)  # Mean frequency
        emg_feas.append(fea_mnf)
    if is_string_in_list(feature_type, 'MDF'):
        fea_mdf = get_emg_feature_mdf(frequency, power)  # Median frequency
        emg_feas.append(fea_mdf)
    if is_string_in_list(feature_type, 'PKF'):
        fea_pkf = frequency[power.argmax()]  # Peak frequency
        emg_feas.append(fea_pkf)
    if is_string_in_list(feature_type, 'SM1'):
        fea_sm1 = get_emg_feature_sm1(x, fs_global)  # Spectral Moment 1
        emg_feas.append(fea_sm1)
    if is_string_in_list(feature_type, 'SM2'):
        fea_sm2 = get_emg_feature_sm2(x, fs_global)  # Spectral Moment 2
        emg_feas.append(fea_sm2)
    if is_string_in_list(feature_type, 'SM3'):
        fea_sm3 = get_emg_feature_sm3(x, fs_global)  # Spectral Moment 3
        emg_feas.append(fea_sm3)

    # 1 time-frequency domain feature
    if is_string_in_list(feature_type, 'WENT'):
        fea_went = get_emg_feature_went(x)  # Wavelet energy
        emg_feas.append(fea_went)

    # 3 information entropy features
    if is_string_in_list(feature_type, 'AE'):
        fea_ae = get_emg_feature_AE(x, m=3, r=0.15)  # Approximate entropy
        emg_feas.append(fea_ae)
    if is_string_in_list(feature_type, 'SE'):
        fea_se = get_emg_feature_SE(x, m=3, r=0.15)  # Sample entropy
        emg_feas.append(fea_se)
    if is_string_in_list(feature_type, 'FE'):
        fea_fe = get_emg_feature_FE(x, m=3, r=0.15, n=2)  # Fuzzy entropy
        emg_feas.append(fea_fe)

    return emg_feas
