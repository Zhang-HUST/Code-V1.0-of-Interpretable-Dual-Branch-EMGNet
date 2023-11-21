import math
import pywt
import numpy as np
from scipy.fft import fft


ssc_threshold_global = 0.000001
fs_global = 1000


def get_emg_feature_wamp(signal, th):
    x = abs(np.diff(signal))
    umbral = x >= th

    return np.sum(umbral)


def get_emg_feature_myop(signal, th):
    umbral = signal >= th

    return np.sum(umbral) / len(signal)


def get_emg_feature_zc(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1

    return cruce


def get_emg_feature_ssc(window_data, threshold=ssc_threshold_global):
    # Slope Sign Change
    if threshold == 'None':
        # delta = np.abs(list(map(lambda first, medium, last: (medium - first) * (medium - last), window_data[0:-2], window_data[1:-1], window_data[2:])))
        # print('ssc max:', np.max(delta), '  ssc min:', np.min(delta))
        threshold = 0.0005
    data_ssc = 0
    for i in range(np.size(window_data, 0) - 2):
        delta_first = window_data[i + 1] - window_data[i]
        delta_last = window_data[i] - window_data[i + 2]
        if delta_first * delta_last > threshold:
            data_ssc = data_ssc + 1

    return data_ssc


def get_emg_feature_kf(window_data):
    # Kurtosis Factor
    D = np.mean(list(map(lambda x: math.pow(x, 2), window_data - np.mean(window_data))))
    data_kf = np.mean(list(map(lambda x: math.pow(x, 4), window_data - np.mean(window_data)))) / D - 3

    return data_kf


def get_emg_feature_ssi(window_data):
    # Simple Square Integrate
    return np.sum(list(map(lambda x: math.pow(x, 2), window_data)))


def get_emg_feature_tm3(window_data):
    # 3rd Temporal Moment
    return np.abs(np.mean(list(map(lambda x: math.pow(x, 3), window_data))))


def get_signal_spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def med_freq(f, P):
    plot = np.sum(P) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0
    while abs(errel) > tol:
        temp += P[i]
        errel = (plot - temp) / plot
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return f[i]


def get_emg_feature_fr(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC


def get_emg_feature_mnf(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den


def get_emg_feature_mdf(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0
    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]


def get_emg_feature_sm1(signal, sampling_rate):
    """SM1（Spectral Moment 1）：SM1 衡量了肌电信号频谱能量的分布。它是通过计算频谱的一阶矩（平均值）来确定的。
    SM1 反映了肌电信号能量在频谱中的分布情况，通常与肌肉疲劳程度有关。较高的 SM1 值可能表示由于肌肉疲劳导致的频谱能量向低频区域集中。"""
    # 频率域转换
    spectrum = fft(signal)
    frequency = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # 计算功率谱密度 PSD
    power_spectrum_density = np.abs(spectrum) ** 2
    # 计算 SM1
    SM1 = np.sum(frequency * power_spectrum_density) / np.sum(power_spectrum_density)

    return SM1


def get_emg_feature_sm2(signal, sampling_rate):
    """SM2（Spectral Moment 2）：SM2 是通过计算频谱的二阶矩（方差）来获得的。它可以描述肌电信号频谱的离散程度。
    较高的 SM2 值可能表示肌电信号频谱能量在频域上更分散，而较低的 SM2 值则可能表示频谱能量更加集中。"""
    # 频率域转换
    spectrum = fft(signal)
    frequency = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # 计算功率谱密度 PSD
    power_spectrum_density = np.abs(spectrum) ** 2
    # 计算 SM1
    SM1 = np.sum(frequency * power_spectrum_density) / np.sum(power_spectrum_density)
    # 计算 SM2
    deviation = frequency - SM1
    SM2 = np.sqrt(np.sum(power_spectrum_density * deviation ** 2) / np.sum(power_spectrum_density))

    return SM2


def get_emg_feature_sm3(signal, sampling_rate):
    """SM3（Spectral Moment 3）：SM3 是通过计算频谱的三阶矩来得到的。它用于描述肌电信号频谱的偏度（Skewness）。
    SM3 可以反映信号频谱的偏斜性，对于鉴别不同类型的肌肉活动和疲劳状态具有一定的意义。"""
    # 频率域转换
    spectrum = fft(signal)
    frequency = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # 计算功率谱密度 PSD
    power_spectrum_density = np.abs(spectrum) ** 2
    # 计算 SM1
    SM1 = np.sum(frequency * power_spectrum_density) / np.sum(power_spectrum_density)
    # 计算 SM2
    deviation = frequency - SM1
    SM2 = np.sqrt(np.sum(power_spectrum_density * deviation ** 2) / np.sum(power_spectrum_density))
    # 计算 SM3
    SM3 = np.sum(power_spectrum_density * deviation ** 3) / (np.sum(power_spectrum_density) * SM2 ** 3)

    return SM3


def wavelet_energy(x, mother, nivel):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    et = np.sum(arr ** 2)
    ca = coeffs[0]
    ea = 100 * np.sum(ca ** 2) / et
    ed = []
    for k in range(1, len(coeffs)):
        cd = list(coeffs[k].values())
        cd = np.asarray(cd)
        ed.append(100 * np.sum(cd ** 2) / et)

    return ea, ed


def get_emg_feature_went(signal):
    h_wavelet = []
    E_a, E = wavelet_energy(signal, 'db7', 4)
    E.insert(0, E_a)
    E = np.asarray(E) / 100
    h_wavelet.append(-np.sum(E * np.log2(E)))

    return h_wavelet


def shannon(x):
    n = len(x)
    nb = 19
    hist, bin_edges = np.histogram(x, bins=nb)
    counts = hist / n
    nz = np.nonzero(counts)

    return np.sum(counts[nz] * np.log(counts[nz]) / np.log(2))


def get_emg_feature_AE(x, m=3, r=0.15):
    """
    近似熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m + 1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x) - m + 1 - temp):
            X.append(x[i:i + m + temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for i in X:
            sub = []
            for j in X:
                sub.append(max(np.abs(i - j)))
            D_value.append(sub)
        # 计算阈值
        F = r * np.std(x, ddof=1)
        # 判断D_value中的每一行中的值比阈值小的个数除以len(x)-m+1的比例
        num = np.sum(D_value < F, axis=1) / (len(x) - m + 1 - temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num))
        entropy = abs(entropy) - Lm

    return entropy


def get_emg_feature_SE(U, m=3, r=0.15):
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(B)

    N = len(U)

    return (-np.log(_phi(m + 1) / _phi(m)))


def get_emg_feature_FE(x, m=3, r=0.15, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m + 1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x) - m + 1 - temp):
            X.append(x[i:i + m + temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i - j)))
            D_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(D_value, n) / r)
        # 计算所有隶属度的平均值
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm

    return entropy
