import os
import xlrd
import numpy as np
from pykalman import KalmanFilter
import scipy.signal as signal


FREQUENCY_BANDS = [
  (0, 0.2),
  (0.2, 0.4),
  (0.4, 0.6),
  (0.6, 1)
]
EYE_FREQ = 120
N_CHANNELS = 1
STFT_N = 256


def get_PSD_DE_for_a_window(mag_fft_data, frequency_band):
    n_channels = mag_fft_data.shape[0]
    band_energy_bucket = np.zeros((n_channels, len(frequency_band)))
    band_frequency_count = np.zeros((n_channels, len(frequency_band)))
    for band_index, (f_start, f_end) in enumerate(frequency_band):
        # Map frequency to data point indices in mag_fft_data array
        fStartNum = np.floor(f_start / EYE_FREQ * STFT_N)
        fEndNum = np.floor(f_end / EYE_FREQ * STFT_N)
        for p in range(int(fStartNum - 1), int(fEndNum)):
            band_energy_bucket[:, band_index] += mag_fft_data[:, p] ** 2
            band_frequency_count[:, band_index] += 1
    window_band_PSD = band_energy_bucket / band_frequency_count  # Scale to uV
    # why times 100 !!!!!!
    window_band_DE = np.log2(100 * window_band_PSD)
    return window_band_PSD, window_band_DE


def get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_band):
    slice_data.shape=(1, slice_data.shape[0])
    window_points = window_size * EYE_FREQ
    step_points = (1 - overlap_rate) * window_points
    window_num = int(np.floor((slice_data.shape[1] - window_points) / step_points + 1))

    band_PSD = np.zeros((N_CHANNELS, len(frequency_band), window_num))
    band_DE = np.zeros_like(band_PSD)
    for window_index in range(window_num):
        data_win = slice_data[:, int(step_points * window_index):int(step_points*window_index+window_points)]
        hanning = signal.hann(window_points)
        han_data = data_win * hanning
        fft_data = np.fft.fft(han_data, n=STFT_N)
        mag_fft_data = np.abs(fft_data[:, 0:int(STFT_N / 2)])
        window_band_PSD, window_band_DE = get_PSD_DE_for_a_window(mag_fft_data)
        band_PSD[:, :, window_index] = window_band_PSD
        band_DE[:, :, window_index] = window_band_DE
    return band_PSD, band_DE


def smooth_features(feature_data, method='LDS'):
    """ Input:
        feature_data:
            (channel_N, frequency_band_N ,sample_point_N) feature array
        method:
            'LDS': Kalman Smooth (Linear Dynamic System)
            'moving_avg': Moving average method

        Output:
            Smoothed data which has the same shape as input
    """
    smoothed_data = np.zeros_like(feature_data)

    state_mean = np.mean(feature_data, axis=2)
    for channel_index in range(feature_data.shape[0]):
        for feature_band_index in range(feature_data.shape[1]):
            kf = KalmanFilter(transition_matrices=1,
                              observation_matrices=1,
                              transition_covariance=0.001,
                              observation_covariance=1,
                              initial_state_covariance=0.1,
                              initial_state_mean=state_mean[channel_index, feature_band_index])

            measurement = feature_data[channel_index, feature_band_index, :]
            smoothed_data[channel_index, feature_band_index, :] = kf.smooth(measurement)[0].flatten()
    return smoothed_data


def get_DE_PSD_smooth(slice_data, window_size, overlap_rate, frequency_bands):
    # Compute DE and PSD feature
    band_PSD, band_DE = get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_bands)

    # Smooth DE and PSD feature
    de_smooth_data = smooth_features(band_DE)
    psd_smooth_data = smooth_features(band_PSD)
    return de_smooth_data, psd_smooth_data


def cal_seconds(time_arr,t1,t2):
    #计算两个时间之间的秒数差
    sec1 = (int(time_arr[t1][0:2]))*3600+(int(time_arr[t1][3:5]))*60+(int(time_arr[t1][6:8]))
    sec2 = (int(time_arr[t2][0:2]))*3600+(int(time_arr[t2][3:5]))*60+(int(time_arr[t2][6:8]))
    if(sec1>sec2):
        sec2=sec2+24*3600
    return sec2-sec1


def cal_mseconds(time_arr,t1,t2):
    #计算两个时间的毫秒差
    sec1 = (int(time_arr[t1][0:2]))*3600+(int(time_arr[t1][3:5]))*60+(int(time_arr[t1][6:8]))
    sec2 = (int(time_arr[t2][0:2]))*3600+(int(time_arr[t2][3:5]))*60+(int(time_arr[t2][6:8]))
    if(sec1>sec2):
        sec2=sec2+24*3600
    sec_tmp = (int(time_arr[t2][9:12])-int(time_arr[t1][9:12]))
    return (sec2-sec1)*1000+sec_tmp


def interpolate(arr):
    # interpolation for items in arr with values -1
    len_a = len(arr)
    sta = -1
    sta_num = -1
    end = -1
    end_num = -1
    for i in range(len_a):
        if sta == -1 and arr[0] == -1:
            sta = 0
            sta_num = 0
            arr[0] = 0
            continue
        if sta == -1 and arr[i] == -1:
            sta = i-1
            sta_num = arr[i-1]
            continue
        if sta != -1 and arr[i] != -1:
            end_num = arr[i]
            for j in range(sta+1, i):
                arr[j] = sta_num + (j - sta) * (end_num - sta_num) / (i - sta)
            sta = -1
    if sta != -1:
        arr[len_a-1] = 0
        for i in range(sta+1, len_a-1):
            arr[i] = sta_num - (i - sta) * sta_num / (len_a - 1 - sta)


def get_pupil_psd_de_smooth(pl, pr, window_size, overlap_rate, eye_freq):
    # convert raw data to numeric
    len_n = len(pl)
    for i in range(len_n):
        if pl[i] == '':
            pl[i] = -1
        pl[i] = float(pl[i])
        if pr[i] == '':
            pr[i] = -1
        pr[i] = float(pr[i])
    # interpolate for items with values -1
    interpolate(pl)
    interpolate(pr)
    pl = np.array(pl)
    pr = np.array(pr)
    de_smoothed_l, psd_smooth_l = get_DE_PSD_smooth(pl, window_size, overlap_rate, FREQUENCY_BANDS)
    de_smooth_r, psd_smooth_r = get_DE_PSD_smooth(pr, window_size, overlap_rate, FREQUENCY_BANDS)
    shape = (len(FREQUENCY_BANDS), -1)
    de_smoothed_l = np.reshape(de_smoothed_l, shape)
    de_smooth_r = np.reshape(de_smooth_r, shape)
    psd_smooth_l = np.reshape(psd_smooth_l, shape)
    psd_smooth_r = np.reshape(psd_smooth_r, shape)
    return de_smoothed_l, psd_smooth_l, de_smooth_r, psd_smooth_r


def get_statistics_fea(time_arr, pl, pr, event, duration, sac_amp, sfreq):
    n_samples = len(pl)
    features_all= np.zeros(0)

    for i in range(n_samples):
        if pl[i] == '':
            pl[i] = -1
        pl[i] = float(pl[i])
        if pr[i] == '':
            pr[i] = -1
        pr[i] = float(pr[i])
    interpolate(pl)  # interpolation for values -1 in the PupilLeft list
    interpolate(pr)  # interpolation for values -1 in the PupilRight list

    data_pl = np.array(pl)
    data_pl.shape=(1,data_pl.shape[0])
    data_pr = np.array(pr)
    data_pr.shape=(1,data_pr.shape[0])

    mean_fix_dur_arr = np.zeros(1)  # fixation duration mean
    std_fix_dur_arr = np.zeros(1)   # fixaation duration std
    freq_fix_dur_arr = np.zeros(1)  # fixation duration frequency
    max_fix_dur_arr = np.zeros(1)   # fixation duration maximum

    mean_sac_dur_arr = np.zeros(1)  # saccade duration mean
    std_sac_dur_arr = np.zeros(1)   # saccade duration std
    freq_sac_dur_arr = np.zeros(1)  # saccade duration frequency
    ave_sac_amp_arr = np.zeros(1)   # saccade amplitude average array
    ave_sac_lat_arr = np.zeros(1)   # saccade latency average array

    mean_bli_dur_arr = np.zeros(1)  # blink duration mean
    std_bli_dur_arr = np.zeros(1)   # blink duration std
    freq_bli_dur_arr = np.zeros(1)  # blink duration frequency

    time_sec = n_samples / sfreq
    pupil_left_mean_arr = np.zeros(time_sec)
    pupil_left_std_arr = np.zeros(time_sec)
    pupil_right_mean_arr = np.zeros(time_sec)
    pupil_right_std_arr = np.zeros(time_sec)
    sac_amp_arr = np.zeros(time_sec)    #saccade amplitude

    duration_sec = cal_seconds(time_arr,0,de_end-de_beg)  #计算开始和结束时间之间的秒数
    if(duration_sec==0):
        duration_sec = 1
    #初始化相关数据
    fix_times = 0
    sac_times = 0
    bli_times = 0
    max_fix_dur = 0
    time_fix_dur = []
    time_sac_dur = []
    time_bli_dur = []
    sac_amp_data = []
    total_sac_latency_sum = 0
    fix_flag = 1
    sac_flag = 1
    bli_flag = 1
    sac_flag_2 = 1
    sac_flag_3 = 1
    #计算一次决策里的saccade amplitude的平均值
    for i in range(n_samples):
        if sac_amp[i] != '':
            if sac_flag_2:
                sac_amp_data.append(float(sac_amp[i]))
                sac_flag_2 = 0
            continue
        sac_flag_2 = 1
    #计算fixation duration的平均值,方差,最大值以及fixation frequency
    for i in range(de_end-de_beg+1):
        if event[i]=='Fixation':
            if fix_flag:
                fix_times=fix_times+1
                tmp_data = int(duration[i])
                time_fix_dur.append(tmp_data)
                if(tmp_data > max_fix_dur):
                    max_fix_dur = tmp_data
                fix_flag = 0
            continue
        fix_flag = 1
    #计算saccade duration的平均值,方差以及saccade frequency和saccade latency(ms)
    sac_time_1 = -1
    for i in range(de_end-de_beg+1):
        if event[i]=='Saccade':
            if sac_flag:
                sac_times=sac_times+1
                time_sac_dur.append(int(duration[i]))
                sac_flag = 0
                if(sac_time_1!=-1):
                    total_sac_latency_sum = total_sac_latency_sum+cal_mseconds(time_arr,sac_time_1,i)
                    sac_time_1 = -1
            continue
        if(sac_flag==0):
            sac_time_1 = i
        sac_flag = 1
    #计算blink duration的平均值,方差以及blink frequency
    for i in range(de_end-de_beg+1):
        if event[i]=='Unclassified':
            if bli_flag:
                bli_times=bli_times+1
                time_bli_dur.append(int(duration[i]))
                bli_flag = 0
            continue
        bli_flag = 1
    #初始化储存所有眼动特征的数组
    features_all_tmp = np.zeros(12)
    if duration_sec == 0:
        duration_sec = 1
    features_all_tmp[0] = mean_fix_dur_arr = np.mean(time_fix_dur)
    features_all_tmp[1] = std_fix_dur_arr = np.std(time_fix_dur)
    features_all_tmp[2] = freq_fix_dur_arr = fix_times/duration_sec
    features_all_tmp[3] = max_fix_dur_arr = max_fix_dur

    features_all_tmp[4] = mean_sac_dur_arr = np.mean(time_sac_dur)
    features_all_tmp[5] = std_sac_dur_arr = np.std(time_sac_dur)
    features_all_tmp[6] = freq_sac_dur_arr = sac_times/duration_sec
    features_all_tmp[7] = ave_sac_amp_arr = np.mean(sac_amp_data)
    if sac_times == 1:
        features_all_tmp[8] = ave_sac_amp_arr = total_sac_latency_sum/sac_times
    else:
        features_all_tmp[8] = ave_sac_amp_arr = total_sac_latency_sum/(sac_times-1)

    if(len(time_bli_dur)==0):
        features_all_tmp[9] = features_all_tmp[10] =features_all_tmp[11] = 0
    else:
        features_all_tmp[9] = mean_bli_dur_arr = np.mean(time_bli_dur)
        features_all_tmp[10] = std_bli_dur_arr = np.std(time_bli_dur)
        features_all_tmp[11] = freq_bli_dur_arr = bli_times/duration_sec
    # 计算瞳孔直径的平均值,方差
    for sec in range(time_sec):
        pupil_left_mean_arr[sec] = np.mean(data_pl[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_right_mean_arr[sec] = np.mean(data_pr[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_left_std_arr[sec] = np.std(data_pl[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_right_std_arr[sec] = np.std(data_pr[0,sec*sfreq:sec*sfreq+sfreq])
        # 计算每秒的saccade amplitude
        for k in range(sec*sfreq,sec*sfreq+sfreq):
            if(sac_amp[k]!=''):
                sac_amp_arr[sec]=float(sac_amp[k])
                break

    features_all_tmp = np.repeat(features_all_tmp,time_sec)
    # copy features into seconds. eg: np.repeat([1,2,3],3) -> [1,1,1,2,2,2,3,3,3]
    # 拼接眼动特征
    features_all=np.concatenate((features_all,pupil_left_mean_arr,pupil_left_std_arr,pupil_right_mean_arr,pupil_right_std_arr,sac_amp_arr,features_all_tmp))
    features_all.shape = ((time_sec,17))
    return features_all


def find_indices_of_triggers(elapsed_time_col, triggers):
    # elapsed_time_col and triggers are both lists of monotonically increasing int values,
    # which are the time elapsed since the recording start time
    # s.t. min(elapsed_time_col) <= min(triggers) <= max(triggers) <= max(elapsed_time_col)
    indices = []
    n_col = len(elapsed_time_col)
    col_idx = 0
    n_trig = len(triggers)
    trig_idx = 0
    while trig_idx < n_trig:
        if elapsed_time_col[col_idx] >= triggers[trig_idx]:
            # boundary condition
            if col_idx == 0:
                indices.append(col_idx)
            elif elapsed_time_col[col_idx] + elapsed_time_col[col_idx-1] >= 2*triggers[trig_idx]:
                indices.append(col_idx-1)
            else:
                indices.append(col_idx)
            trig_idx += 1
        else:
            col_idx += 1
    assert len(indices) == len(triggers)
    return indices


def get_eye_feature_smooth(data, start_idx, end_idx, window_size, overlap_rate):
    desc_row = data.row_values(0)
    colindex = data.col_values(0)
    n_cols = len(desc_row)
    len_c = len(colindex)

    # 提取出相关的列
    for i in range(n_cols):
        if desc_row[i] == 'LocalTimeStamp':
            time_col = data.col_values(i)[1+start_idx:2+end_idx]
        if desc_row[i] == 'PupilLeft':
            pupil_col_l = data.col_values(i)[1 + start_idx: 2 + end_idx]
            pupil_col_r = data.col_values(i+1)[1 + start_idx: 2 + end_idx]
        if desc_row[i] == 'GazeEventType':
            gaze_event = data.col_values(i)[1 + start_idx:2 + end_idx]
            gaze_duration = data.col_values(i+1)[1 + start_idx:2 + end_idx]
        if desc_row[i] == 'SaccadicAmplitude':
            sac_amplitude = data.col_values(i)[1 + start_idx:2 + end_idx]
    # 计算瞳孔直径的DE和PSD特征
    band_DE_l, band_PSD_l, band_DE_r, band_PSD_r = get_pupil_psd_de_smooth(time_col, pupil_col_l, pupil_col_r,
                                                                           window_size, overlap_rate, EYE_FREQ)
    # 计算其他眼动特征
    features_all = get_statistics_fea(time_col, pupil_col_l, pupil_col_r, gaze_event, gaze_duration,
                                      sac_amplitude, start_idx, end_idx, EYE_FREQ)
    # 返回拼接好的特征 shape=(n,33)
    return np.concatenate((band_DE_l.T, band_DE_r.T, band_PSD_l.T, band_PSD_r.T, features_all), axis=1)


if __name__ == '__main__':
    print("start")
    a = [0, -1, 3, -1, -1, 2, -1]
    interpolate(a)
    print(a)
