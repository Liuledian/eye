import os
# 1.2.0 xlrd
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
N_CHANNELS = 1
STFT_N = 256


def get_PSD_DE_for_a_window(mag_fft_data, frequency_band, sample_freq):
    n_channels = mag_fft_data.shape[0]
    band_energy_bucket = np.zeros((n_channels, len(frequency_band)))
    band_frequency_count = np.zeros((n_channels, len(frequency_band)))
    for band_index, (f_start, f_end) in enumerate(frequency_band):
        # Map frequency to data point indices in mag_fft_data array
        fStartNum = np.floor(f_start / sample_freq * STFT_N)
        fEndNum = np.floor(f_end / sample_freq * STFT_N)
        for p in range(int(fStartNum - 1), int(fEndNum)):
            band_energy_bucket[:, band_index] += mag_fft_data[:, p] ** 2
            band_frequency_count[:, band_index] += 1
    window_band_PSD = band_energy_bucket / band_frequency_count  # Scale to uV
    # why times 100 !!!!!!
    window_band_DE = np.log2(window_band_PSD)
    return window_band_PSD, window_band_DE


def get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_band, sample_freq):
    slice_data = np.reshape(slice_data, [N_CHANNELS, -1])
    window_points = window_size * sample_freq
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
        window_band_PSD, window_band_DE = get_PSD_DE_for_a_window(mag_fft_data, frequency_band, sample_freq)
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


def get_DE_PSD_smooth(slice_data, window_size, overlap_rate, frequency_bands, sample_freq):
    # Compute DE and PSD feature
    band_PSD, band_DE = get_PSD_DE_fea(slice_data, window_size, overlap_rate, frequency_bands, sample_freq)

    # Smooth DE and PSD feature
    de_smooth_data = smooth_features(band_DE)
    psd_smooth_data = smooth_features(band_PSD)
    return de_smooth_data, psd_smooth_data


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


def preprocess_pupil(pupil_data):
    n_samples = len(pupil_data)
    for i in range(n_samples):
        if pupil_data[i] == '':
            pupil_data[i] = -1
        pupil_data[i] = float(pupil_data[i])

    # interpolate for items with values -1
    interpolate(pupil_data)
    return np.array(pupil_data)


def get_pupil_psd_de_smooth(pl, pr, window_size, overlap_rate, sample_freq):
    # convert raw data to numeric
    pl = preprocess_pupil(pl)
    pr = preprocess_pupil(pr)
    de_smoothed_l, psd_smooth_l = get_DE_PSD_smooth(pl, window_size, overlap_rate, FREQUENCY_BANDS, sample_freq)
    de_smooth_r, psd_smooth_r = get_DE_PSD_smooth(pr, window_size, overlap_rate, FREQUENCY_BANDS, sample_freq)
    shape = (len(FREQUENCY_BANDS), -1)
    de_smoothed_l = np.reshape(de_smoothed_l, shape)
    de_smooth_r = np.reshape(de_smooth_r, shape)
    psd_smooth_l = np.reshape(psd_smooth_l, shape)
    psd_smooth_r = np.reshape(psd_smooth_r, shape)
    return de_smoothed_l, psd_smooth_l, de_smooth_r, psd_smooth_r


def get_statistics_fea(time_arr, pl, pr, event, duration, window_size, overlap_rate, sample_freq):
    n_samples = len(pl)
    window_points = window_size * sample_freq
    step_points = window_points * (1 - overlap_rate)
    n_windows = int((n_samples - window_points) / step_points) + 1
    pl = preprocess_pupil(pl)
    pr = preprocess_pupil(pr)

    duration_sec = (time_arr[len(time_arr) - 1] - time_arr[0]) / 1e6  # 计算开始和结束时间之间的秒数
    if duration_sec == 0:
        duration_sec = 1

    # 初始化相关数据
    fix_times = 0
    sac_times = 0
    bli_times = 0
    max_fix_dur = 0
    fix_dur_arr = []
    sac_dur_arr = []
    time_bli_dur = []
    # sac_amp_data = []
    total_sac_latency_sum = 0
    fix_flag = True
    sac_flag = True
    bli_flag = True

    # 计算一次决策里的saccade amplitude的平均值
    # for i in range(n_samples):
    #     if sac_amp[i] != '':
    #         if sac_flag_2:
    #             sac_amp_data.append(float(sac_amp[i]))
    #             sac_flag_2 = 0
    #         continue
    #     sac_flag_2 = 1

    # 计算fixation duration的平均值,方差,最大值以及fixation frequency
    for i in range(n_samples):
        if event[i] == 'Fixation':
            if fix_flag:
                fix_times += 1
                tmp_data = int(duration[i])
                fix_dur_arr.append(tmp_data)
                if tmp_data > max_fix_dur:
                    max_fix_dur = tmp_data
                fix_flag = False
        else:
            fix_flag = True

    # 计算saccade duration的平均值,方差以及saccade frequency和saccade latency(ms)
    prev_sac_end = -1
    for i in range(n_samples):
        if event[i] == 'Saccade':
            if sac_flag:
                sac_times += 1
                sac_dur_arr.append(int(duration[i]))
                if prev_sac_end != -1:
                    total_sac_latency_sum += int((time_arr[i] - time_arr[prev_sac_end]) / 1000)
                sac_flag = False
        else:
            if not sac_flag:
                prev_sac_end = i
            sac_flag = True

    # 计算blink duration的平均值,方差以及blink frequency
    for i in range(n_samples):
        if event[i] == 'Unclassified':
            if bli_flag:
                bli_times += 1
                time_bli_dur.append(int(duration[i]))
                bli_flag = False
        else:
            bli_flag = True

    # 11 statistic features for the whole slice
    features_all_tmp = np.zeros(11)
    if len(fix_dur_arr) > 0:
        features_all_tmp[0] = np.mean(fix_dur_arr)
        features_all_tmp[1] = np.std(fix_dur_arr)
        features_all_tmp[2] = fix_times / duration_sec
        features_all_tmp[3] = max_fix_dur
    if len(sac_dur_arr) > 0:
        features_all_tmp[4] = np.mean(sac_dur_arr)
        features_all_tmp[5] = np.std(sac_dur_arr)
        features_all_tmp[6] = sac_times / duration_sec
        # features_all_tmp[7] = np.mean(sac_amp_data)
    if sac_times > 1:
        features_all_tmp[7] = total_sac_latency_sum / (sac_times - 1)

    if len(time_bli_dur) > 0:
        features_all_tmp[8] = np.mean(time_bli_dur)
        features_all_tmp[9] = np.std(time_bli_dur)
        features_all_tmp[10] = bli_times / duration_sec

    # 计算瞳孔直径的平均值,方差
    pupil_left_mean_arr = np.zeros(n_windows)
    pupil_left_std_arr = np.zeros(n_windows)
    pupil_right_mean_arr = np.zeros(n_windows)
    pupil_right_std_arr = np.zeros(n_windows)
    # sac_amp_arr = np.zeros(time_sec)    #saccade amplitude

    # 4 statistic features for each window
    for w in range(n_windows):
        start_idx = w * step_points
        end_idx = start_idx + step_points
        pupil_left_mean_arr[w] = np.mean(pl[start_idx: end_idx])
        pupil_right_mean_arr[w] = np.mean(pr[start_idx: end_idx])
        pupil_left_std_arr[w] = np.std(pl[start_idx: end_idx])
        pupil_right_std_arr[w] = np.std(pr[start_idx: end_idx])

        # 计算每秒的saccade amplitude
        # for k in range(sec * eye_freq, sec * eye_freq + eye_freq):
        #     if(sac_amp[k]!=''):
        #         sac_amp_arr[sec]=float(sac_amp[k])
        #         break

    # concatenate all features
    # features_all_tmp shape (n_windows, 11)
    features_all_tmp = np.tile(features_all_tmp, (n_windows, 1))
    pupil_left_mean_arr = np.expand_dims(pupil_left_mean_arr, axis=1)
    pupil_left_std_arr = np.expand_dims(pupil_left_std_arr, axis=1)
    pupil_right_mean_arr = np.expand_dims(pupil_right_mean_arr, axis=1)
    pupil_right_std_arr = np.expand_dims(pupil_right_std_arr, axis=1)
    features_all = np.concatenate((pupil_left_mean_arr, pupil_left_std_arr,
                                   pupil_right_mean_arr, pupil_right_std_arr, features_all_tmp), axis=1)
    assert features_all.shape[1] == 15
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


def get_eye_feature_smooth(data, start_idx, end_idx, window_size, overlap_rate, sample_freq, fea_type):
    desc_row = data.row_values(0)
    n_cols = len(desc_row)

    # 提取出相关的列
    for i in range(n_cols):
        if desc_row[i] == 'Recording timestamp':
            time_col = data.col_values(i)[1+start_idx:2+end_idx]
        if desc_row[i] == 'Pupil diameter left':
            pupil_col_l = data.col_values(i)[1 + start_idx: 2 + end_idx]
            pupil_col_r = data.col_values(i+1)[1 + start_idx: 2 + end_idx]
        if desc_row[i] == 'Eye movement type':
            gaze_event = data.col_values(i)[1 + start_idx:2 + end_idx]
            gaze_duration = data.col_values(i+1)[1 + start_idx:2 + end_idx]

    # 计算瞳孔直径的DE和PSD特征 shape: (n_bands, n_windows)
    band_DE_l, band_PSD_l, band_DE_r, band_PSD_r = get_pupil_psd_de_smooth(pupil_col_l, pupil_col_r,
                                                                           window_size, overlap_rate, sample_freq)
    # 计算统计眼动特征 shape: (n_windows, 15)
    stat_features = get_statistics_fea(time_col, pupil_col_l, pupil_col_r, gaze_event, gaze_duration, window_size,
                                       overlap_rate, sample_freq)

    # 返回拼接好的特征 shape=(n_windows, 23)
    if fea_type == 'PSD':
        all_feature = np.concatenate((band_PSD_l.T, band_PSD_r.T, stat_features), axis=1)
    else:
        all_feature = np.concatenate((band_DE_l.T, band_DE_r.T, stat_features), axis=1)
    assert all_feature.shape[1] == 23
    return all_feature


# Extract and save eye features for an experiment session
def extract_save_eye_fea(xlsx_path, save_dir, start_triggers, end_triggers,
                         window_size, overlap_rate, sample_freq, fea_type):
    # sanity check
    assert len(start_triggers) == len(end_triggers)
    assert os.path.exists(xlsx_path)
    # create directory tree if save_dir not exists
    if not os.path.exists(save_dir):
        print("Create directory: " + save_dir)
        os.makedirs(save_dir)

    print("Start open {}".format(xlsx_path))
    eye_file = xlrd.open_workbook(xlsx_path)
    print("Successfully open {} !".format(xlsx_path))
    eye_table = eye_file.sheets()[0]
    desc_row = eye_table.row_values(0)
    n_cols = len(desc_row)
    time_col = None
    for i in range(n_cols):
        if desc_row[i] == 'Recording timestamp':
            time_col = eye_table.col_values(i)[1:]
    assert time_col is not None
    start_indices = find_indices_of_triggers(time_col, start_triggers)
    end_indices = find_indices_of_triggers(time_col, end_triggers)
    # iterate over all trials
    for i in range(len(start_triggers)):
        print("Process trial {}".format(i))
        trial_fea = get_eye_feature_smooth(eye_table, start_indices[i], end_indices[i],
                                           window_size, overlap_rate, sample_freq, fea_type)
        np.save(save_dir + '/eye_fea_{}.npy'.format(i), trial_fea)
        print(trial_fea)


if __name__ == '__main__':
    xlsx_path = '../data/raw/lirui-confidence-text confidence_text_hanxiao_20210113 copy.xlsx'
    save_dir = '../data/eye_feature'
    window_size = 1
    overlap_rate = 0
    sample_freq = 120
    fea_type = 'DE'
    start_triggers = [111562, 2111565, 4000111]
    end_triggers = [2111562, 4000000, 9000000]
    extract_save_eye_fea(xlsx_path, save_dir, start_triggers, end_triggers,
                         window_size, overlap_rate, sample_freq, fea_type)
