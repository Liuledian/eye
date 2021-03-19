#import mne
import numpy as np
import string
import matplotlib.pyplot as plt
import os
import xlrd
from xlrd import xldate_as_tuple
import xlsxwriter
import xlwt
import scipy.io as scio
from datetime import datetime
from pykalman import KalmanFilter
import scipy.signal as signal

FREQUENCY_BANDS = [
  (0,0.2),
  (0.2,0.4),
  (0.4,0.6),
  (0.6,1)
]
freq = 120
nchan = 1
STFTN = 256
def normalization(matrix, flag, axis):
    """
    将矩阵归一化
    :param matrix: 待归一化矩阵
    :param flag: 归一化方法
                 0：(0-1) 标准化      x_n = (x-min)/(max-min)
                 1: Z-score 标准化   x_n = (x-mean)/std
                 2: log函数 标准化    x_n = log(x)/log(max)
                 3：反正切函数 标准化  x_n = atan(x)*2/pi
    :param axis: 0表示按列归一化， 1表示按行归一化
    :return: return_norm_matrix 归一化后的矩阵
    """
    if axis == 1:
        matrix = matrix.T
    if flag > 3:
        flag = 1
    if flag == 0:
        min_axis = np.min(matrix, 0)
        max_axis = np.max(matrix, 0)
        range_axis = max_axis - min_axis
        return_norm_matrix = (matrix - min_axis)/range_axis
    if flag == 1:
        mean_axis = np.mean(matrix, 0)
        std_axis = np.std(matrix, 0)
        return_norm_matrix = (matrix-mean_axis)/std_axis
    if flag == 2:
        max_axis = np.max(matrix, 0)
        return_norm_matrix = np.log10(matrix)/np.log(max_axis)
    if flag == 3:
        return_norm_matrix = np.arctan(matrix)*2/np.pi
    if axis == 1:
        return return_norm_matrix.T
    return return_norm_matrix

def cal_de_psd(slice_data, window_size, overlap_rate,frequency_bands=FREQUENCY_BANDS):
    band_PSD, band_DE=get_band_PSD_DE_features(slice_data,window_size,overlap_rate,frequency_bands)
    return band_PSD,band_DE

def cal_de_psd_smooth(slice_data,window_size,overlap_rate,frequency_bands=FREQUENCY_BANDS):
    #计算DE和PSD特征
    #print(slice_data.shape)
    band_PSD, band_DE=get_band_PSD_DE_features(slice_data,window_size,overlap_rate,frequency_bands)
    #print(band_PSD.shape,band_DE.shape)
    de_smooth_data = smooth_features(band_DE)
    psd_smooth_data = smooth_features(band_PSD)
    return de_smooth_data,psd_smooth_data

def get_band_PSD_DE_features(slice_data,window_size,overlap_rate,frequency_bands):
    slice_data.shape=(1,slice_data.shape[0])
    window_points=window_size*freq
    step_points = (1-overlap_rate)*window_points
    #window_num = int(np.floor((slice_data.get_data().shape[1]-window_points)/step_points+1))
    #print(slice_data.shape)
    window_num = int(np.floor((slice_data.shape[1]-window_points)/step_points+1))


    band_PSD = np.zeros((nchan,len(frequency_bands),window_num))
    band_DE = np.zeros_like(band_PSD)
    for window_index in range(window_num):
        #data_win=slice_data.get_data()[:,int(step_points * window_size):int(step_points*window_index+window_points)]
        data_win=slice_data[:,int(step_points * window_index):int(step_points*window_index+window_points)]
        data_win=np.asarray(data_win)
        x = signal.hann(int(freq*window_size))
        Hdata = data_win * x
        FFTdata = np.fft.fft(Hdata,n=STFTN)
        magFFTdata = np.abs(FFTdata[:,0:int(STFTN/2)])
        window_band_PSD,window_band_DE = get_frequency_band_feature(magFFTdata,frequency_bands)

        band_PSD[:,:,window_index] = window_band_PSD
        band_DE[:,:,window_index] = window_band_DE
    return band_PSD,band_DE

def get_frequency_band_feature(magFFTdata,frequency_bands):
    band_energy_bucket = np.zeros((magFFTdata.shape[0], len(frequency_bands)))
    #print(band_energy_bucket.shape)
    #print(magFFTdata.shape)
    band_frequency_count = np.zeros(len(frequency_bands))
    for band_index, (fmin, fmax) in enumerate(frequency_bands):
        fStartNum = np.floor(fmin / freq * STFTN)
        fEndNum = np.floor(fmax / freq * STFTN)
        for p in range(int(fStartNum - 1), int(fEndNum)):
            band_energy_bucket[:, band_index] += magFFTdata[:, p] * magFFTdata[:, p]
            band_frequency_count[band_index] += 1
    window_band_PSD = band_energy_bucket / band_frequency_count  # Scale to uV
    window_band_DE = np.log2(100 * window_band_PSD)


    return window_band_PSD, window_band_DE

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

    state_mean = np.mean(feature_data,axis=2)
    for channel_index in range(feature_data.shape[0]):
        for feature_band_index in range(feature_data.shape[1]):
            kf = KalmanFilter(transition_matrices=1,
                              observation_matrices=1,
                              transition_covariance=0.001,
                              observation_covariance=1,
                              initial_state_covariance=0.1,
                              initial_state_mean=state_mean[channel_index,feature_band_index])

            measurement = feature_data[channel_index, feature_band_index, :]
            smoothed_data[channel_index, feature_band_index, :] = kf.smooth(measurement)[0].flatten()
            #print(smoothed_data.shape)
    return smoothed_data


# hebing shunxu -> features_all
# 17 features in sum
concatenation_order_of_feature_names =['pupil_mean_left','pupil_std_left','pupil_mean_right','pupil_std_right','sac_amplitude',
'fix_duration_mean','fix_duration_std','fix_duration_freq','fix_duration_max','sac_duration_mean',
'sac_duration_std','sac_duration_freq','sac_amplitude_ave','sac_latency_ave','bli_duration_mean',
'bli_duration_std','bli_duration_freq']

'''
    for i in range(len_r):
        if rowindex[i]=='LocalTimeStamp':
            print('LocalTimeStamp:',i)
        if rowindex[i]=='PupilLeft':
            print('PupilLeft',i)
        if rowindex[i]=='EyeTrackerTimestamp':
            print('EyeTrackerTimestamp',i)
        if rowindex[i]=='GazeEventType':
            print('GazeEventType',i)
        if rowindex[i]=='SaccadicAmplitude':
            print('SaccadicAmplitude',i)

    time_col = data.col_values(25)[1+index_sta:2+index_end]
    #time_stamp = data.col_values(26)[1+index_sta:2+index_end]
    gaze_event = data.col_values(44)[1+index_sta:2+index_end]
    gaze_duration = data.col_values(45)[1+index_sta:2+index_end]
    sac_amplitude = data.col_values(48)[1+index_sta:2+index_end]
    pupil_col_l = data.col_values(78)[1+index_sta:2+index_end]
    pupil_col_r = data.col_values(79)[1+index_sta:2+index_end]
result:
LocalTimeStamp: 25
EyeTrackerTimestamp 26
GazeEventType 44
SaccadicAmplitude 48
PupilLeft 78
'''
def cal_eye_feature_smooth(data,index_sta,index_end,window_size,overlap_rate):
    rowindex = data.row_values(0)
    colindex = data.col_values(0)
    len_r = len(rowindex)
    len_c = len(colindex)

    #print(len_r,len_c,index_sta,index_end)
    #提取出相关的列
    for i in range(len_r):
        if rowindex[i]=='LocalTimeStamp':
            time_col = data.col_values(i)[1+index_sta:2+index_end]
            #print('LocalTimeStamp:',i)
        if rowindex[i]=='PupilLeft':
            pupil_col_l = data.col_values(i)[1+index_sta:2+index_end]
            pupil_col_r = data.col_values(i+1)[1+index_sta:2+index_end]
            #print('PupilLeft',i)
        if rowindex[i]=='GazeEventType':
            gaze_event = data.col_values(i)[1+index_sta:2+index_end]
            gaze_duration = data.col_values(i+1)[1+index_sta:2+index_end]
            #print('GazeEventType',i)
        if rowindex[i]=='SaccadicAmplitude':
            sac_amplitude = data.col_values(i)[1+index_sta:2+index_end]
            #print('SaccadicAmplitude',i)
    #time_stamp = data.col_values(26)[1+index_sta:2+index_end]
    #计算瞳孔直径的DE和PSD特征
    band_DE_l,band_PSD_l,band_DE_r,band_PSD_r = cal_pupil_psd_de_smooth(time_col,pupil_col_l,pupil_col_r,index_sta,index_end,window_size,overlap_rate,120)
    #计算其他眼动特征
    features_all = cal_features(time_col,pupil_col_l,pupil_col_r,gaze_event,gaze_duration,sac_amplitude,index_sta,index_end,120)
    #返回拼接好的特征
    return np.concatenate((band_DE_l.T,band_DE_r.T,band_PSD_l.T,band_PSD_r.T,features_all),axis=1) #shape =(n,33)

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

def cal_inter(arr):  #interpolation for values -1
    #对数组取插值平均
    len_a = len(arr)
    sta = -1
    sta_num = -1
    end = -1
    end_num = -1
    for i in range(len_a):
        if(sta == -1 and arr[0]==-1):
            sta = 0
            sta_num = 0
            arr[0] = 0
            continue
        if(sta == -1 and arr[i]==-1):
            sta = i-1
            sta_num = arr[i-1]
            continue
        if(sta!=-1 and arr[i]!=-1):
            end_num = arr[i]
            for j in range(sta+1,i):
                arr[j]= sta_num+ (j-sta)*(end_num-sta_num)/(i-sta)
            sta = -1
    if sta != -1:
        arr[len_a-1]=0
        for i in range(sta+1,len_a-1):
            arr[i]=sta_num-(i-sta)*sta_num/(len_a-1-sta)

def cal_pupil_psd_de_smooth(time_arr,pl,pr,sta,end,window_size,overlap_rate,sfreq):
    #计算瞳孔直径的PSD和DE特征
    len_n = len(pl)
    for i in range(len_n):
        if pl[i]=='':
            pl[i] = -1
        pl[i]=float(pl[i])
        if pr[i]=='':
            pr[i] = -1
        pr[i]=float(pr[i])
    #对瞳孔数据取插值平均，将-1(没有采集到的数据)的值插值
    cal_inter(pl)    # interpolation for values -1
    cal_inter(pr)
    data_pl = np.array(pl)
    data_pr = np.array(pr)
    len_z = int((end-sta+1)/sfreq)
    #print(len_z, len(pl),data_pl.shape)
    band_DE_l = np.zeros((1,4,len_z))
    band_DE_r = np.zeros((1,4,len_z))
    band_PSD_l = np.zeros((1,4,len_z))
    band_PSD_r = np.zeros((1,4,len_z))
    band_DE_l,band_PSD_l = cal_de_psd_smooth(data_pl,window_size,overlap_rate)
    band_DE_r,band_PSD_r = cal_de_psd_smooth(data_pr,window_size,overlap_rate)
    band_DE_l.shape=(4,len_z)
    band_DE_r.shape=(4,len_z)
    band_PSD_l.shape=(4,len_z)
    band_PSD_r.shape=(4,len_z)
    return band_DE_l,band_PSD_l,band_DE_r,band_PSD_r


def cal_features(time_arr,pl,pr,event,duration,sac_amp,de_beg,de_end,sfreq):
    len_n = len(pl)
    len_e = len(event)
    features_all= np.zeros(0)

    for i in range(len_n):
        if pl[i]=='':
            pl[i] = -1
        pl[i]=float(pl[i])
        if pr[i]=='':
            pr[i] = -1
        pr[i]=float(pr[i])
    cal_inter(pl)  # interpolation for values -1 in the PupilLeft list
    cal_inter(pr)  # interpolation for values -1 in the PupilRight list

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

    time_sec = int(np.floor((de_end-de_beg+1)/sfreq))
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
    for i in range(de_end-de_beg+1):
        if (sac_amp[i]!=''):
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
    #计算瞳孔直径的平均值,方差
    for sec in range(time_sec):
        pupil_left_mean_arr[sec] = np.mean(data_pl[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_right_mean_arr[sec] = np.mean(data_pr[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_left_std_arr[sec] = np.std(data_pl[0,sec*sfreq:sec*sfreq+sfreq])
        pupil_right_std_arr[sec] = np.std(data_pr[0,sec*sfreq:sec*sfreq+sfreq])
        #计算每秒的saccade amplitude
        for k in range(sec*sfreq,sec*sfreq+sfreq):
            if(sac_amp[k]!=''):
                sac_amp_arr[sec]=float(sac_amp[k])
                break

    features_all_tmp = np.repeat(features_all_tmp,time_sec)   #copy features into seconds. eg: np.repeat([1,2,3],3) -> [1,1,1,2,2,2,3,3,3]
    #拼接眼动特征
    features_all=np.concatenate((features_all,pupil_left_mean_arr,pupil_left_std_arr,pupil_right_mean_arr,pupil_right_std_arr,sac_amp_arr,features_all_tmp))
    features_all.shape = ((time_sec,17))
    return features_all
