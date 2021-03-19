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
import time
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from imblearn.under_sampling import RandomUnderSampler
import random
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
#from eyemove_extract_features import cal_eye_feature_smooth

def cal_stamp_ms(x):        #计算时间戳的毫秒
    time_ms = int(x*1000) - (int(x))*1000
    return time_ms

def find_index_of_stamp(stamp_arr,stamp_arr_ms,time_col):
    cnt_1 = 0
    len_sti = len(stamp_arr)                         #由时间戳转化成的时间字符串数组
    len_c = len(time_col)                            #excel表时间字符串数组
    time_list_trig= np.zeros(len_sti,dtype=int)      #excel表对应时间戳的索引
    cnt_11 = 0
    # for i in range(len_sti):
    #     if stamp_arr_ms[i]>= 990:
    #         stamp_arr_ms[i] = stamp_arr_ms[i]-10
    for i in range(len_c):
        if time_col[i][:8] == stamp_arr[cnt_1]:     #时,分,秒与时间戳匹配
            cnt_11 = cnt_11 + 1
            ms_tmp = stamp_arr_ms[cnt_1]
            if int(time_col[i+1][9:])> ms_tmp and int(time_col[i][9:]) <= ms_tmp:    #时间戳毫秒是否介于这行和下一行之间
                time_list_trig[cnt_1] = i
                cnt_1 = cnt_1 +1
                if cnt_1==len_sti:                   #匹配完成
                    break
                continue
            elif cnt_11 == 1 and int(time_col[i][9:])>ms_tmp:    #时间戳毫秒是否大于0且小于当前行
                time_list_trig[cnt_1] = i
                cnt_1 = cnt_1 + 1
                if cnt_1 == len_sti:                 #匹配完成
                    break
            elif int(time_col[i][9:])<= ms_tmp and int(time_col[i][9:])>=int('990'):    #时间戳毫秒是否大于990
                time_list_trig[cnt_1] = i
                cnt_1 = cnt_1 + 1
                if cnt_1 == len_sti:                 #匹配完成
                    break
        else:
            cnt_11 = 0

    return time_list_trig


def extract_eye_data(data,trig_data,confidence,win,overlap_rate):
    time_sta = trig_data[0]
    time_end = trig_data[1]
    tmp_i = 0
    aa=time_list_trig_31
    bb=time_list_trig_32
    len1 = len(time_list_trig_31)
    len2 = len(time_list_trig_32)
    #print(len1,len2)
    tmp_j = len(time_list_trig_31)
    #脑电中每张图片开始索引，眼动中记录了相应时间向量
    #如果计算飞机取时间介于time_list_trig_3[0]:time_list_trig_3[1]之间的time_list_trig_31
    #如果计算船取时间介于time_list_trig_3[1]:time_list_trig_2之间的time_list_trig_31
    for i in range(len1):
        if time_list_trig_31[i] >= time_sta:
            tmp_i = i
            print(i)
            break
    for i in range(len1):
        if time_list_trig_31[len1-i-1]<= time_end:
            print(time_list_trig_31[len1-i-1])
            print(time_end)
            tmp_j = len1-i-1
            print(i)
            break

    index_stimuli = time_list_trig_31[tmp_i:tmp_j+1]   #图片索引数组

    #脑电中每次点击鼠标圈飞机或船的索引，眼动中记录了相应时间向量
    #如果计算飞机只取时间介于time_list_trig_3[0]:time_list_trig_3[1]之间的time_list_trig_32
    #如果计算船只取时间介于time_list_trig_3[1]:time_list_trig_2之间的time_list_trig_32
    for i in range(len2):
        if time_list_trig_32[i]>= time_sta:
            tmp_i = i
            print(i)
            break
    for i in range(len2):
        if time_list_trig_32[len2-i-1]<= time_end:
            tmp_j = len2-i-1
            print(i)
            break
    index_click = time_list_trig_32[tmp_i:tmp_j+1]    #点击鼠标数组

    label_arr=np.zeros(0)
    feature_arr=np.zeros(0)

    #计算方法:有i张图片,一张图片stimuli对应j次决策click,其中第一次决策时间段为这张图片stimuli开始到第一次click
    #之后的决策时间段为上一次click到这一次click之间,直到下一张图片开始
    for i in range(len(index_stimuli)):
        first_flag=True     #判断是否是图片i的第一次决策
        num=-1              #计算click j是这张图片的第几次决策
        for j in range(len(index_click)):
            if i !=len(index_stimuli)-1:
                end=index_stimuli[i+1]         #不是最后一张图片时的图片结束时间
            else:
                end=trig_data[1]               #最后一张图片的图片结束时间
            if index_click[j]>index_stimuli[i] and index_click[j]<end:     #判断click j是否属于图片i
                num += 1
                try:
                    a=confidence[i][num]
                except:
                    b=1
                    c=1
                if confidence[i][num] != 0:    #去掉为0的误触决策
                    if first_flag:             #判断是否是第i张图片的第一次click
                        index_beg = index_stimuli[i]
                        index_end = index_click[j]
                        first_flag=False
                    else:
                        index_beg = index_click[j-1]
                        index_end = index_click[j]
                    if index_end-index_beg >= win*120:    #计算时间窗长度为1s(120个数据)的眼动特征
                        feature_i = cal_eye_feature_smooth(data,index_beg,index_end,window_size=win,overlap_rate=overlap_rate)
                        if(i==0 and j==0):
                            feature_arr = feature_i
                        else:
                            feature_arr = np.row_stack((feature_arr,feature_i))
                        label_arr = np.append(label_arr,[confidence[i][num]]*feature_i.shape[0])
    return feature_arr,label_arr   #返回眼动特征和对应的标签


def contract_eye_eeg(load_path,load_label_path,save_path,save_label_path,file_name):
    for exp_num in ['1','2']:
        i = 0
        for root,par,name in os.walk(load_path):
            if i == 0:
                i = i + 1
                for name_item in par:
                    print('contract---',file_name[0],'-: ',name_item)
                    eeg_data_f1 = name_item+'/'+exp_num+file_name[1]
                    eeg_label_f1 = name_item+'/'+exp_num+file_name[2]
                    load_path_eeg_data1 = '../data/wet data/EEG_DATA/de/de_smooth_short_fft256_win_1_shortime_10000000/'+ eeg_data_f1
                    load_path_eeg_label1 = '../data/wet data/EEG_DATA/label/label_fft256_win_1_shortime_10000000/'+ eeg_label_f1

                    load_path_eye_data1 = load_path+name_item+'/'+exp_num+'/'+file_name[3]
                    load_path_eye_label1 = load_label_path+name_item+'/'+exp_num+'/'+file_name[4]

                    #读取脑电特征文件和标签文件
                    eeg_data_f1 = np.load(load_path_eeg_data1)
                    eeg_label_f1 = np.load(load_path_eeg_label1)
                    #读取眼动特征文件和标签文件
                    eye_data_f1 = np.load(load_path_eye_data1)
                    eye_label_f1 = np.load(load_path_eye_label1)

                    #将(5,62)转化为(310)脑电特征
                    s1 = eeg_data_f1.shape
                    eeg_data_f1 = eeg_data_f1.reshape((s1[0],s1[1]*s1[2]))
                    save_to=save_path+name_item+'/'+exp_num
                    if not os.path.exists(save_to):
                        os.makedirs(save_to)
                    np.save(save_to+file_name[5],eeg_data_f1)


                    l1 = len(eye_label_f1)
                    l2 = len(eeg_label_f1)
                    print('before')
                    print('eye:',l1)
                    print('eeg:', l2)
                    j = 0
                    flag = 1
                    not_match = []
                    #对比脑电特征和眼动特征的标签
                    for i in range(l1):
                        if flag and (eye_label_f1[i] == eeg_label_f1[j]):
                            j = j + 1
                            if j == l2:
                                print('contract ok!',j,l1,l2)
                                flag = 0
                        else:
                            not_match.append(i)

                    #删除眼动特征不匹配的数据
                    eye_data_f1 = np.delete(eye_data_f1,not_match,axis=0)
                    eye_label_f1 = np.delete(eye_label_f1,not_match)
                    #保存匹配一直的眼动特征和脑电特征
                    save_to_lab=save_label_path+name_item+'/'+exp_num
                    if not os.path.exists(save_to_lab):
                        os.makedirs(save_to_lab)
                    np.save(save_path+name_item+'/'+exp_num+file_name[6],eye_data_f1)
                    np.save(save_label_path+name_item+'/'+exp_num+file_name[7],eye_label_f1)
                    #匹配成功
                    if j == l2:
                        print('matched ok!',eye_label_f1.shape==eeg_label_f1.shape)
                    print('after')
                    print('eye:', eye_label_f1.shape)
                    print('eeg:', eeg_label_f1.shape)

def cal_feature():
    win = 1
    overlap_rate = 0
    save_path_eye='../data/wet data/EYE_FDATA/'
    if not os.path.exists(save_path_eye):
        os.makedirs(save_path_eye)
    for exp_num in ['1','2']:
        load_path_eye_data = '../data/wet data/eye_data/'+exp_num+'/'
        for root,parents,name in os.walk(load_path_eye_data):
            for name_item in name:
                    #name_item='lirui_Confidence_20190110_confidence_huangzhongyu_20191103_p1_wet.xlsx'
    
                    peo = root +'/' + name_item
                    print('processing----',peo)
                    eye_file = xlrd.open_workbook(load_path_eye_data+name_item)
                    data = eye_file.sheets()[0]
                    rowindex = data.row_values(0)
                    colindex = data.col_values(0)
                    len_r = len(rowindex)
                    len_c = len(colindex)
                    load_path_time_data = '../data/wet data/choice/part'+exp_num+'/'+name_item[26:-5]+'/part'+exp_num+'/'
                    file_time = np.load(load_path_time_data+'time_list_trig.npz')   #读取时间戳文件
                    time_all = []
                    time_all_ms = []
                    for i in range(2):                     #求时间戳毫秒并且保存时间戳和时间戳毫秒数据
                        time_all_ms.append(cal_stamp_ms(file_time['time_list_trig_3'][i]))
                        time_all.append(file_time['time_list_trig_3'][i])
                    time_all_ms.append(cal_stamp_ms(file_time['time_list_trig_2'][0]))
                    time_all.append(file_time['time_list_trig_2'][0])
                    for i in range(3):                     #时间戳转时间字符串
                        time_all[i] = time.strftime("%H:%M:%S", time.localtime(time_all[i]))
    
                    for i in range(len_r):                 #眼动数据文件中存时间字符串的一列
                        if rowindex[i] == 'LocalTimeStamp':
                            time_col = data.col_values(i)
    
                    time_stimuli = []
                    time_stimuli_ms = []
                    time_click = []
                    time_click_ms = []
    
                    for i in range(0,len(file_time['time_list_trig_31'])):
                        time_stimuli_ms.append(cal_stamp_ms(file_time['time_list_trig_31'][i]))
                        time_stimuli.append(time.strftime("%H:%M:%S", time.localtime(file_time['time_list_trig_31'][i])))
                    for i in range(0,len(file_time['time_list_trig_32'])):
                        time_click_ms.append(cal_stamp_ms(file_time['time_list_trig_32'][i]))
                        time_click.append(time.strftime("%H:%M:%S", time.localtime(file_time['time_list_trig_32'][i])))
    
    
                    time_list_trig_3 =  find_index_of_stamp(time_all,time_all_ms,time_col)          #求trigger时间戳在文件中的索引
                    time_list_trig_31 = find_index_of_stamp(time_stimuli,time_stimuli_ms,time_col)
                    time_list_trig_32 = find_index_of_stamp(time_click,time_click_ms,time_col)
    
                    confidence_plane = np.load(load_path_time_data+'task3_plane.npz')['confidence']    #加载决策信心文件
                    #print('confidence:',load_path_time_data)
                    save_to = '../data/wet data/EYE_FDATA/feature/'+name_item[37:-21]+'/'+exp_num
                    save_to_lab = '../data/wet data/EYE_FDATA/label/'+name_item[37:-21]+'/'+exp_num
    
                    if not os.path.exists(save_to):
                        os.makedirs(save_to)
                    if not os.path.exists(save_to_lab):
                        os.makedirs(save_to_lab)
    
                    feature_arr,label_arr = extract_eye_data(data,time_list_trig_3[0:2],confidence_plane,win,overlap_rate)
    
                    np.save(save_to+'/EYE_31.npy',feature_arr)
                    np.save(save_to_lab + '/label31.npy', label_arr)
    
    
                    confidence_ship = np.load(load_path_time_data+'task3_ship.npz')['confidence']    #加载决策信心文件
                    #print('confidence:',load_path_time_data)
                    feature_arr,label_arr = extract_eye_data(data,time_list_trig_3[1:3],confidence_ship,win,overlap_rate)
                    np.save(save_to + '/EYE_32.npy', feature_arr)
                    np.save(save_to_lab + '/label32.npy', label_arr)	

def contract():
    save_path_eye = '../data/wet data/EYE_FINAL'
    load_path_eye_data = '../data/wet data/EYE_FDATA/feature/'
    load_path_eye_label = '../data/wet data/EYE_FDATA/label/'
    save_path_data = '../data/wet data/DATA_pre/feature/'
    save_path_label = '../data/wet data/DATA_pre/label/'
    if not os.path.exists(save_path_data):
        os.makedirs(save_path_data)
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    #对比脑电特征和眼动特征
    file_name_1=['plane','/DE_31.npy','/label31.npy','EYE_31.npy','label31.npy','/EEG31.npy','/EYE31.npy','/label31.npy']
    file_name_2=['ship','/DE_32.npy','/label32.npy','EYE_32.npy','label32.npy','/EEG32.npy','/EYE32.npy','/label32.npy']
    contract_eye_eeg(load_path_eye_data,load_path_eye_label,save_path_data,save_path_label,file_name_1)
    contract_eye_eeg(load_path_eye_data,load_path_eye_label,save_path_data,save_path_label,file_name_2)
def make_data_for_confidence(people_name, trial_list, cro_flag, imb_type,folder_path,folder_label_path):
    cro=5
    train_data = np.zeros((0,33))
    train_label = np.zeros(0)
    test_data = np.zeros((0,33))
    test_label = np.zeros(0)
    eye_list = []
    ns = cro
    kf = KFold(n_splits=ns)
    eye_data_all = np.zeros((0, 33))
    label_data_all = np.zeros(0)

    train_data_eeg = np.zeros((0,5,62))
    test_data_eeg = np.zeros((0,5,62))
    eeg_data_all = np.zeros((0, 5,62))
    for ii in trial_list:
        eye_data=np.zeros((0, 33))
        eeg_data = np.zeros((0, 5,62))
        label_data = np.zeros(0)

        for exper_num in range(1, 3):

            eeg_path = folder_path + '/' + people_name + '/' + str(exper_num) + '/EEG' + str(ii) + '.npy'
            eye_path = folder_path + '/' + people_name + '/' + str(exper_num) + '/EYE' + str(ii) + '.npy'

            eeg_data_i = np.load(eeg_path).reshape(-1,5,62)
            eye_data_i = np.load(eye_path)

            # len_part=int(eeg_data.shape[0]/3)
            label_path = folder_label_path + '/' + people_name + '/' + str(exper_num) + '/label' + str(ii) + '.npy'
            label_data_i = np.load(label_path) - 1
            eye_data = np.row_stack((eye_data, eye_data_i))
            eeg_data = np.row_stack((eeg_data, eeg_data_i))
            label_data = np.append(label_data, label_data_i)

        for i in range(5):
            lab_index = np.where(label_data == i)[0]
            train_index_arr = []
            test_index_arr = []
            for train_index, test_index in kf.split(lab_index):
                train_index_arr.append(train_index)
                test_index_arr.append(test_index)
            train_data = np.row_stack((train_data, eye_data[lab_index[train_index_arr[cro_flag]], :]))
            train_data_eeg = np.row_stack((train_data_eeg, eeg_data[lab_index[train_index_arr[cro_flag]], :]))
            train_label = np.append(train_label, label_data[lab_index[train_index_arr[cro_flag]]])

            test_data = np.row_stack((test_data, eye_data[lab_index[test_index_arr[cro_flag]], :]))
            test_data_eeg = np.row_stack((test_data_eeg, eeg_data[lab_index[test_index_arr[cro_flag]], :]))
            test_label = np.append(test_label, label_data[lab_index[test_index_arr[cro_flag]]])
    # if channel != -1:
    #     train_data = train_data[:, :, channel]
    #     test_data = test_data[:, :, channel]
    # if band != -1:
    #     train_data = train_data[:, band, :]
    #     test_data = test_data[:, band, :]

    # s = 1
    # for i in range(1, len(train_data.shape)):
    #     s = s * train_data.shape[i]
    # train_data = train_data.reshape(-1, s)
    # test_data = test_data.reshape(-1, s)
    # if imb_type == 0:
    #     ee = RandomUnderSampler(random_state=0, replacement=True)
    #     X_resampled, y_resampled = ee.fit_sample(train_data, train_label)
    #     train_data = X_resampled
    #     train_label = y_resampled
    # elif imb_type == 1:
    #     # X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(eeg_data_all, label_data_all)
    #     X_resampled, y_resampled = BorderlineSMOTE().fit_sample(train_data, train_label)
    #     train_data = X_resampled
    #     train_label = y_resampled
    # elif imb_type == 2:
    #     X_resampled, y_resampled = SMOTE().fit_sample(train_data, train_label)
    #     train_data = X_resampled
    #     train_label = y_resampled
    # elif imb_type == 3:
    #     nm = NearMiss()
    #     X_resampled, y_resampled = nm.fit_sample(train_data, train_label)
    #     train_data = X_resampled
    #     train_label = y_resampled
    # elif imb_type == 4:
    #     X_resampled, y_resampled = ADASYN().fit_sample(train_data, train_label)
    #     train_data = X_resampled
    #     train_label = y_resampled

    return train_data, train_label, test_data, test_label,train_data_eeg, test_data_eeg
def eeg_eye_rearrange():
    load_path_data = '../data/wet data/DATA_pre/feature/'
    load_path_label = '../data/wet data/DATA_pre/label/'
    folder_path=load_path_data
    folder_label_path=load_path_label
    cro=5


    trialname_list = ['31_32', '2','31','32', '31_32_2']
    trial_list_all = [[31, 32], [2],[31],[32], [31, 32, 2]]
    for trial_index in range(1):
        trial_name = trialname_list[trial_index]
        trial_list = trial_list_all[trial_index]
        # channel_intro_dic = {'all_channel':-1,'frontal_pole__FP1_FP2':[8,9],'F3_Fz_F4':[2,3,4],'F7_T3_T5_F8_T4_T6':[14,10,11,15,17,16],'P3_P4':[0,6],'O1_O2':[12,13],'C3_Cz_C4':[1,7,5]}
        # band_intro_dic = {-1: 'all_band', 0: 'delta', 1: 'theta', 2: 'alpha', 3: 'beta', 4: 'gama'}
        # channel_intro_dic = {'all_channel':-1,'frontal_pole__FP1_FP2':[8,9],'F3_Fz_F4':[2,3,4],'FP1_FP2_F3_Fz_F4':[8,9,2,3,4],'F7_T3_T5_F8_T4_T6':[14,10,11,15,17,16],'P3_P4':[0,6],'O1_O2':[12,13],'C3_Cz_C4':[1,7,5]}
        # channel_intro_dic = {'all_channel': -1}
        # band_intro_dic = {-1: 'all_band'}
        #imb_type_dic = {2: 'sm_0', 0: 'random', 1: 'sm_1', 3: 'nm', 4: 'ADASYN'}
        imb_type_dic = {-1: 'raw'}
        c_num = 5
        savename = 't_' + trial_name + '_l_' + str(c_num)
        for imb_type, imb_type_name in imb_type_dic.items():
            for metric_ave in ['macro']:
                # ！！！！！！！！！！！！！！
                log_path = '../data/wet data/EYE_feature_FINAL/'
                #print(band_name, channel_name)
                save_emotional_model_path = log_path
                isExists = os.path.exists(log_path)
                if not isExists:
                    os.makedirs(log_path)

                participant_name_list = os.listdir(folder_path)
                #participant_name_list=['diaoyuqi']

                for participant in participant_name_list:
                    # pool.apply_async(multiprocessing_svm, (participant, save_emotional_model_path, log_path,
                    #                 return_list, result_save_name, 0, 0))
                    if participant not in ['']:
                        try:
                            for cross_val in range(cro):
                                train_data, train_label, test_data, test_label ,train_data_eeg, test_data_eeg= make_data_for_confidence(participant, trial_list,
                                                                                                      cross_val, imb_type,folder_path,folder_label_path)
                                data_save_path=log_path + 'EYE_feature/' + participant+'/'+str(cross_val)+'/'
                                if not os.path.exists(data_save_path):
                                    os.makedirs(data_save_path)
                                np.save(data_save_path + '/train_data.npy', train_data)
                                np.save(data_save_path + '/test_data.npy', test_data)
                                np.save(data_save_path + '/train_label.npy', train_label)
                                np.save(data_save_path + '/test_label.npy', test_label)
                                data_save_path_eeg=log_path + 'EEG_feature/' + participant+'/'+str(cross_val)+'/'
                                if not os.path.exists(data_save_path_eeg):
                                    os.makedirs(data_save_path_eeg)
                                np.save(data_save_path_eeg + '/train_data.npy', train_data_eeg)
                                np.save(data_save_path_eeg + '/test_data.npy', test_data_eeg)
                                np.save(data_save_path_eeg + '/train_label.npy', train_label)
                                np.save(data_save_path_eeg + '/test_label.npy', test_label)
                        except:
                            a=1


if __name__ == '__main__':
	eeg_eye_rearrange()


