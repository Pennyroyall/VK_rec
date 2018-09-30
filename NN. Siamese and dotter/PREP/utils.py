import io
from dateutil.parser import parse
import six
import datetime
from time import time
import numpy as np
import pickle
import scipy.sparse as sparse
import logging
import sys
import random
from os import listdir
import io
from os.path import isfile, join



path_data = './../DATA/'
path_log = './../LOGS/'
path_res = './../RESULTS/'


path_dataset = path_data + 'discover_final_20180814_20180820/dataset/'
path_items = path_data + 'discover_final_20180814_20180820/itemFactors/'
path_users = path_data + 'discover_final_20180814_20180820/userFactors/'



def pickle_dump(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

    
def pickle_load(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data         


target_info = ['target_VIEW', 'target_GAZE', 'target_LIKE', 'target_COMMENT', 'target_SUBSCRIBE',
               'target_UNSUBSCRIBE', 'target_DISCOVER_OPEN', 'target_EXPAND', 'target_HIDE',
               'target_LINK_CLICK', 'target_LINK_CLICK_INTERNAL', 'target_LINK_CLICK_SNIPPET',
               'target_LINK_CLICK_SNIPPET_BUTTON', 'target_OPEN', 'target_OPEN_AUDIO', 'target_OPEN_GROUP',
               'target_OPEN_LAYER', 'target_OPEN_PHOTO', 'target_OPEN_USER', 'target_OPEN_WIKI',
               'target_REPORT', 'target_TRANSITION', 'target_VIDEO_START', 'target_READ', 'target_CAPTION_LINK_CLICK']


uid_info = ['uid', 
 # 'uid_age',
 # 'uid_curr_country_AM',
 # 'uid_curr_country_AZ',
 # 'uid_curr_country_BY',
 # 'uid_curr_country_DE',
 # 'uid_curr_country_EE',
 # 'uid_curr_country_FR',
 # 'uid_curr_country_GB',
 # 'uid_curr_country_IT',
 # 'uid_curr_country_KG',
 # 'uid_curr_country_KZ',
 # 'uid_curr_country_LV',
 # 'uid_curr_country_MD',
 # 'uid_curr_country_NL',
 # 'uid_curr_country_PL',
 # 'uid_curr_country_RU',
 # 'uid_curr_country_TR',
 # 'uid_curr_country_UA',
 # 'uid_curr_country_UNKNOWN',
 # 'uid_curr_country_US',
 # 'uid_curr_country_UZ',
 # 'uid_fans_num_log',
 # 'uid_friends_num_log',
 # 'uid_groups_num_log',
 # 'uid_idols_num_log',
 # 'uid_is_uni_region',
 # 'uid_joined_days_log',
 # 'uid_log', 
 'uid_sex_0',
 # 'uid_sex_1',
 # 'uid_sex_2',
 # 'uid_subscriptions_log',
 # 'uid_weights_l1',
 # 'uid_h_apps',
 # 'uid_h_audios',
 # 'uid_h_comment',
 # 'uid_h_gazed',
 # 'uid_h_hints',
 # 'uid_h_impression',
 # 'uid_h_interact',
 # 'uid_h_lowered',
 # 'uid_h_message',
 # 'uid_h_mutual_friends',
 # 'uid_h_videos',
 # 'uid_h_viewed'
 ]



oid_info = [
 # 'oid_fans_num_log',
 # 'oid_glavred_scores_ads_score_0',
 # 'oid_glavred_scores_ads_score_0_1',
 # 'oid_glavred_scores_ads_score_0_25',
 # 'oid_glavred_scores_ads_score_0_5',
 # 'oid_glavred_scores_ads_score_0_75',
 # 'oid_glavred_scores_ads_score_0_9',
 # 'oid_glavred_scores_ads_score_1',
 # 'oid_glavred_scores_ads_score_avg',
 # 'oid_glavred_scores_bad_ads_score_0',
 # 'oid_glavred_scores_bad_ads_score_0_1',
 # 'oid_glavred_scores_bad_ads_score_0_25',
 # 'oid_glavred_scores_bad_ads_score_0_5',
 # 'oid_glavred_scores_bad_ads_score_0_75',
 # 'oid_glavred_scores_bad_ads_score_0_9',
 # 'oid_glavred_scores_bad_ads_score_1',
 # 'oid_glavred_scores_bad_ads_score_avg',
 # 'oid_glavred_scores_likesos_score_0',
 # 'oid_glavred_scores_likesos_score_0_1',
 # 'oid_glavred_scores_likesos_score_0_25',
 # 'oid_glavred_scores_likesos_score_0_5',
 # 'oid_glavred_scores_likesos_score_0_75',
 # 'oid_glavred_scores_likesos_score_0_9',
 # 'oid_glavred_scores_likesos_score_1',
 # 'oid_glavred_scores_likesos_score_avg',
 # 'oid_glavred_scores_porno_score_0',
 # 'oid_glavred_scores_porno_score_0_1',
 # 'oid_glavred_scores_porno_score_0_25',
 # 'oid_glavred_scores_porno_score_0_5',
 # 'oid_glavred_scores_porno_score_0_75',
 # 'oid_glavred_scores_porno_score_0_9',
 # 'oid_glavred_scores_porno_score_1',
 # 'oid_glavred_scores_porno_score_avg',
 # 'oid_glavred_scores_suicide_score_0',
 # 'oid_glavred_scores_suicide_score_0_1',
 # 'oid_glavred_scores_suicide_score_0_25',
 # 'oid_glavred_scores_suicide_score_0_5',
 # 'oid_glavred_scores_suicide_score_0_75',
 # 'oid_glavred_scores_suicide_score_0_9',
 # 'oid_glavred_scores_suicide_score_1',
 # 'oid_glavred_scores_suicide_score_avg',
 'oid_is_gid',
 # 'oid_originality',
 # 'oid_pickcher_scores_porno_score_0',
 # 'oid_pickcher_scores_porno_score_0_1',
 # 'oid_pickcher_scores_porno_score_0_25',
 # 'oid_pickcher_scores_porno_score_0_5',
 # 'oid_pickcher_scores_porno_score_0_75',
 # 'oid_pickcher_scores_porno_score_0_9',
 # 'oid_pickcher_scores_porno_score_1',
 # 'oid_pickcher_scores_porno_score_avg',
 # 'oid_size_log',
 # 'oid_subject_id_0',
 # 'oid_subject_id_1',
 # 'oid_subject_id_10',
 # 'oid_subject_id_1001',
 # 'oid_subject_id_1002',
 # 'oid_subject_id_11',
 # 'oid_subject_id_12',
 # 'oid_subject_id_13',
 # 'oid_subject_id_14',
 # 'oid_subject_id_15',
 # 'oid_subject_id_16',
 # 'oid_subject_id_17',
 # 'oid_subject_id_18',
 # 'oid_subject_id_19',
 # 'oid_subject_id_2',
 # 'oid_subject_id_20',
 # 'oid_subject_id_21',
 # 'oid_subject_id_22',
 # 'oid_subject_id_23',
 # 'oid_subject_id_24',
 # 'oid_subject_id_25',
 # 'oid_subject_id_26',
 # 'oid_subject_id_27',
 # 'oid_subject_id_28',
 # 'oid_subject_id_29',
 # 'oid_subject_id_3',
 # 'oid_subject_id_30',
 # 'oid_subject_id_31',
 # 'oid_subject_id_32',
 # 'oid_subject_id_33',
 # 'oid_subject_id_34',
 # 'oid_subject_id_35',
 # 'oid_subject_id_36',
 # 'oid_subject_id_37',
 # 'oid_subject_id_38',
 # 'oid_subject_id_39',
 # 'oid_subject_id_4',
 # 'oid_subject_id_40',
 # 'oid_subject_id_41',
 # 'oid_subject_id_42',
 # 'oid_subject_id_5',
 # 'oid_subject_id_6',
 # 'oid_subject_id_7',
 # 'oid_subject_id_8',
 # 'oid_subject_id_9'
 ]






