import numpy as np

#given vector of strings like ['2:0.306008', '3:0.000450', '4:0.077048', '5:0.230439', '6:0.386055', '1:1.000000'],
#extract features into numpy vector
def extract_features(strv):
    feat = np.array([0.0]*6)
    for i in range(6):
        tmp = strv[i].split(":")
        feat_index = int(tmp[0])
        feat_value = float(tmp[1])
        feat[feat_index-1] = feat_value
    return(feat)
    

#returns timestamp, offered ad id, click (binary), user features, vector of eligible ad ids, vector of eligible ad id features
def parse_line(line):
    #remove \n at the end of the line
    line = line.strip()
    line = line.split("|")
  
    #get timestamp, offered ad id, click (binary)
    decision_info = line[0].split(" ")
    timestamp = int(decision_info[0])
    offered_ad_id = int(decision_info[1])
    click = int(decision_info[2])

    #get user features
    user_info = line[1].split(" ")[1:]
    user_feat = extract_features(user_info)

    #get eligible ads + features
    ad_info = line[2].split(" ")
    eligible_ads_ids = np.array([int(ad_info[0])])
    eligible_ads_feat = np.array([extract_features(ad_info[1:])])
    for i in range(3, len(line)):
        ad_info = line[i].split(" ")
        if len(ad_info[1:]) >= 6:
            eligible_ads_ids = np.append(eligible_ads_ids,np.array([int(ad_info[0])]))
            eligible_ads_feat = np.append(eligible_ads_feat,np.array([extract_features(ad_info[1:])]), axis=0)
        #else:
            #print('Ad Feature Amomaly: ' + line[i])
            
    sorted_inds = np.argsort(eligible_ads_ids)
    eligible_ads_ids = eligible_ads_ids[sorted_inds]
    eligible_ads_feat = eligible_ads_feat[sorted_inds]
  
    return(timestamp, offered_ad_id, click, user_feat, eligible_ads_ids, eligible_ads_feat)

#Read Data
path="/R6/ydata-fp-td-clicks-v1_0.20090501"


   
 
