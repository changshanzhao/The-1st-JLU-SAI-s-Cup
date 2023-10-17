import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction import FeatureHasher
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
df = pd.read_csv("train_data.csv")
item_watched_times_sum = [0.] * 3131
item_watched_times_count = [0.] * 3131
item_feature_dict = {i+1: [] for i in range(3131)}
user_watched_items_dict = {i+1: [] for i in range(1326)}
mlb = MultiLabelBinarizer(classes=[str(i) for i in range(31)])
appended = [0.] * 3131
l_user_id = [0.] * 1326
# feature_item_dict = {i: [] for i in range(31)}

for iter, row in df.iterrows():
    user_id = row['user_id']
    l_user_id.append(str(user_id))
    item_id = row['item_id']
    watched_times = row['watched_times']
    item_feature = row['item_feature']
    item_watched_times_sum[item_id] += watched_times
    item_watched_times_count[item_id] += 1
    user_watched_items_dict[user_id].append(item_id)
    if appended[item_id] == 0:
        item_feature = item_feature.replace('[', '')
        item_feature = item_feature.replace(']', '')
        t = item_feature.split(', ')
        for tt in t:
            item_feature_dict[item_id].append(tt)
        appended[item_id] = 1

# for i in range(1,3132):
#     for t in item_feature_dict[i]:
#         feature_item_dict[eval(t)].append(i)


y = df['watched_times']
t_list = []
for i in item_feature_dict:
    t_list.append(tuple(item_feature_dict[i]))
one_hot_f = mlb.fit_transform(t_list)
one_hot_df = pd.DataFrame(one_hot_f, columns=[f'feature_{i+1}' for i in range(31)])
merged_df = pd.merge(df, one_hot_df, left_on='item_id', right_index=True)
grouped = merged_df.groupby('user_id')
arrays = [grouped.get_group(user_id) for user_id in grouped.groups]
list = []
for i in range(1326):
    t = pd.DataFrame(arrays[i])
    # x_1 = t.iloc[:, lambda t: [3, 4]]
    x_2 = t.iloc[:, 6:]
    # x = pd.concat([x_1, x_2], axis=1)
    y = t.iloc[:, 5]
    model = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10, oob_score=True)
    model.fit(x_2, y)
    with open('model/model_'+str(i)+'.pkl', 'wb') as file:
        pickle.dump(model, file)

for i in range(1, 3131):
    item_watched_times_sum[i] = item_watched_times_sum[i] / item_watched_times_count[i]
item_watched_times_sum = np.array(item_watched_times_sum)
idx = np.argsort(-item_watched_times_sum)

ans = {
    'id': [],
    'user_id': [],
    'recommend_list': []
}

for user_id in range(1, 1327):
    ans['id'].append(user_id - 1)
    ans['user_id'].append(user_id)
    pointer = 0
    recommend_list = []
    recommend_score_list = []
    while pointer < 40:
        if idx[pointer] not in user_watched_items_dict[user_id]:
            with open('model/model_'+str(user_id - 1)+'.pkl', 'rb') as file:
                model = pickle.load(file)
            recommend_score_list.append(model.predict(pd.DataFrame(one_hot_df.iloc[idx[pointer]-1]).transpose())[0])
        else:
            pass
        pointer += 1

    recommend_score_list = np.array(recommend_score_list)
    idx_ = np.argsort(-recommend_score_list)
    num = 0
    while num < 10:
        recommend_list.append(idx[idx_[num]])
        num += 1
    ans['recommend_list'].append(recommend_list)

dataframe = pd.DataFrame(ans)
dataframe.to_csv("data/PopItem.csv", index=False)
