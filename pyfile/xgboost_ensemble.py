# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import xgboost as xgb
import os
from sklearn.utils import class_weight

list_file = [
           'mixed_inputs_resize.csv',
           'mixed_inputs_resize_hsv.csv',
           'mixed_inputs_mask.csv',
           'mixed_inputs_mask_hsv.csv',
]

# merge test feature files
test_feature_path = '~/Dropbox/RoofMaterial/submission/'
test_feature_df = pd.read_csv(os.path.join(test_feature_path, '../submission_format.csv'))['id']
for test_file in list_file:
    test_feature_temp = pd.read_csv(os.path.join(test_feature_path, test_file))
    test_feature_df = pd.merge(test_feature_df, test_feature_temp, on = 'id')

test_feature_temp = pd.read_csv(os.path.join(test_feature_path, '../rawdata/image_resize/test/test_data_with_dist.csv'))
test_feature_temp = pd.concat([test_feature_temp, pd.get_dummies(test_feature_temp.place)], axis = 1)
test_feature_temp = test_feature_temp[['id',
                                        'log_building_area',
                                        'building_vertices',
                                        'building_width',
                                        'building_height',
                                        'R_mean',
                                        'G_mean',
                                        'B_mean',
                                        'R_std',
                                        'G_std',
                                        'B_std',
                                        'H_mean',
                                        'S_mean',
                                        'V_mean',
                                        'H_std',
                                        'S_std',
                                        'V_std',
                                        'distance_10_all',
                                        'distance_20_all',
                                        'distance_50_all',
                                        'healthy_10_train',
                                        'incomplete_10_train',
                                        'irregular_10_train',
                                        'other_10_train',
                                        'concrete_20_train',
                                        'healthy_20_train',
                                        'incomplete_20_train',
                                        'irregular_20_train',
                                        'other_20_train',
                                        'concrete_50_train',
                                        'healthy_50_train',
                                        'incomplete_50_train',
                                        'irregular_50_train',
                                        'other_50_train',
                                        'borde_rural',
                                        'borde_soacha',
                                        'dennery',
                                        'mixco_1_and_ebenezer',
                                        'mixco_3'
                                        ]]

test_feature_df = pd.merge(test_feature_df, test_feature_temp, on = 'id')
test_feature = test_feature_df.drop(['id'], axis = 1).to_numpy()

# merge training feature files
train_feature_path = '~/Dropbox/RoofMaterial/feature/'
train_feature_df = pd.read_csv(os.path.join(train_feature_path, '../train_labels.csv'))['id']
for train_file in list_file:
    train_feature_temp = pd.read_csv(os.path.join(train_feature_path, train_file))
    train_feature_df = pd.merge(train_feature_df, train_feature_temp, on = 'id')

train_feature_temp = pd.read_csv(os.path.join(train_feature_path, '../rawdata/image_resize/train/train_data_with_dist.csv'))
train_feature_temp = pd.concat([train_feature_temp, pd.get_dummies(train_feature_temp.place)], axis = 1)
train_feature_temp = train_feature_temp[['id',
                                        'log_building_area',
                                        'building_vertices',
                                        'building_width',
                                        'building_height',
                                        'R_mean',
                                        'G_mean',
                                        'B_mean',
                                        'R_std',
                                        'G_std',
                                        'B_std',
                                        'H_mean',
                                        'S_mean',
                                        'V_mean',
                                        'H_std',
                                        'S_std',
                                        'V_std',
                                        'distance_10_all',
                                        'distance_20_all',
                                        'distance_50_all',
                                        'healthy_10_train',
                                        'incomplete_10_train',
                                        'irregular_10_train',
                                        'other_10_train',
                                        'concrete_20_train',
                                        'healthy_20_train',
                                        'incomplete_20_train',
                                        'irregular_20_train',
                                        'other_20_train',
                                        'concrete_50_train',
                                        'healthy_50_train',
                                        'incomplete_50_train',
                                        'irregular_50_train',
                                        'other_50_train',
                                        'borde_rural',
                                        'borde_soacha',
                                        'dennery',
                                        'mixco_1_and_ebenezer',
                                        'mixco_3'
                                        ]]

train_feature_df = pd.merge(train_feature_df, train_feature_temp, on = 'id')
train_feature = train_feature_df.drop(['id'], axis = 1).to_numpy()

# training labels
train_labels = pd.read_csv(os.path.join(train_feature_path, '../train_labels.csv'))
train_labels = train_labels[train_labels['verified'] == True]
train_labels = train_labels[['id', 'concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']]

train_labels['label'] = ''
train_labels['label'][train_labels.concrete_cement == 1.0] = 'concrete_cement'
train_labels['label'][train_labels.healthy_metal == 1.0] = 'healthy_metal'
train_labels['label'][train_labels.incomplete == 1.0] = 'incomplete'
train_labels['label'][train_labels.irregular_metal == 1.0] = 'irregular_metal'
train_labels['label'][train_labels.other == 1.0] = 'other'

train_labels = train_labels[['label']]

# XGBoost

sample_weights = class_weight.compute_sample_weight('balanced',
                                                    train_labels.label)

model = xgb.XGBClassifier(n_estimators = 150)

model.fit(train_feature, train_labels.label.to_numpy(),
        sample_weight = sample_weights,
          eval_set=[(train_feature, train_labels.label.to_numpy())],
          eval_metric = 'mlogloss',
          verbose = True,
          )

pred = model.predict_proba(test_feature)

pred_pd = pd.DataFrame({'id': test_feature_df.id})
pred_pd['concrete_cement'] = pred[:, 0]
pred_pd['healthy_metal'] = pred[:, 1]
pred_pd['incomplete'] = pred[:, 2]
pred_pd['irregular_metal'] = pred[:, 3]
pred_pd['other'] = pred[:, 4]

submission_file_name = 'xgboost_ensemble.csv'
pred_pd.to_csv(os.path.join(test_feature_path, submission_file_name), index = False)


