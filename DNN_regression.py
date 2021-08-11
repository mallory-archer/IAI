import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

import seaborn as sns

pd.options.display.max_columns = 50


# ----- FUNCTIONS -----
def plot_loss(log):
    plt.plot(log.history['loss'], label='loss')
    plt.plot(log.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel(['Error'])
    plt.legend()
    plt.grid(True)


def avp_plot(actual_f, predicted_f):
    a = plt.axes(aspect='equal')
    plt.scatter(actual_f, predicted_f)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    lims = [min(min(actual_f), min(predicted_f)), max(max(actual_f), max(predicted_f))]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)


###### IMPORT DEV DATA SET ######
import os
fp_input = 'output'
fn_input = 'deep_learning_data.csv'
df = pd.read_csv(os.path.join(fp_input, fn_input))

# --- add features ----
df = pd.merge(df, df.loc[~df['preflop_fold_TF']].groupby(['game', 'hand']).player.nunique().reset_index().rename(columns={'player': 'tot_num_players_play_preflop'}), how='left', on=['game', 'hand'])

df['win_TF'] = df.outcome > 0
########

# ----- Define parameters -----
target_name = 'outcome'     # 'preflop_fold_TF'
use_observation_conds = [{'col_name': 'preflop_fold_TF', 'col_use_val': False}]     # for predicting probability of winning the hand during a preflop decision, only use hands where the player chose to play

colnames = ['game', 'hand', 'player', 'slansky', 'seat',
              'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise',
              'outcome_previous', 'human_player_TF',
              'outcome_previous_cat',
              'blind_only_outcome_previous_TF', 'zero_or_blind_only_outcome_previous_TF', 'zero_or_small_outcome_previous_TF',
              'loss_outcome_xonlyblind_previous_TF', 'loss_outcome_large_previous_TF',
              'win_outcome_xonlyblind_previous_TF', 'win_outcome_large_previous_TF',
              'preflop_fold_TF']

numeric_feature_names = ['slansky', 'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise',
                         'tot_num_players_play_preflop']    #, 'preflop_hand_num_raise']   # ['game', 'hand', 'slansky',
                         # 'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise',
                         # 'outcome_previous', 'human_player_TF',
                         # 'blind_only_outcome_previous_TF',
                         # 'zero_or_blind_only_outcome_previous_TF', 'zero_or_small_outcome_previous_TF',
                         # 'loss_outcome_xonlyblind_previous_TF',
                         # 'loss_outcome_large_previous_TF', 'win_outcome_xonlyblind_previous_TF',
                         # 'win_outcome_large_previous_TF'
                         # ]
bucketized_feature_names = {}   # {feature_name: [list of bucket boundaries]}
categorical_feature_names = ['seat', 'win_TF']      # ['seat', 'player', 'outcome_previous_cat']
embedding_feature_names = {}    # {feature_name: int_for_num_dimensions}
crossed_feature_names = []  #[['seat', 'player']] #   [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]      # dictionary of list of tuples: [([feature1_name, feature2_name], int_hash_bucket_size)]    [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]


# --- CALCS ----
select_feature_names = list(set(numeric_feature_names + list(bucketized_feature_names.keys()) + categorical_feature_names + list(embedding_feature_names.keys()) + [item for sublist in crossed_feature_names for item in sublist]))
for i in range(len(crossed_feature_names)):
    crossed_feature_names[i] = (crossed_feature_names[i], len(df[crossed_feature_names[i][0]].unique()) * len(df[crossed_feature_names[i][1]].unique()))

print('--Data segmentation--')
print('Number of observations meet conditions:')
use_obs_TF = pd.Series(index=df.index, data=True)
for c in use_observation_conds:
    t_vec = (df[c['col_name']] == c['col_use_val'])
    use_obs_TF = use_obs_TF & t_vec
    print('\t%s = %s: %d' % (c['col_name'], c['col_use_val'], t_vec.sum()))
    del t_vec
print('%d observations meet all conditions for use' % use_obs_TF.sum())

# ----- PREPROCESS -----
# ----- Drop rows with missing data or target value
print('\n---Missing values cleanup---')
print('%d observations, %d missing value for target name %s' % (len(df), df[target_name].isnull().sum(), target_name))
print('%d observations, %d missing value for selected features' % (len(df), df[select_feature_names].isnull().apply('any', axis=1).sum()))
print('\tNumber of observations missing select feature:')
for c in select_feature_names:
    t_num_missing = df[c].isnull().sum()
    if t_num_missing > 0:
        print('\t\t%s: %d' % (c, t_num_missing))
    else:
        print('\t\t%s: --' % (c))
del c, t_num_missing

print('\n--- Data set modification summary ---')
df.drop(df.index[df[select_feature_names].isnull().apply('any', axis=1) | df[target_name].isnull()], inplace=True)
print('%d observations remain after dropping observations with missing values for target OR select features' % (len(df)))
df.drop(use_obs_TF.index[~use_obs_TF], inplace=True)  # drop data not used in model due to segmentation
print('%d observations remain after dropping observations flagged for exclusion due to data set segmentation' % (len(df)))

# ----- Convert categorical values to dummies
for c in categorical_feature_names:
    select_feature_names.pop(select_feature_names.index(c))
    select_feature_names = select_feature_names + [c + '_' + str(x) for x in df[c].unique()]

df = pd.get_dummies(df, columns=categorical_feature_names, prefix=categorical_feature_names, prefix_sep='_')

# --- split into test, train, evaluation
train, test = train_test_split(df[select_feature_names + [target_name]], test_size=0.2)
# train, val = train_test_split(train, test_size=0.2)
print('%d train examples' % (len(train)))
# print('%d validation examples' % (len(val)))
print('%d test examples' % (len(test)))

# sns.pairplot(train[[target_name, 'slansky']], diag_kind='kde')

train_features = train.copy()
test_features = test.copy()

train_labels = train_features.pop(target_name)
test_labels = test_features.pop(target_name)

# ----- scale features
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())


# ----- fit model
def build_and_compile_model(data_f):
    model_f = tf.keras.Sequential([
        data_f,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model_f.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model_f


model = build_and_compile_model(normalizer)     # model = tf.keras.Sequential([normalizer, layers.Dense(units=1)])
model.summary()
model_train_log = model.fit(
    train_features, train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

# --- Examine fitting procedure
# df_log = pd.DataFrame(model_train_log.history)
# df_log['epoch'] = model_train_log.epoch
# df_log.head()
# test_results['basic'] = model.evaluate(test_features, test_labels, verbose=0)   # save summary

# graphical
plot_loss(model_train_log)
avp_plot(test_labels, model.predict(test_features))
