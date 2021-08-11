import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers

import seaborn as sns

pd.options.display.max_columns = 25


def df_to_dataset(dataframe_f, target_name_f, shuffle_f=True, batch_size_f=32):
    dataframe_f = dataframe_f.copy()
    labels_f = dataframe_f.pop(target_name_f)
    ds_f = tf.data.Dataset.from_tensor_slices((dict(dataframe_f), labels_f))
    if shuffle_f:
        ds_f = ds_f.shuffle(buffer_size=len(dataframe_f))
    ds_f = ds_f.batch(batch_size_f)
    # ds_f = ds_f.prefetch(batch_size_f)
    return ds_f


def specify_feature_cols(df_f, numeric_feature_names_f=[], bucketized_feature_names_f={}, categorical_feature_names_f=[],
                         embedding_feature_names_f={}, crossed_feature_names_f={}):
    feature_columns_f = []

    # convert numeric
    for c in numeric_feature_names_f:
        feature_columns_f.append(feature_column.numeric_column(c))

    # convert bucketized
    for k, v in bucketized_feature_names_f.items():
        feature_columns_f.append(feature_column.bucketized_column(feature_column.numeric_column(k), boundaries=v))

    # convert categorical
    for c in categorical_feature_names_f:
        feature_columns_f.append(
            feature_column.indicator_column(feature_column.categorical_column_with_vocabulary_list(c, df_f[c].unique())))

    # convert embedding
    for k, v in embedding_feature_names_f.items():
        feature_columns_f.append(
            feature_column.embedding_column(feature_column.categorical_column_with_vocabulary_list(k, df_f[k].unique()),
                                            dimension=v))

    # convert crossed
    for f in crossed_feature_names_f:
        feature_columns_f.append(
            feature_column.indicator_column(feature_column.crossed_column(f[0], hash_bucket_size=f[1])))

    return feature_columns_f


###### IMPORT DEV DATA SET ######
import os
fp_input = 'output'
fn_input = 'deep_learning_data.csv'
df = pd.read_csv(os.path.join(fp_input, fn_input))
########

# ----- Define parameters -----
target_name = 'outcome'     # 'preflop_fold_TF'
use_observation_conds = [{'col_name': 'preflop_fold_TF', 'col_use_val': False}]     # for predicting probability of winning the hand during a preflop decision, only use hands where the player chose to play
batch_size = 32
epochs = 10
optimizer = 'adam'
loss = tf.keras.losses.MSE  # tf.keras.losses.BinaryCrossentropy(from_logits=True) #
activation = 'relu'  # 'relu'

metrics = ['accuracy']

colnames = ['game', 'hand', 'player', 'slansky', 'seat',
              'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise',
              'outcome_previous', 'human_player_TF',
              'outcome_previous_cat',
              'blind_only_outcome_previous_TF', 'zero_or_blind_only_outcome_previous_TF', 'zero_or_small_outcome_previous_TF',
              'loss_outcome_xonlyblind_previous_TF', 'loss_outcome_large_previous_TF',
              'win_outcome_xonlyblind_previous_TF', 'win_outcome_large_previous_TF',
              'preflop_fold_TF']
numeric_feature_names = ['slansky'] #, 'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise']   # ['game', 'hand', 'slansky',
                         # 'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise',
                         # 'outcome_previous', 'human_player_TF',
                         # 'blind_only_outcome_previous_TF',
                         # 'zero_or_blind_only_outcome_previous_TF', 'zero_or_small_outcome_previous_TF',
                         # 'loss_outcome_xonlyblind_previous_TF',
                         # 'loss_outcome_large_previous_TF', 'win_outcome_xonlyblind_previous_TF',
                         # 'win_outcome_large_previous_TF'
                         # ]
bucketized_feature_names = {}   # {feature_name: [list of bucket boundaries]}
categorical_feature_names = ['seat', 'player']      # ['seat', 'player', 'outcome_previous_cat']
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
# --- split into test, train, evaluation
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

train, test = train_test_split(df[select_feature_names + [target_name]], test_size=0.2)
train, val = train_test_split(train, test_size=0.2)
print('%d train examples' % (len(train)))
print('%d validation examples' % (len(val)))
print('%d test examples' % (len(test)))

# --- create formatted feature columns
feature_columns = specify_feature_cols(df_f=df, numeric_feature_names_f=numeric_feature_names,
                                       categorical_feature_names_f=categorical_feature_names,
                                       crossed_feature_names_f=crossed_feature_names)
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# --- create keras datasets
train_ds = df_to_dataset(dataframe_f=train, target_name_f=target_name, batch_size_f=batch_size)
val_ds = df_to_dataset(dataframe_f=val, target_name_f=target_name, shuffle_f=False, batch_size_f=batch_size)
test_ds = df_to_dataset(dataframe_f=test, target_name_f=target_name, shuffle_f=False, batch_size_f=batch_size)

print('Number of features: %d' % feature_layer(next(iter(train_ds))[0]).numpy().shape[1])

# sns.pairplot(train_ds)

# ------ TRAIN MODEL -----
model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation=activation),
    layers.Dense(128, activation=activation),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])  # last layer has to have sigmoid activation so that predictions are between 0 and 1

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=metrics)
########
for feature_batch, label_batch in train_ds.take(1):
    # t_select_feature = 'player' #'blind_only_outcome_previous_TF'
    print('Features: ')
    for f in list(feature_batch.keys()):
        print(f)
    # print('A batch of %s: %s' % (t_select_feature, feature_batch[t_select_feature]))
    print('A batch of targets: ', label_batch)
########

print('Included variables')
for t in model.get_config()['layers'][0]['config']['feature_columns']:
    try:
        print('%s type: %s' % (t['class_name'], t['config']['key']))
    except KeyError:
        try:
            print('%s type: %s' % (t['class_name'], t['config']['categorical_column']['config']['keys']))
        except KeyError:
            print('%s type: %s' % (t['class_name'], t['config']['categorical_column']['config']['key']))


model.fit(train_ds,
          validation_data=val_ds,
          epochs=epochs)

# --- evaluate fit
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print('\nTest accuracy: %3.1f%%' % (test_acc * 100))
print('\nBase accuracy (assign all "1"/"0"): %3.1f%% / %3.1f%%' % (test[target_name].sum()/test[target_name].shape[0] * 100, (1-test[target_name].sum()/test[target_name].shape[0])* 100))

# ----- INFERENCE -----
from matplotlib import pyplot as plt
import numpy as np

inference_ds_name = 'test'
all_data_sets = {'train': train_ds, 'val': val_ds, 'test': test_ds}

# make predictions for test set
inference_probabilities = np.array([])
inference_predictions = np.array([])
inference_labels = np.array([])
for t_feat, t_label in all_data_sets[inference_ds_name]:
    t_pred = model.predict(t_feat)
    t_len = len(t_pred)
    inference_probabilities = np.concatenate([inference_probabilities, np.array(t_pred).reshape(t_len,)])
    inference_predictions = np.concatenate([inference_predictions, np.array((t_pred > 0.5).astype("int32")).reshape(t_len,)])
    inference_labels = np.concatenate([inference_labels, np.array(t_label)])
    del t_pred, t_len
del t_feat, t_label

# generate histrogram of predicted probabilities
plt.hist(inference_probabilities)
plt.title("Predicted probabilities of " + target_name + ' (' + inference_ds_name + ' data set)')

# plot actual v predicted prob
plt.scatter(inference_probabilities, inference_labels)

# calculate confusion matrix
from sklearn.metrics import confusion_matrix
print('--- Confusion matrix for %s (row = actual, col = pred) ---' % inference_ds_name)
print(confusion_matrix(inference_labels, inference_predictions))

