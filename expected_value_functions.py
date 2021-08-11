import os
import copy
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


# =========== DEFINE FUNCTIONS ========
def import_dataset(fp_input='output', fn_input='deep_learning_data.csv'):
    return pd.read_csv(os.path.join(fp_input, fn_input))


def create_feature_layer(df_f, numeric_feature_names_f=[], bucketized_feature_names_f={}, categorical_feature_names_f=[],
                         embedding_feature_names_f={}, crossed_feature_names_f={}):
    feature_columns_f = []

    # convert numeric
    for c in numeric_feature_names_f:
        feature_columns_f.append(tf.feature_column.numeric_column(c))

    # convert bucketized
    for k, v in bucketized_feature_names_f.items():
        feature_columns_f.append(tf.feature_column.bucketized_column(tf.feature_column.numeric_column(k), boundaries=v))

    # convert categorical
    for c in categorical_feature_names_f:
        feature_columns_f.append(
            tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(c, df_f[c].unique())))

    # convert embedding
    for k, v in embedding_feature_names_f.items():
        feature_columns_f.append(
            tf.feature_column.embedding_column(tf.feature_column.categorical_column_with_vocabulary_list(k, df_f[k].unique()),
                                            dimension=v))

    # convert crossed
    for f in crossed_feature_names_f:
        feature_columns_f.append(
            tf.feature_column.indicator_column(tf.feature_column.crossed_column(f[0], hash_bucket_size=f[1])))

    # create a list of column names
    select_feature_names_f = list(set(
        numeric_feature_names_f + list(bucketized_feature_names_f.keys()) + categorical_feature_names_f + list(
            embedding_feature_names_f.keys()) + [item for sublist in crossed_feature_names_f for item in sublist]))
    for i in range(len(crossed_feature_names_f)):
        crossed_feature_names_f[i] = (crossed_feature_names_f[i], len(df[crossed_feature_names_f[i][0]].unique()) * len(
            df[crossed_feature_names_f[i][1]].unique()))

    # print feature column summary
    print('\n---Missing values cleanup---')
    print('%d observations, %d missing value for selected features' % (
        len(df_f), df_f[select_feature_names_f].isnull().apply('any', axis=1).sum()))
    print('\tNumber of observations missing select feature:')
    for c in select_feature_names_f:
        t_num_missing = df_f[c].isnull().sum()
        if t_num_missing > 0:
            print('\t\t%s: %d' % (c, t_num_missing))
        else:
            print('\t\t%s: --' % c)
    del c, t_num_missing

    return feature_columns_f, select_feature_names_f


def drop_observations(df_f, select_feature_names_f, target_name_f, obs_use_cond_f):
    df_f = copy.copy(df_f)
    print('--Data segmentation--')
    print('Number of observations meet usage conditions:')
    use_obs_TF_f = pd.Series(index=df.index, data=True)
    for c in obs_use_cond_f:
        t_vec = (df[c['col_name']] == c['col_use_val'])
        use_obs_TF_f = use_obs_TF_f & t_vec
        print('\t%s = %s: %d' % (c['col_name'], c['col_use_val'], t_vec.sum()))
        del t_vec
    print('%d observations meet all conditions for use' % use_obs_TF_f.sum())
    
    print('\n--- Data set modification summary ---')
    df_f.drop(df_f.index[df_f[select_feature_names_f].isnull().apply('any', axis=1) | df_f[target_name_f].isnull()], inplace=True)
    print('%d observations remain after dropping observations with missing values for target OR select features' % (
        len(df_f)))
    df_f.drop(use_obs_TF_f.index[~use_obs_TF_f], inplace=True)  # drop data not used in model due to segmentation
    print('%d observations remain after dropping observations flagged for exclusion due to data set segmentation' % (
        len(df_f)))

    return df_f


def df_to_dataset(df_f, target_name_f, batchsize_f=None):
    df_f = df_f.copy()
    labels_f = df_f.pop(target_name_f)
    ds_f = tf.data.Dataset.from_tensor_slices((dict(df_f), labels_f))
    if batchsize_f is None:
        return ds_f.batch(df_f.shape[0])
    else:
        return ds_f.batch(batchsize_f)


def build_and_compile_model(feature_layer_f, width_hidden_layer_f, activation_hidden_f, activation_output_f, optimizer_f, loss_f, metric_f=None):
    model_f = tf.keras.Sequential([
        feature_layer_f,
        tf.keras.layers.Dense(width_hidden_layer_f, activation=activation_hidden_f),
        tf.keras.layers.Dense(width_hidden_layer_f, activation=activation_hidden_f),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1, activation=activation_output_f)
    ])  # last layer has to have sigmoid activation so that predictions are between 0 and 1

    if metric_f is not None:
        model_f.compile(optimizer=optimizer_f,
                        loss=loss_f,
                        metrics=metric_f)
    else:
        model_f.compile(optimizer=optimizer_f,
                        loss=loss_f)

    print('Included variables')
    for t in model_f.get_config()['layers'][0]['config']['feature_columns']:
        try:
            print('%s type: %s' % (t['class_name'], t['config']['key']))
        except KeyError:
            try:
                print('%s type: %s' % (t['class_name'], t['config']['categorical_column']['config']['keys']))
            except KeyError:
                print('%s type: %s' % (t['class_name'], t['config']['categorical_column']['config']['key']))

    return model_f


# =========== DEFINE PARAMETERS ===========
# --- features
target_name = 'outcome' # 'win_TF'
numeric_feature_names = ['slansky', 'preflop_hand_num_raise', 'preflop_hand_tot_amount_raise', 'preflop_num_final_participants', 'win_TF']  #
bucketized_feature_names = {}   # {feature_name: [list of bucket boundaries]}
categorical_feature_names = ['seat']      # ['seat', 'player', 'outcome_previous_cat']
embedding_feature_names = {}    # {feature_name: int_for_num_dimensions}
crossed_feature_names = []  #[['seat', 'player']] #   [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]      # dictionary of list of tuples: [([feature1_name, feature2_name], int_hash_bucket_size)]    [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]

# --- segmenting
obs_use_conds = [{'col_name': 'preflop_fold_TF', 'col_use_val': False}]  #[{'col_name': 'preflop_fold_TF', 'col_use_val': False}]     # for predicting probability of winning the hand during a preflop decision, only use hands where the player chose to play
test_set_size = 0.2

# --- model training
probability_model_params = {'width_hidden_layer_f': 128,
                            'activation_hidden_f': 'relu',
                            'activation_output_f': 'sigmoid',
                            'optimizer_f': 'adam',
                            'loss_f': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            'metric_f': 'accuracy'}
payoff_model_params = {'width_hidden_layer_f': 64,
                       'activation_hidden_f': 'relu',
                       'activation_output_f': 'sigmoid',
                       'optimizer_f': tf.keras.optimizers.Adam(0.001),
                       'loss_f': 'mean_squared_error'}     # activation_output_f and mean_squared_error not included in regression example code, may need to delete these?
epochs = 100

# =========== IMPORT DATA SET ==========
df = import_dataset()

# =========== ADD/PREPROCESS FEATURES ==========
df['win_TF'] = df.outcome > 0   # ###### move this to upstream data processing

feature_layer, select_feature_names = create_feature_layer(df,
                                                           numeric_feature_names_f=numeric_feature_names,
                                                           categorical_feature_names_f=categorical_feature_names,
                                                           crossed_feature_names_f=crossed_feature_names)
feature_layer = tf.keras.layers.DenseFeatures(feature_layer)

# =========== SPLIT INTO TEST/TRAIN ==========
# --- drop observations with missing data or that do not match specified criteria (data segmentation)
df_model = drop_observations(df_f=df, select_feature_names_f=select_feature_names, target_name_f=target_name, obs_use_cond_f=obs_use_conds)

# --- split data sets
train, test = train_test_split(df[select_feature_names + [target_name]], test_size=test_set_size)
train, val = train_test_split(train, test_size=test_set_size)

# --- train model
model = build_and_compile_model(feature_layer, **payoff_model_params)
model_train_log = model.fit(df_to_dataset(train, target_name),
                            validation_data=df_to_dataset(val, target_name),
                            epochs=epochs)
model.summary()

# --- evaluate fit
# categorical
test_loss, test_acc = model.evaluate(df_to_dataset(test, target_name), verbose=2)
print('\nTest accuracy: %3.1f%%' % (test_acc * 100))
print('\nBase accuracy (assign all "1"/"0"): %3.1f%% / %3.1f%%' % (test[target_name].sum()/test[target_name].shape[0] * 100, (1-test[target_name].sum()/test[target_name].shape[0])* 100))

# regression
import matplotlib.pyplot as plt


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


plot_loss(model_train_log)
avp_plot(test[target_name], model.predict(df_to_dataset(test, target_name)))


# fit payment amount model

# calc expected value
