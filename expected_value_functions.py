import os
import copy
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from statsmodels.api import OLS

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

pd.options.display.max_columns = 25


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
    def calc_num_crossed_features(f_f):
        return len(df[f_f[0]].unique()) * len(df[f_f[1]].unique())

    for f in crossed_feature_names_f:
        print(f)
        feature_columns_f.append(
            tf.feature_column.indicator_column(tf.feature_column.crossed_column(f, hash_bucket_size=calc_num_crossed_features(f))))

    # create a list of column names
    select_feature_names_f = list(set(
        numeric_feature_names_f + list(bucketized_feature_names_f.keys()) + categorical_feature_names_f + list(
            embedding_feature_names_f.keys()) + [item for sublist in crossed_feature_names_f for item in sublist]))

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
    df_f.drop(use_obs_TF_f.index[~use_obs_TF_f], inplace=True, errors='ignore')  # drop data not used in model due to segmentation
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
        tf.keras.layers.Dropout(0.4),
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


def plot_loss(log):
    plt.plot(log.history['loss'], label='loss')
    plt.plot(log.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel(['Error'])
    plt.legend()
    plt.grid(True)


def calc_pot_amount(df_f):
    return df_f.loc[df.preflop_play_TF & ~df.win_TF, 'preflop_pot_stake'].sum() + (-1 * df_f.loc[~df.preflop_play_TF, 'outcome'].sum())


def create_pred_dict(df_f, fp_output_f, fn_output_f, save_TF=False):
    pred_dict_f = dict(zip(df_f.player.unique(), [{} for i in range(len(df_f.player.unique()))]))
    for p in df_f.player.unique():
        df_f_p = df_f.loc[df_f.player == p]
        pred_dict_f.update(
            {p: dict(zip([int(x) for x in df_f_p.game.unique()], [{} for i in range(len(df_f_p.game.unique()))]))})
        for g in df_f_p.game.unique():
            df_f_pg = df_f_p.loc[df_f_p.game == g]
            pred_dict_f[p][g].update(
                {int(df_f_pg.iloc[i].hand): df_f_pg.iloc[i].prob_winning for i in range(df_f_pg.hand.nunique())})

    check_count = 0
    all_probs = list()
    for p in pred_dict_f.keys():
        for g in pred_dict_f[p].keys():
            for h in pred_dict_f[p][g]:
                check_count += 1
                all_probs.append(pred_dict_f[p][g][h])
    if check_count != df_f.shape[0]:
        print('Not all player-game-hands accounted for in predicted probabilities')
        plt.hist(all_probs)
        plt.title('All predicted probabilities')

    if save_TF:
        with open(os.path.join(fp_output_f, fn_output_f), 'w') as f:
            json.dump(pred_dict_f, fp=f)

    return pred_dict_f


def create_payoff_dict(df_f, fp_output_f, fn_output_f, save_TF=False):
    payoff_dict_f = dict(zip(df_f.player.unique(), [{} for i in range(len(df_f.player.unique()))]))
    for p in df_f.player.unique():
        # pass
        df_f_p = df_f.loc[df_f.player == p]
        payoff_dict_f.update(
            {p: dict(zip([int(x) for x in df_f_p.game.unique()], [{} for i in range(len(df_f_p.game.unique()))]))})
        for g in df_f_p.game.unique():
            # pass
            df_f_pg = df_f_p.loc[df_f_p.game == g]
            payoff_dict_f[p][g].update(
                {int(df_f_pg.iloc[i].hand): {
                    'play': {'win': df_f_pg.iloc[i].predicted_win_play, 'lose': df_f_pg.iloc[i].predicted_loss_play},
                    'fold': {'lose': df_f_pg.iloc[i].predicted_loss_fold}}
                 for i in range(df_f_pg.hand.nunique())})

    check_count = 0
    all_payoffs = list()
    for p in payoff_dict_f.keys():
        for g in payoff_dict_f[p].keys():
            for h in payoff_dict_f[p][g]:
                check_count += 1
                all_payoffs.append(payoff_dict_f[p][g][h])
    if check_count != df_f.shape[0]:
        print('Not all player-game-hands accounted for in payoff probabilities')

    if save_TF:
        with open(os.path.join(fp_output_f, fn_output_f), 'w') as f:
            json.dump(payoff_dict_f, fp=f)

    return payoff_dict_f

print('hello world')

# =========== DEFINE PARAMETERS ===========
fp_output = 'output'
fn_output_prob = 'prob_dict_dnn.json'
fn_output_payoff = 'payoff_dict_dnn.json'
save_dict_TF = False

# --- features
target_name = 'win_TF'
numeric_feature_names = ['slansky'] #, 'preflop_hand_tot_amount_raise', 'preflop_hand_num_raise', 'preflop_num_final_participants']
bucketized_feature_names = {}   # {feature_name: [list of bucket boundaries]}
categorical_feature_names = ['seat']      # ['seat', 'player', 'outcome_previous_cat']
embedding_feature_names = {}    # {feature_name: int_for_num_dimensions}
crossed_feature_names = [('player', 'seat'), ('seat', 'slansky')] #[('player', 'opponents_string')]  #[('player', 'opponent_pluribus')] #[('seat', 'player')] #[('player', 'opponents_string')] #[('player', 'opponent_pluribus')]   #   [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]      # dictionary of list of tuples: [([feature1_name, feature2_name], int_hash_bucket_size)]    [['human_player_TF', 'outcome_previous_cat'], ['seat', 'player']]

# --- segmenting
obs_use_conds = [{'col_name': 'preflop_play_TF', 'col_use_val': True},
                 {'col_name': 'preflop_num_final_participants_more_than_1', 'col_use_val': True}]  #[{'col_name': 'preflop_fold_TF', 'col_use_val': False}]     # for predicting probability of winning the hand during a preflop decision, only use hands where the player chose to play
test_set_size = 0.2

# --- model training
probability_model_params = {'width_hidden_layer_f': 128,
                            'activation_hidden_f': 'relu',
                            'activation_output_f': 'sigmoid',
                            'optimizer_f': 'adam',
                            'loss_f': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            'metric_f': ['accuracy']}
# payoff_model_params = {'width_hidden_layer_f': 64,
#                        'activation_hidden_f': 'relu',
#                        'activation_output_f': 'sigmoid',
#                        'optimizer_f': tf.keras.optimizers.Adam(0.001),
#                        'loss_f': 'mean_squared_error'}     # activation_output_f and mean_squared_error not included in regression example code, may need to delete these?
epochs = 500
patience = 25   # early stopping criteria to prevent overtraining (number of consistent increaess in valuation loss

# =========== IMPORT DATA SET ==========
df = import_dataset()

# =========== ADD/PREPROCESS FEATURES ==========
df['win_TF'] = df.outcome > 0   # ###### move this to upstream data processing
df['preflop_num_final_participants_more_than_1'] = df.preflop_num_final_participants > 1

# ---- get list of opponents
df = df.merge(df.loc[df.preflop_play_TF].groupby(['game', 'hand']).player.unique().reset_index().rename(columns={'player': 'preflop_players'}), how='left', on=['game', 'hand'])
if sum(df.apply(lambda x: len(x.preflop_players) - x.preflop_num_final_participants, axis=1)) != 0:
    print('WARNING: opponent creation feature may have errors. Check')
df['opponents'] = df.apply(lambda x: [y for y in x.preflop_players if y != x.player and x.preflop_play_TF], axis=1).apply(lambda x: None if len(x) == 0 else x)
df['opponents_string'] = df.apply(lambda x: '-'.join(x.opponents) if x.opponents is not None else '-', axis=1)
df['preflop_players_string'] = df.apply(lambda x: '-'.join(x.preflop_players), axis=1)
df['opponents_string_fold_fill'] = df.apply(lambda x: x.opponents_string if x.opponents_string != '-' else x.preflop_players_string, axis=1)
df['opponent_pluribus'] = df.apply(lambda x: int(any([y == 'Pluribus' for y in x.opponents])) if x.opponents is not None else 0, axis=1)

feature_layer, select_feature_names = create_feature_layer(df_f=df,
                                                           numeric_feature_names_f=numeric_feature_names,
                                                           categorical_feature_names_f=categorical_feature_names,
                                                           crossed_feature_names_f=crossed_feature_names)
feature_layer = tf.keras.layers.DenseFeatures(feature_layer)

# =========== SPLIT INTO TEST/TRAIN ==========
# --- drop observations with missing data or that do not match specified criteria (data segmentation)
df_model = drop_observations(df_f=df, select_feature_names_f=select_feature_names, target_name_f=target_name, obs_use_cond_f=obs_use_conds)

# --- split data sets
train, test = train_test_split(df_model[select_feature_names + [target_name]], test_size=test_set_size)
train, val = train_test_split(train, test_size=test_set_size)
print('%d train examples' % (len(train)))
print('%d validation examples' % (len(val)))
print('%d test examples' % (len(test)))

# --- train model
model = build_and_compile_model(feature_layer, **probability_model_params)
model_train_log = model.fit(df_to_dataset(train, target_name),
                            validation_data=df_to_dataset(val, target_name),
                            epochs=epochs,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)])
model.summary()
plt.figure()
plot_loss(model_train_log)
plt.title("Training error")

# --- get estimations of payoffs ---
# separate into losses and wins
# get avg amount won/lost
plt.figure()
plt.plot(df_model.slansky, df_model.outcome, 'x')
for i in df_model.slansky.unique():
    plt.plot(i, df_model.loc[df_model.slansky == i].outcome.mean(), 'r.')
    plt.plot(i, df_model.loc[df_model.slansky == i].outcome.median(), 'g.')
plt.ylim([-500, 500])
plt.title('Model estimation data by Slanksy rank')
plt.xlabel('Slansky rank')
plt.ylabel('Group outcome')
plt.legend(['mean', 'median'])

avg_win = df_model.loc[df_model.outcome > 0, 'outcome'].mean()
avg_loss = df_model.loc[df_model.outcome < 0, 'outcome'].mean()
med_win = df_model.loc[df_model.outcome > 0, 'outcome'].median()
med_loss = df_model.loc[df_model.outcome < 0, 'outcome'].median()

# --- make predictions of $ outcomes as expected value calc
test_pred = model.predict(df_to_dataset(test, target_name))
test_med_val = [x[0] * med_win + (1 - x[0]) * med_loss for x in test_pred]
test_exp_val = [x[0] * avg_win + (1 - x[0]) * avg_loss for x in test_pred]

plt.figure()
plt.hist(test_pred)
# plt.hist(test_exp_val)
# plt.hist(test_med_val)
plt.title("Predicted probabilities of winning for test set\nmean=%3.2f%%, median=%3.2f%%" % (test_pred.mean()*100, np.median(test_pred)*100))

# plot actual v predicted prob
plt.figure()
plt.scatter(test_pred, [x + np.random.normal(0, .1) for x in test[target_name]])
plt.title("Actual outcomes vs. predicted probabilities")
plt.xlabel('predicted')
plt.ylabel('actual + noise for visual separation')
plt.axvline(np.mean([test_pred[i][0] for i in range(len(test[target_name])) if test[target_name].iloc[i] == 1]), c='r')
plt.axvline(np.mean([test_pred[i][0] for i in range(len(test[target_name])) if test[target_name].iloc[i] == 0]), c='g')
plt.legend(['actual play average predictions, ' + str(round(np.mean([test_pred[i][0] for i in range(len(test[target_name])) if test[target_name].iloc[i] == 1]) * 100, 1)) + '%',
            'actual fold average predictions ' + str(round(np.mean([test_pred[i][0] for i in range(len(test[target_name])) if test[target_name].iloc[i] == 0]) * 100, 1)) + '%'])

_, axs = plt.subplots(2, 2)
axs[0, 1].scatter(df_model.loc[test.index].outcome, test_exp_val)
axs[0, 1].set_title('Actual outcomes vs. predicted outcomes')
axs[0, 1].set_xlabel('Actual outcome')
axs[0, 1].set_ylabel('Predicted E[outcome] using mean observed outcome')
axs[0, 0].hist(test_exp_val)
axs[0, 0].set_title('Histogram of predicted outcomes')
axs[1, 1].hist(df_model.loc[test.index].outcome)
axs[1, 1].set_title('Histogram of actual outcomes')

# ========== RESULTS EXAMINATION
# --- evaluate fit
# categorical
eval_params = model.evaluate(df_to_dataset(test, target_name), verbose=2)
eval_params_dict = dict(zip(['loss'] + probability_model_params['metric_f'], eval_params))
test_acc = eval_params_dict['accuracy']
test_pred = model.predict(df_to_dataset(test, target_name))
print('\nTest accuracy: %3.1f%%' % (test_acc * 100))
print('\nBase accuracy (assign all "1"/"0"): %3.1f%% / %3.1f%%' % (test[target_name].sum()/test[target_name].shape[0] * 100, (1-test[target_name].sum()/test[target_name].shape[0])* 100))

ROC_ratio = [confusion_matrix(test[target_name], test_pred > thresh)[1, 1] / confusion_matrix(test[target_name], test_pred > thresh)[0, 1] for thresh in [x/1000 for x in range(1000)]]
plt.figure()
plt.scatter([x/1000 for x in range(1000)], ROC_ratio)
plt.title('ROC curve')
thresh_optimizer = [x/1000 for x in range(1000)][ROC_ratio.index(max(ROC_ratio))]

print('--- Confusion matrix (row = actual, col = pred) at threshold = 50% ---')
print(confusion_matrix(test[target_name], test_pred > 0.5))

print('--- Confusion matrix (row = actual, col = pred) at threshold ROC optimizer = %3.2f%% ---' % (thresh_optimizer * 100))
print(confusion_matrix(test[target_name], test_pred > thresh_optimizer))

# ========== MAKE PROBABILITY PREDICTIONS FOR ENTIRE DATA SET ===============
missing_val_sub_cols = {'opponents_string_fold_fill': 'opponents_string'}
prob_pred_all = model.predict(df_to_dataset(df[[x for x in select_feature_names if x not in list(missing_val_sub_cols.values())] + [k for k, v in missing_val_sub_cols.items() if v in select_feature_names] + [target_name]].rename(columns=missing_val_sub_cols),
                                            target_name))
df = pd.concat([df, pd.Series([x[0] for x in prob_pred_all], name='prob_winning')], axis=1)

pred_dict = create_pred_dict(df, fp_output, fn_output_prob, save_TF=save_dict_TF)

plt.figure()
plt.hist(prob_pred_all)
plt.title('Histogram of predicted probabilities, all data, n=' + str(len(prob_pred_all)))

plt.figure()
plt.hist(test_pred)
plt.title('Histogram of predicted probabilities, test data, n=' + str(len(test_pred)))

# ========== PAYOFFS ===============
# if you fold preflop, whatever you lost in the outcome column is whatever you had in preflop
# if you play preflop, then you must have had in whatever the preflop tot stake was raised to

# ultimate win payoff for play is going to be correlated with how much money was on the table preflop
# money on the table preflop is sum of all the stakes of what the "play" people put in (preflop tot stake)
# plus the amount of what the "fold" people put in, i.e. what they lost in outcome

# ultimate win payoff = preflop amount on table + e = num_preflop_play * preflop stake + sum(outcome|fold)
# lose payoff = preflop_stake + e
# ----------side calculations
# --- calculate amount required to put in pot if raises occur
df = df.merge(df.loc[df.preflop_action.apply(lambda x: x.find('r') > -1), 'preflop_action'].apply(lambda x: float(x[1:])).rename('preflop_raise_amount_tot_stake'), how='left', left_index=True, right_index=True)
df['preflop_pot_stake'] = df.groupby(['game', 'hand']).preflop_raise_amount_tot_stake.transform('max')
df.drop(columns=['preflop_raise_amount_tot_stake'], inplace=True)

# ---- calculate amount required to put in pot if no raises occur and just blinds are called
df.loc[df.preflop_hand_num_raise == 0, 'preflop_pot_stake'] = 100   # if no one raises, the stake is the amount of the big blind

df_preflop_pot_amount = df.groupby(['game', 'hand']).apply(calc_pot_amount).rename('preflop_pot_tot_amount').reset_index()
df = df.merge(df_preflop_pot_amount, how='left', on=['game', 'hand'])
del df_preflop_pot_amount

# --- check to see how well outcomes correlate with preflop factors
# plt.figure()
# plt.scatter(df.loc[df.win_TF, 'preflop_pot_tot_amount'], df.loc[df.win_TF, 'outcome'])
# plt.figure()
# plt.scatter(df.loc[df.preflop_play_TF & ~df.win_TF & ((df.seat != 1) | (df.seat != 2)), 'preflop_pot_stake'], df.loc[df.preflop_play_TF & ~df.win_TF & ((df.seat != 1) | (df.seat != 2)), 'outcome'])

# --- Predict winnings -----
# --- scale
t_filter_cond = df.win_TF
t_df = df.loc[t_filter_cond & (df.outcome < 5000), ['outcome', 'preflop_pot_tot_amount', 'preflop_num_final_participants']]
t_outcome_scaling_mean = t_df['outcome'].mean()
t_outcome_scaling_stddev = t_df['outcome'].std()
t_preflop_pot_scaling_mean = t_df['preflop_pot_tot_amount'].mean()
t_preflop_pot_scaling_stddev = t_df['preflop_pot_tot_amount'].std()
t_df['outcome_scaled'] = (t_df['outcome'] - t_outcome_scaling_mean) / t_outcome_scaling_stddev
t_df['preflop_pot_tot_amount_scaled'] = (t_df['preflop_pot_tot_amount'] - t_preflop_pot_scaling_mean) / t_preflop_pot_scaling_stddev

# --- fit model
model_win = OLS(endog=t_df['outcome_scaled'],  ###########
                exog=t_df[['preflop_pot_tot_amount_scaled', 'preflop_num_final_participants']])    ##########
results_win = model_win.fit()
print('\n--- Regression results for predicting win outcome------')
print(results_win.summary())

# ----- make predictions
df['preflop_pot_tot_amount_scaled'] = (df['preflop_pot_tot_amount'] - t_preflop_pot_scaling_mean) / t_preflop_pot_scaling_stddev
df['predicted_win_play_scaled'] = results_win.predict(df[['preflop_pot_tot_amount_scaled', 'preflop_num_final_participants']])
df['predicted_win_play'] = df.predicted_win_play_scaled * t_outcome_scaling_stddev + t_outcome_scaling_mean

df.drop(columns=['preflop_pot_tot_amount_scaled', 'predicted_win_play_scaled'], inplace=True)
del t_df, t_outcome_scaling_mean, t_outcome_scaling_stddev, t_preflop_pot_scaling_mean, t_preflop_pot_scaling_stddev

# --- Examine accuracy
t_min = min(min(df.loc[t_filter_cond, 'outcome']), min(df.loc[t_filter_cond, 'predicted_win_play']))
t_max = max(max(df.loc[t_filter_cond, 'outcome']), max(df.loc[t_filter_cond, 'predicted_win_play']))
plt.figure()
plt.scatter(df.loc[t_filter_cond, 'outcome'], df.loc[t_filter_cond, 'predicted_win_play'])
plt.plot(pd.Series([t_min, t_max]), pd.Series([t_min, t_max]), 'r-')
plt.title('AVP of predicted WINNINGS given:\npreflop pot and number of post preflop players')
plt.xlabel('actual outcomes (winning)')
plt.ylabel('predicted outcomes (winning)')
del t_min, t_max, t_filter_cond

# --- Predict losses -----
# --- scale
t_filter_cond = df.preflop_play_TF & ~df.win_TF   # & ((df.seat != 1) | (df.seat != 2))
t_df = df.loc[t_filter_cond & (df.outcome > -5000), ['outcome', 'preflop_pot_stake']].dropna()
t_outcome_scaling_mean = t_df['outcome'].mean()
t_outcome_scaling_stddev = t_df['outcome'].std()
t_preflop_pot_scaling_mean = t_df['preflop_pot_stake'].mean()
t_preflop_pot_scaling_stddev = t_df['preflop_pot_stake'].std()

t_df['outcome_scaled'] = (t_df['outcome'] - t_outcome_scaling_mean) / t_outcome_scaling_stddev
t_df['preflop_pot_stake_scaled'] = (df['preflop_pot_stake'] - t_preflop_pot_scaling_mean) / t_preflop_pot_scaling_stddev

# --- fit model
model_lose = OLS(endog=t_df['outcome_scaled'],
                 exog=t_df['preflop_pot_stake_scaled'])
results_lose = model_lose.fit()

print('\n--- Regression results for predicting lose outcome------')
print(results_lose.summary())

# --- make predictions
df['preflop_pot_stake_scaled'] = (df['preflop_pot_stake'] - t_preflop_pot_scaling_mean) / t_preflop_pot_scaling_stddev
df['predicted_loss_play_scaled'] = results_lose.predict(df['preflop_pot_stake_scaled'])
df['predicted_loss_play'] = df.predicted_loss_play_scaled * t_outcome_scaling_stddev + t_outcome_scaling_mean

del t_df, t_outcome_scaling_mean, t_outcome_scaling_stddev, t_preflop_pot_scaling_mean, t_preflop_pot_scaling_stddev
df.drop(columns=['preflop_pot_stake_scaled', 'predicted_loss_play_scaled'], inplace=True)

# ---- examine results
t_min = min(min(df.loc[t_filter_cond, 'outcome']), min(df.loc[t_filter_cond, 'predicted_loss_play']))
t_max = max(max(df.loc[t_filter_cond, 'outcome']), max(df.loc[t_filter_cond, 'predicted_loss_play']))
plt.figure()
plt.scatter(df.loc[t_filter_cond, ['outcome']], df.loc[t_filter_cond, 'predicted_loss_play'])
plt.plot(pd.Series([t_min, t_max]), pd.Series([t_min, t_max]), 'r-')
plt.title('AVP of predicted LOSSES given:\npreflop pot and number of post preflop players')
plt.xlabel('actual outcomes (losing)')
plt.ylabel('predicted outcomes (losing)')
del t_min, t_max, t_filter_cond

# --- predict payoffs for play win/lose and fold (lose only) for all hands/players
# if you choose not to play pre-flop, then whatever you lost in the outcome must have been what you put in before making the decision to fold
# if you chose to play pre-flop, assume that you would have folded immediately and lost 0 if you weren't blind, or blind if you were in seat 1 or 2 #### this could be refined to calc exactly how much money a player had in the pot when making the decision to fold
df['predicted_loss_fold'] = 0  # base assumption (if you played preflop and weren't seat 1 or 2, so we don't know if outcome is from preflop or later, assume you folded immediately
df.loc[df.preflop_play_TF & (df.seat == 1), 'predicted_loss_fold'] = -50 # if you played as small blind but hypothetically would have folded
df.loc[df.preflop_play_TF & (df.seat == 2), 'predicted_loss_fold'] = -100 # if you played as big blind but hypothetically would have folded
df.loc[~df.preflop_play_TF , 'predicted_loss_fold'] = df.loc[~df.preflop_play_TF, 'outcome']   # if you folded

payoff_dict = create_payoff_dict(df, fp_output, fn_output_payoff, save_TF=save_dict_TF)

#######


#### ====== EXAMINING IMPLIED EXPECTED VALUES ################
def calc_exp_value(prob_win_f, payoff_win_f, payoff_lose_f, payoff_fold_f):
    return (prob_win_f * payoff_win_f) + ((1-prob_win_f) * payoff_lose_f) + payoff_fold_f


for ind, row in df.iterrows():
    df.loc[ind, 'pred_exp_value'] = calc_exp_value(prob_win_f=pred_dict[row.player][row.game][row.hand],
                                                   payoff_win_f=payoff_dict[row.player][row.game][row.hand]['play']['win'],
                                                   payoff_lose_f=payoff_dict[row.player][row.game][row.hand]['play']['lose'],
                                                   payoff_fold_f=payoff_dict[row.player][row.game][row.hand]['fold']['lose'])

plt.figure()
plt.scatter(df.outcome, df.pred_exp_value, c=df.preflop_play_TF)
plt.title('AVP of expected values and outcomes from roll-up of \nprediction and payoff dictionaries')
plt.xlabel('Actual outcome')
plt.ylabel('Pred. expected value')
plt.axhline(0)
print('--- Rational choice:')
print('(Rational)   Play - positive expected value: %3.1f%%' % (df.loc[(df.pred_exp_value > 0) & df.preflop_play_TF].shape[0]/sum(df.pred_exp_value > 0)*100))
print('(Irrational) Fold - positive expected value: %3.1f%%' % (df.loc[(df.pred_exp_value > 0) & ~df.preflop_play_TF].shape[0]/sum(df.pred_exp_value > 0)*100))
print('(Irrational)   Play - negative expected value: %3.1f%%' % (df.loc[(df.pred_exp_value < 0) & df.preflop_play_TF].shape[0]/sum(df.pred_exp_value < 0)*100))
print('(Rational) Fold - negative expected value: %3.1f%%' % (df.loc[(df.pred_exp_value < 0) & ~df.preflop_play_TF].shape[0]/sum(df.pred_exp_value < 0)*100))

plt.figure()
plt.hist(df.pred_exp_value, bins=1000)
plt.title('Histogram of predicted expected values')
#####################################################

# ========== FOR USE WHEN DOING PAYOFF REGRESSION =============
# def avp_plot(actual_f, predicted_f):
#     a = plt.axes(aspect='equal')
#     plt.scatter(actual_f, predicted_f)
#     plt.xlabel('True Values')
#     plt.ylabel('Predictions')
#     lims = [min(min(actual_f), min(predicted_f)), max(max(actual_f), max(predicted_f))]
#     plt.xlim(lims)
#     plt.ylim(lims)
#     _ = plt.plot(lims, lims)
#
#
# avp_plot(test[target_name], model.predict(df_to_dataset(test, target_name)))
