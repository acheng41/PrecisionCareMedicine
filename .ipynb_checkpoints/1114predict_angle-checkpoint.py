import numpy as np
import scipy.io as sio
from scipy.signal import resample
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import KFold, train_test_split, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from getParameters import get_gait_parameters_insole
from getParameters import gait_aligned_jnt
from pre import butter_lowpass_filter
#%%
names = ['foot_trace_r', 'cop_x_r', 'cop_y_r', 'cont_area_r', 'pp_r', 'pp_x_r', 'pp_y_r',
             'foot_trace_l', 'cop_x_l', 'cop_y_l', 'cont_area_l', 'pp_l', 'pp_x_l', 'pp_y_l'] # matrix name
FM_all = []
ankle_all = []
d = ['1', '2', '4', '6'] # valid normal walk
for n in d:
    data = sio.loadmat('data/gait_recording_102324_walk'+ n +'.mat')
    insoleAll_l = data['insoleAll_l'].astype(np.float64)
    insoleAll_r = data['insoleAll_r'].astype(np.float64)
    t_insole_l = data['t_insole_l'].astype(np.float64)
    t_insole_r = data['t_insole_r'].astype(np.float64)

    t_trackers = data['t_trackers'].astype(np.float64)
    jnt_angles_all_l = np.array(data['jnt_angles_all_l'])
    jnt_angles_all_r = np.array(data['jnt_angles_all_r'])
    jnt_pos_all_l = np.array(data['jnt_pos_all_l'])
    jnt_pos_all_r = np.array(data['jnt_pos_all_r'])

    # feature matrix
    g = get_gait_parameters_insole(insoleAll_r, insoleAll_l, t_insole_r, t_insole_l)
    FM_r = [g['foot_trace_r'],g['cop_x_r'],g['cop_y_r'],g['cont_area_r'],g['pp_r'],g['pp_x_r'],g['pp_y_r']]
    FM_l = [g['foot_trace_l'],g['cop_x_l'],g['cop_y_l'],g['cont_area_l'],g['pp_l'],g['pp_x_l'],g['pp_y_l']]
    FM_r, FM_l = np.column_stack(FM_r), np.column_stack(FM_l)

    # resample
    fm = []
    for step in range(10,110):
        start_i, end_i = g['strike_l'][step], g['strike_l'][step+1]
        fm_l=resample(FM_l[start_i:end_i,:], 100, axis=0)
        fm_r=resample(FM_r[start_i:end_i, :], 100, axis=0)
        fm.append(np.hstack((fm_l, fm_r)))
    FM = np.vstack(fm)

    joint = gait_aligned_jnt(g, jnt_angles_all_l, jnt_angles_all_r, jnt_pos_all_l, jnt_pos_all_r, t_trackers)
    angle = joint['resampled_angles_l']['ankle'][100*9:109*100,:]

    for i in range(FM.shape[0]):
        FM_all.append(FM[i, :])
        ankle_all.append(angle[i, :])

feature_matrix = np.array(FM_all)
ankle_matrix = np.array(ankle_all)

#%% predict ankle angle
X_train, X_test, y_train, y_test = train_test_split(feature_matrix, ankle_matrix, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=5)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

y_true, y_pred_rf, y_pred_lr, y_pred_knn = [],[],[],[]
mse_rf, r2_rf = [], []
mse_lr, r2_lr = [], []
mse_knn, r2_knn = [], []

for train_index, test_index in kf.split(X_train):
    X_cv_train, X_cv_test = X_train[train_index], X_train[test_index]
    y_cv_train, y_cv_test = y_train[train_index], y_train[test_index]
    y_true.extend(y_cv_test)

    rf.fit(X_cv_train, y_cv_train)
    y_cv_pred = rf.predict(X_cv_test)
    y_pred_rf.extend(y_cv_pred)
    mse_rf.append(mean_squared_error(y_cv_test, y_cv_pred))
    r2_rf.append(r2_score(y_cv_test, y_cv_pred))

    lr.fit(X_cv_train, y_cv_train)
    y_cv_pred = lr.predict(X_cv_test)
    y_pred_lr.extend(y_cv_pred)
    mse_lr.append(mean_squared_error(y_cv_test, y_cv_pred))
    r2_lr.append(r2_score(y_cv_test, y_cv_pred))

    knn.fit(X_cv_train, y_cv_train)
    y_cv_pred = knn.predict(X_cv_test)
    y_pred_knn.extend(y_cv_pred)
    mse_knn.append(mean_squared_error(y_cv_test, y_cv_pred))
    r2_knn.append(r2_score(y_cv_test, y_cv_pred))

y_true = np.array(y_true)
y_pred_rf = np.array(y_pred_rf)
y_pred_lr = np.array(y_pred_lr)
y_pred_knn = np.array(y_pred_knn)

mse_rf_mean, mse_rf_std = np.mean(mse_rf), np.std(mse_rf)
r2_rf_mean, r2_rf_std = np.mean(r2_rf), np.std(r2_rf)

mse_lr_mean, mse_lr_std = np.mean(mse_lr), np.std(mse_lr)
r2_lr_mean, r2_lr_std = np.mean(r2_lr), np.std(r2_lr)

mse_knn_mean, mse_knn_std = np.mean(mse_knn), np.std(mse_knn)
r2_knn_mean, r2_knn_std = np.mean(r2_knn), np.std(r2_knn)


print(f"Random Forest - MSE: {mse_rf_mean:.4f} ± {mse_rf_std:.4f}, R²: {r2_rf_mean:.4f} ± {r2_rf_std:.4f}")
print(f"Linear Regression - MSE: {mse_lr_mean:.4f} ± {mse_lr_std:.4f}, R²: {r2_lr_mean:.4f} ± {r2_lr_std:.4f}")
print(f"KNN - MSE: {mse_knn_mean:.4f} ± {mse_knn_std:.4f}, R²: {r2_knn_mean:.4f} ± {r2_knn_std:.4f}")

#%% visualization
# predict-true
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(y_true, y_pred_rf, color='blue', alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # 理想的对角线
plt.title('Random Forest Prediction (5-Fold CV)')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.subplot(1, 3, 2)
plt.scatter(y_true, y_pred_lr, color='green', alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # 理想的对角线
plt.title('Linear Regression Prediction (5-Fold CV)')
plt.xlabel('True Values')
plt.ylabel('Predictions')

plt.subplot(1, 3, 3)
plt.scatter(y_true, y_pred_knn, color='orange', alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)  # 理想的对角线
plt.title('KNN Prediction (5-Fold CV)')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
#%%
# MSE
models = ['Random Forest', 'Linear Regression', 'KNN']
mse_means = [mse_rf_mean, mse_lr_mean, mse_knn_mean]
mse_stds = [mse_rf_std, mse_lr_std, mse_knn_std]

r2_means = [r2_rf_mean, r2_lr_mean, r2_knn_mean]
r2_stds = [r2_rf_std, r2_lr_std, r2_knn_std]
x = range(len(models))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.bar(x, mse_means, yerr=mse_stds, capsize=5, color=['blue', 'green', 'red'])
plt.xticks(x, models)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')

# R²
plt.subplot(1, 2, 2)
plt.bar(x, r2_means, yerr=r2_stds, capsize=5, color=['blue', 'green', 'red'])
plt.xticks(x, models)
plt.title('R² Score')
plt.ylabel('R²')

plt.tight_layout()
plt.show()

#%% feature importance
# Random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
cv_results = cross_validate(rf, X_train, y_train, cv=5, return_estimator=True, scoring='neg_mean_squared_error')
# feature importance
all_feature_importance = []
for model in cv_results['estimator']:
    feature_importance = model.feature_importances_
    all_feature_importance.append(feature_importance)
avg_feature_importance = np.mean(all_feature_importance, axis=0)
sorted_idx = np.argsort(avg_feature_importance)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(np.arange(len(avg_feature_importance)), avg_feature_importance[sorted_idx], align='center')
plt.yticks(np.arange(len(avg_feature_importance)), [names[i] for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance (5-fold Cross-Validation, Sorted)')
plt.show()
