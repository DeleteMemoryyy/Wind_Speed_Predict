import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn import metrics

rand_seed = 2017

# class Stacking(object):
#     def __init__(self, n_folds, base_models, stackers, added_results, weights):
#         assert len(stackers) + len(added_results) == len(weights)
#         assert np.array(weights).sum() == 1.0
#         self.y_dim = 1

#         self.n_folds = n_folds
#         self.base_models = base_models
#         self.stackers = stackers
#         self.added_results = added_results
#         self.weights = weights

#     def fit(self, X, y):
#         X = np.array(X)
#         y = np.array(y)

#         # self.y_dim = y.shape[1]
#         self.y_dim = y.shape[1]

#         kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

#         s_train = np.zeros((X.shape[0], self.y_dim, len(self.base_models)))

#         for i, mod in enumerate(self.base_models):
#             j = 0
#             for idx_train, idx_valid in kf.split(range(len(y))):
#                 x_train_j = X[idx_train]
#                 y_train_j = y[idx_train]
#                 x_valid_j = X[idx_valid]

#                 mod.fit(x_train_j, y_train_j)

#                 y_valid_j = mod.predict(x_valid_j)[:]
#                 s_train[idx_valid, :, i] = y_valid_j

#                 j += 1

#         for stacker in self.stackers:
#             stacker.fit(s_train.reshape(s_train.shape[0],-1), y)

#     def predict(self, T):
#         T = np.array(T)
#         s_test = np.zeros((T.shape[0], self.y_dim, len(self.base_models)))
#         y_predict = np.zeros((T.shape[0], self.y_dim, len(self.stackers)))
#         y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

#         for i, mod in enumerate(self.base_models):
#             s_test[:, :, i] = mod.predict(T)[:]

#         for i, stacker in enumerate(self.stackers):
#             y_predict[:, :, i] = stacker.predict(s_test.reshape((s_test.shape[0],-1)))[:]
#             y_predict_weighted += y_predict[:, :, i] * self.weights[i]

#         stackers_num = len(self.stackers)
#         for i, result in enumerate(self.added_results):
#             y_predict_weighted += result * self.weights[stackers_num+i]

#         return y_predict_weighted

#     def fit_predict(self, X, y, T):
#         X = np.array(X)
#         y = np.array(y)
#         T = np.array(T)

#         self.y_dim = y.shape[1]

#         kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

#         s_train = np.zeros((X.shape[0], (len(self.base_models)), self.y_dim))
#         s_test = np.zeros((T.shape[0], (len(self.base_models)), self.y_dim))
#         y_predict = np.zeros((T.shape[0], (len(self.stackers)), self.y_dim))
#         y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

#         for i, mod in enumerate(self.base_models):
#             s_test_i = np.zeros((
#                 s_test.shape[0], kf.get_n_splits(), self.y_dim))

#             j = 0
#             for idx_train, idx_valid in kf.split(range(len(y))):
#                 x_train_j = X[idx_train]
#                 y_train_j = y[idx_train]
#                 x_valid_j = X[idx_valid]

#                 mod.fit(x_train_j, y_train_j)

#                 y_valid_j = mod.predict(x_valid_j)[:]
#                 s_train[idx_valid, i, :] = y_valid_j
#                 s_test_i[:, j, :] = mod.predict(T)[:]

#                 j += 1

#             s_test[ :, i, :] = s_test_i.mean(1)

#         for stacker in self.stackers:
#             stacker.fit(s_train.reshape((s_train.shape[0],-1)), y)

#         for i, stacker in enumerate(self.stackers):
#             y_predict[:, i, :] = stacker.predict(s_test.reshape((s_test.shape[0],-1)))[:]
#             y_predict_weighted += y_predict[:, i, :] * self.weights[i]

#         stackers_num = len(self.stackers)
#         for i, result in enumerate(self.added_results):
#             y_predict_weighted += result * self.weights[stackers_num + i]

#         return y_predict_weighted

class Stacking(object):
    def __init__(self, n_folds, base_models, stackers, added_results, weights):
        assert len(stackers) + len(added_results) == len(weights)
        assert np.array(weights).sum() == 1.0
        self.y_dim = 1

        self.n_folds = n_folds
        self.base_models = base_models
        self.stackers = stackers
        self.added_results = added_results
        self.weights = weights

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape((y.shape[0], -1))

        self.y_dim = y.shape[1]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

        s_train = np.zeros((X.shape[0], self.y_dim, len(self.base_models)))

        for i, mod in enumerate(self.base_models):
            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train, :]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j)

                y_valid_j = mod.predict(x_valid_j)[:].reshape((-1, 1))
                s_train[idx_valid, :, i] = y_valid_j

                j += 1

        for stacker in self.stackers:
            stacker.fit(s_train.reshape(s_train.shape[0], -1), y)

    def predict(self, T):
        T = np.array(T)
        s_test = np.zeros((T.shape[0], self.y_dim, len(self.base_models)))
        y_predict = np.zeros((T.shape[0], self.y_dim, len(self.stackers)))
        y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

        for i, mod in enumerate(self.base_models):
            s_test[:, :, i] = mod.predict(T).reshape((-1, 1))

        for i, stacker in enumerate(self.stackers):
            y_predict[:, :, i] = stacker.predict(
                s_test.reshape((s_test.shape[0], -1)))[:]
            y_predict_weighted += y_predict[:, :, i] * self.weights[i]

        stackers_num = len(self.stackers)
        for i, result in enumerate(self.added_results):
            y_predict_weighted += result * self.weights[stackers_num + i]

        return y_predict_weighted

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y).reshape((y.shape[0], -1))
        T = np.array(T)

        self.y_dim = y.shape[1]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=rand_seed)

        s_train = np.zeros((X.shape[0], (len(self.base_models)), self.y_dim))
        s_test = np.zeros((T.shape[0], (len(self.base_models)), self.y_dim))
        y_predict = np.zeros((T.shape[0], (len(self.stackers)), self.y_dim))
        y_predict_weighted = np.zeros((T.shape[0], self.y_dim))

        for i, mod in enumerate(self.base_models):
            s_test_i = np.zeros((
                s_test.shape[0], kf.get_n_splits(), self.y_dim))

            j = 0
            for idx_train, idx_valid in kf.split(range(len(y))):
                x_train_j = X[idx_train]
                y_train_j = y[idx_train, :]
                x_valid_j = X[idx_valid]

                mod.fit(x_train_j, y_train_j)

                y_valid_j = mod.predict(x_valid_j).reshape((-1, 1))
                s_train[idx_valid, i, :] = y_valid_j
                s_test_i[:, j, :] = mod.predict(T).reshape((-1, 1))

                j += 1

            s_test[:, i, :] = s_test_i.mean(1)

        for stacker in self.stackers:
            stacker.fit(s_train.reshape((s_train.shape[0], -1)), y)

        for i, stacker in enumerate(self.stackers):
            y_predict[:, i, :] = stacker.predict(
                s_test.reshape((s_test.shape[0], -1)))[:].reshape((-1, 1))
            y_predict_weighted += y_predict[:, i, :] * self.weights[i]

        stackers_num = len(self.stackers)
        for i, result in enumerate(self.added_results):
            y_predict_weighted += result * self.weights[stackers_num + i]

        return y_predict_weighted

def CV(model, dl, verbose=1, model_name=None):
    # assert dl.mode == 'norm'
    model_name = model_name or ''

    score = -cross_val_score(model,dl.x_train,dl.y_train,cv=5,scoring='neg_mean_squared_error')
    model.fit(dl.x_train, dl.y_train)
    y_train_predict = model.predict(dl.x_train)
    y_test_predict = model.predict(dl.x_test)
    valid_rmse = np.sqrt(score).mean()
    train_rmse = np.sqrt(metrics.mean_squared_error(dl.y_train,y_train_predict))
    test_rmse = np.sqrt(metrics.mean_squared_error(dl.y_test, y_test_predict))
    if verbose != 0:
        print('{0:s}  valid_rmse: {1:f}'.format(model_name,valid_rmse))
        print('{0:s}  train_rmse: {1:f}'.format(model_name,train_rmse))
        print('{0:s}  test_rmse: {1:f}'.format(model_name, test_rmse))

    return valid_rmse, test_rmse


def grid_search(model, dl, param_grid, cv=None, verbose=0, model_name=None):
    # assert dl.mode == 'norm'
    model_name = model_name or ''

    grid = GridSearchCV(model, param_grid,
                        scoring='neg_mean_squared_error', cv=cv, n_jobs=4, verbose=verbose)
    grid.fit(dl.x_train, dl.y_train)

    print('Best {0:s} parameters: {1}'.format(model_name, grid.best_params_))

    return grid.best_estimator_

def varify_on_test(model, dl, verbose=0, model_name=None):
    # assert dl.mode == 'norm'
    model_name = model_name or ''

    model.fit(dl.x_train, dl.y_train)
    y_train_predict = model.predict(dl.x_train)
    y_test_predict = model.predict(dl.x_test)
    train_rmse = np.sqrt(metrics.mean_squared_error(
        dl.y_train, y_train_predict))
    test_rmse = np.sqrt(metrics.mean_squared_error(dl.y_test, y_test_predict))

    if verbose != 0:
        print('{0:s}  train_rmse: {1:f}'.format(model_name, train_rmse))
        print('{0:s}  test_rmse: {1:f}'.format(model_name, test_rmse))

    return train_rmse, test_rmse


