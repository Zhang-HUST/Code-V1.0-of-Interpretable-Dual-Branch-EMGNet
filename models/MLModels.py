import pickle
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.multioutput import MultiOutputRegressor


# from sklearn.preprocessing import OneHotEncoder


class GeneralMLModels:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model = None
        self.model_train = False
        self.x_train, self.y_train = None, None

    def init(self):
        if self.model_type == 'KNNClassifier':
            self.model = KNeighborsClassifier()
        elif self.model_type == 'LDAClassifier':
            self.model = LinearDiscriminantAnalysis()
        elif self.model_type == 'SVMClassifier':
            self.model = SVC()
        elif self.model_type == 'RFClassifier':
            self.model = RandomForestClassifier()

        elif self.model_type == 'SVMRegressor':
            self.model = MultiOutputRegressor(SVR())
        elif self.model_type == 'RFRegressor':
            self.model = RandomForestRegressor()
        elif self.model_type == 'MLPRegressor':
            self.model = MLPRegressor()
        elif self.model_type == 'GPRegressor':
            self.model = GaussianProcessRegressor()
        else:
            raise Exception('Unknown model_type')

    def parameter_optimization(self, train_features, train_labels):
        if self.model is None:
            raise Exception('please run init() first!')
        else:
            parameters = self.get_parameters()
            if self.model_type in ['KNNClassifier', 'LDAClassifier', 'SVMClassifier', 'RFClassifier']:
                gs = GridSearchCV(self.model, parameters, scoring='accuracy', refit=True, cv=5, verbose=1, n_jobs=-1)
            else:
                # scoring='neg_mean_squared_error' / 'r2'
                gs = GridSearchCV(self.model, parameters, scoring='neg_mean_squared_error', refit=True, cv=5,
                                  verbose=1, n_jobs=-1)
            gs.fit(train_features, train_labels)
            best_params = gs.best_params_
            return best_params

    def set_params(self, params):
        if self.model_type == 'KNNClassifier':
            self.model = KNeighborsClassifier(**params)
        elif self.model_type == 'LDAClassifier':
            self.model = LinearDiscriminantAnalysis(**params)
        elif self.model_type == 'SVMClassifier':
            self.model = SVC(**params)
        elif self.model_type == 'RFClassifier':
            self.model = RandomForestClassifier(**params)

        elif self.model_type == 'SVMRegressor':
            self.model = MultiOutputRegressor(**params)
        elif self.model_type == 'RFRegressor':
            self.model = RandomForestRegressor(**params)
        elif self.model_type == 'MLPRegressor':
            self.model = MLPRegressor(**params)
        elif self.model_type == 'GPRegressor':
            self.model = GaussianProcessRegressor(**params)
        else:
            raise Exception('Unknown model_type')

    def get_parameters(self):
        if self.model is None:
            raise Exception('please run init() first!')
        else:
            if self.model_type == 'KNNClassifier':
                parameters = {'n_neighbors': range(2, 11),
                              'weights': ['uniform', 'distance'],
                              'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
                              'p': [1, 2]
                              }

            elif self.model_type == 'LDAClassifier':
                parameters = {'solver': ['svd', 'lsqr', 'eigen'], 'n_components': [1, 2]}

            elif self.model_type == 'SVMClassifier':
                parameters = {'gamma': ['scale', 0.01, 0.1, 1, 10],
                              'C': [0.1, 1, 10, 100],
                              'kernel': ['linear', 'rbf', 'poly']}

            elif self.model_type == 'SVMRegressor':
                gammas = ['scale', 0.01, 0.1, 1, 10]
                Cs = [0.1, 1, 10, 100]
                kernels = ['linear', 'rbf', 'poly']
                params = {'gamma': [], 'C': [], 'kernel': []}
                parameters = {'estimator': []}
                for gamma in gammas:
                    for C in Cs:
                        for kernel in kernels:
                            params['gamma'] = gamma
                            params['C'] = C
                            params['kernel'] = kernel
                            parameters['estimator'].append(SVR(**params))

            elif self.model_type in ['RFClassifier', 'RFRegressor']:
                parameters = {'n_estimators': range(100, 241, 20),
                              'max_depth': range(5, 15)}

            elif self.model_type == 'MLPRegressor':
                parameters = {'hidden_layer_sizes': [(100,), (200,), (300,),
                                                     (20, 100), (20, 200), (20, 300),
                                                     (20, 100, 200)],
                              # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
                              'alpha': [0.0001, 0.001, 0.01],
                              'max_iter': [200, 500, 1000]}

            elif self.model_type == 'GPRegressor':
                parameters = {'kernel': [1.0 * RBF(length_scale=1.0), 1.0 * RBF(length_scale=0.5),
                                         1.0 * RBF(length_scale=0.1), 3.0 * RBF(length_scale=0.1),
                                         1.0 * ConstantKernel(constant_value=1.0),
                                         1.0 * ConstantKernel(constant_value=0.5),
                                         1.0 * ConstantKernel(constant_value=0.1),
                                         ConstantKernel(1.0) * RBF(1.0), ConstantKernel(0.5) * RBF(0.5),
                                         ConstantKernel(0.1) * RBF(0.1)],
                              'alpha': [0.1, 0.05, 0.01],
                              'normalize_y': [True, False]}
                # {'alpha': 0.01, 'kernel': 1.73 ** 2 * RBF(length_scale=0.1), 'normalize_y': True}

            else:
                raise Exception('Unknown model_type')

            return parameters

    def get_model_name(self):
        model_name = self.model_type
        return model_name

    def train(self, x_train, y_train):
        if self.model is not None:
            self.x_train, self.y_train = x_train, y_train
            print('x_train.shape: ', x_train.shape, 'y_train.shape: ', y_train.shape)
            self.model.fit(x_train, y_train)
            self.model_train = True
        else:
            raise Exception('No model initialization!')

    def predict(self, x_test):
        if self.model_train:
            y_test = self.model.predict(x_test)
            return y_test
        else:
            raise Exception('No model training was performed!')

    def save(self, save_name):
        if self.model_train:
            with open(save_name, 'wb') as f:
                pickle.dump(self.model, f)
            # load model
            # with open('train_model.pkl', 'rb') as f:
            #     gbr = pickle.load(f)
        else:
            raise Exception('No model training was performed!')
