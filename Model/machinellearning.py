from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression




def create_svm_model(kernel='linear', C=1, gamma='scale'):

    svm_model = SVC(kernel=kernel, C=C, gamma=gamma)

    return svm_model

def create_log_reg_model(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42):
    log_reg_model = LogisticRegression(multi_class=multi_class, solver=solver, 
                                       max_iter=max_iter, random_state=random_state)
    return log_reg_model


