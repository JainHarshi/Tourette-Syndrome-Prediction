# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_excel
import seaborn as sns
#data = pd.read_excel("C:/Users/DELL/OneDrive/Desktop/BOOSTUP/publised paper/renamme.xlsx")
data =pd.read_excel("D:\projects\publised paper\kt_data.xlsx")
#data1 =pd.read_excel("D:\projects\publised paper\Book1.xlsx")
print(data.tail())
#data = sns.load_dataset('car_crashes')

# plotting the density plot 
# for 'speeding' attribute
# using plot.density()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins = 50 , figsize = (20,15))
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np. random.permutation(len(data))
    print(shuffled)
    test_set_size= int(len(data)* test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    b = data.iloc[train_indices]
    c = data.iloc[test_indices]
    return data.iloc[train_indices],data.iloc[test_indices]
train_set , test_set = split_train_test(data,0.3)
print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data,test_size = 0.2 , random_state = 42)
print(f"Rows in train set : {len(train_set)}\nRows in test set : {len(test_set)}\n")
from sklearn.model_selection import StratifiedShuffleSplit
split= StratifiedShuffleSplit(n_splits= 1,test_size = 0.2,random_state = 42)
for train_index , test_index in split.split(data , data["GroupTouretteSyndromePerHealthyControls"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
strat_test_set['GroupTouretteSyndromePerHealthyControls'].value_counts()
strat_train_set['GroupTouretteSyndromePerHealthyControls'].value_counts()
strat_test_set['GroupTouretteSyndromePerHealthyControls'].info()
corr_matrix= data.corr()
corr_matrix['GroupTouretteSyndromePerHealthyControls'].sort_values(ascending= False)
from pandas.plotting import scatter_matrix
attributes = ['diseaseDurationYears','medication', 'DigialSpanForward','TMTA','TMTB','educationYears', 'smokingBurdentCigarettesPerDay','ageYears' , 'gender','DigialSpanBackward', 'handedness'  , 'odorThreshold', 'odorDiscrimination', 'odorIdentification', 'TDI'                                                   ]
scatter_matrix(data[attributes],figsize=(31,101))
data.plot(kind='scatter',x = 'diseaseDurationYears',y = 'medication', alpha = 0.8)
data = strat_train_set.drop('GroupTouretteSyndromePerHealthyControls',axis = 1)
data_labels = strat_train_set['GroupTouretteSyndromePerHealthyControls'].copy()
data.head()
data.shape
median = data['diseaseDurationYears'].median()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(data)
data.shape
imputer.statistics_

imputer.statistics_.shape
data.info()
data.shape
X = imputer.transform(data)
data_tr = pd.DataFrame(X,columns = data.columns)
data_tr.describe()
data = strat_train_set.drop('GroupTouretteSyndromePerHealthyControls',axis = 1)
data_labels = strat_train_set['GroupTouretteSyndromePerHealthyControls'].copy()
data.head()
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('imputer',SimpleImputer(strategy = 'median')),
                        ('std_scaler', StandardScaler()),
                       ])
data_num_tr = my_pipeline.fit_transform(data_tr)
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
nb = GaussianNB()
#model = MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=1000)
model= SGDClassifier()
#model= RandomForestClassifier()
#model = LogisticRegression()
#model = svm.SVC(kernel = 'linear', gamma = 'auto',C = 2)

#model = tree.DecisionTreeRegressor()
#model = tree.DecisionTreeClassifier()
#model = KNeighborsClassifier(n_neighbors=5)
model.fit(data_num_tr,data_labels)
#model
some_data = data.iloc[:5].values
some_labels = data_labels.iloc[:5].values
prepared_data = my_pipeline.transform(some_data)
list(some_labels)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
data_predictions =model.predict(data_num_tr)
lin_mse = mean_squared_error(data_labels, data_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_accu = accuracy_score(data_labels,data_predictions)
data_tr.info()
lin_rmse
lin_accu
X_test = strat_test_set.drop('GroupTouretteSyndromePerHealthyControls',axis = 1)
Y_test = strat_test_set['GroupTouretteSyndromePerHealthyControls'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
print(final_predictions, list(Y_test))
lin_accu1 = accuracy_score(final_predictions, list(Y_test))
lin_accu1
from sklearn.model_selection import cross_val_score
gamar =(Y_test).values.reshape(-1,1)
x = cross_val_score(model ,(final_predictions.reshape(-1,1)
)
                    , ([[1],
       [0],
       [0],
       [0],
       [0],
       [1],
       [1],
       [1],
       [1],
       [1],
       [0],
       [0]] ),scoring = 'accuracy', cv = 5 )
gamar2 =final_predictions.reshape(-1,1)

def print_scores(scores):
    print('scores' , scores)
    print('mean' , scores.mean())
    print('standard deviation', scores.std())
from sklearn.metrics import confusion_matrix , classification_report , confusion_matrix
confusion_matrix(y_true = data_labels , y_pred = data_predictions)

from sklearn.metrics import f1_score, matthews_corrcoef
matthews_corrcoef(data_labels,data_predictions)
y_true = data_labels 
y_pred = data_predictions
from sklearn.metrics import f1_score
f1_scores = (data_labels,data_predictions)
from sklearn.metrics import precision_score,recall_score
precision = precision_score(data_labels,data_predictions , average = None )
print(classification_report(data_labels,data_predictions))
m =final_predictions.reshape(-1,1)
from sklearn.metrics import roc_curve
fpr , tpr , thrsh = roc_curve(data_labels,data_predictions)
def plot_roc_curve (fpr , tpr):
    plt.plot(fpr , tpr , linewidth = 2)
    plt.plot([0,1],[0,1], 'k--')
    plt.xlabel('false Positive rate(1-Specificity)')
    plt.ylabel('True Positive rate(Sensitivity)')
plot_roc_curve (fpr , tpr)
plt.show()
