# # remove warnings
import warnings
warnings.filterwarnings('ignore')
# un/comment these lines later if reqd

# %matplotlib inline    # This is for ipythonnb
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_columns = 100

# # Load data

data = pd.read_csv('./train.csv')

# print data.shape   # (891, 12)

# print data.head()    # peek first 5 rows

# print data.describe()   # shows some stats for the data - count, mean etc

# # notice many (177) age values missing. Replace it with median age.

data['Age'].fillna(data['Age'].median(), inplace=True)
# print data.describe()  # it's updated

# # some visualisations for EDA

# # survival based on gender
# survived_sex = data[data['Survived']==1]['Sex'].value_counts()
# dead_sex = data[data['Survived']==0]['Sex'].value_counts()
# # print type(survived_sex) # pandas -> series :: <class 'pandas.core.series.Series'>
# # print survived_sex
# # print
# # print dead_sex
# df = pd.DataFrame([survived_sex, dead_sex])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize=(15, 8))
# plt.show()
# # women are more likely to survive


# # survival based on age
# figure = plt.figure(figsize=(15, 8))
# plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
# plt.xlabel('Age')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.show()
# # notice young children are more likely to survive


# # correlate fare ticket with survival
# figure = plt.figure((15, 8))
# plt.hist([data[data['Survived']==1]['Fare'], data[data['Survived']==0]['Fare']], stacked=True, color=['g', 'r'], bins=30, label=['Survived', 'Dead'])
# plt.xlabel('Fare')
# plt.ylabel('Number of passengers')
# plt.legend()
# plt.show()
# # passengers with cheaper fare are more likely to die


# # combining age, fare

# plt.figure(figsize=(15, 8))
# ax = plt.subplot()
# ax.scatter(data[data['Survived']==1]['Age'], data[data['Survived']==1]['Fare'], c='green', s=40)
# ax.scatter(data[data['Survived']==0]['Age'], data[data['Survived']==0]['Fare'], c='red', s=40)
# ax.set_xlabel('Age')
# ax.set_ylabel('Fare')
# ax.legend(('survived', 'dead'), scatterpoints=1, loc='upper right', fontsize=15)
# plt.show()
# # one distinct cluster of dead passengers can be seen. Age 15-50 and lower ticket fare

# # Correlation between ticket fare and class
# ax = plt.subplot()
# ax.set_ylabel('Average fare')
# data.groupby('Pclass').mean()['Fare'].plot(kind='bar', figsize=(15, 8), ax= ax)
# plt.show()
# # Ticket fare is directly proportional with the class


# # embarkation site vs survival
# survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
# dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
# df = pd.DataFrame([survived_embark, dead_embark])
# df.index = ['Survived', 'Dead']
# df.plot(kind='bar', stacked=True, figsize = (15, 8))
# plt.show()
# # info not very useful


# # Feature Engineering

def status(feature):
    # # check feature processed
    print 'Processing', feature, ': ok'



def get_combined_data():
    # # USEFUL TIP: Combine test and train data
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')

    # # Remove labels/targets from train
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)

    # # merge and return
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined


combined = get_combined_data()
# print combined.shape    # (1309, 11)
# print combined.head()


def get_titles():
    # # Extracting passenger titles
    global combined

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"
    }

    combined['Title'] = combined.Title.map(title_Dictionary)


get_titles()

# print combined.head()


# # processing ages. cos replacing with median may not be the best solution

grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])
grouped_median_train = grouped_train.median()
# print grouped_median_train

grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])
grouped_median_test = grouped_test.median()
# print grouped_median_test

# # We will fill up missing ages based on this grouping

def process_age():

    global combined

    def fillAges(row, grouped_median):
        if row['Sex'] == 'female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 1, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 1, 'Mrs']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['female', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['female', 1, 'Royalty']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 2, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 2, 'Mrs']['Age']

        elif row['Sex'] == 'female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return grouped_median.loc['female', 3, 'Miss']['Age']
            elif row['Title'] == 'Mrs':
                return grouped_median.loc['female', 3, 'Mrs']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 1, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 1, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 1, 'Officer']['Age']
            elif row['Title'] == 'Royalty':
                return grouped_median.loc['male', 1, 'Royalty']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 2, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 2, 'Mr']['Age']
            elif row['Title'] == 'Officer':
                return grouped_median.loc['male', 2, 'Officer']['Age']

        elif row['Sex'] == 'male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return grouped_median.loc['male', 3, 'Master']['Age']
            elif row['Title'] == 'Mr':
                return grouped_median.loc['male', 3, 'Mr']['Age']

    combined.head(891).Age = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis=1)
    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)

    # # entire function can be replaced with line below
    # combined["Age"] = combined.groupby(['Sex', 'Pclass', 'Title'])['Age'].transform(lambda x: x.fillna(x.median()))


    status('age')


process_age()
# print combined.info()
# # notice age is fixed but 1 fare, 2 embarked and many cabim values missing

# # remove names and dummy title
def process_names():

    global combined
    combined.drop('Name', axis=1, inplace=True)

    titles_dummies = pd.get_dummies(combined['Title'], prefix='Title')
    combined = pd.concat([combined, titles_dummies], axis=1)

    combined.drop('Title', axis=1, inplace=True)

    status('names')


process_names()
# print combined.head()


# # processing fares
def process_fare():
    global combined

    # # filling one missing value
    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace=True)
    combined.iloc[891:].Fare.fillna(combined.iloc[891].Fare.mean(), inplace=True)

    status('fare')


process_fare()


def process_embarked():

    global combined
    # # missing values filled with most frequent one
    combined.head(891).Embarked.fillna('S', inplace=True)
    combined.iloc[891:].Embarked.fillna('S', inplace=True)

    # # dummy encoding
    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix='Embarked')
    combined = pd.concat([combined, embarked_dummies], axis=1)
    combined.drop('Embarked', axis=1, inplace=True)

    status('embarked')


process_embarked()


# # processing cabin

def process_cabin():

    global combined

    # # replacing missing cabins with U for unkown?
    combined.Cabin.fillna('U', inplace=True)

    # # mapping cabin value with cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c: c[0])

    # # dummy encoding
    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix='Cabin')
    combined = pd.concat([combined, cabin_dummies], axis=1)

    combined.drop('Cabin', axis=1, inplace=True)

    status('cabin')


process_cabin()
# print combined.info()    # notice no missing values now
# print combined.head()


# # processing sex

def process_sex():

    global combined
    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    status('sex')


process_sex()


# # processing pclass

def process_pclass():

    global combined
    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix='Pclass')
    combined = pd.concat([combined, pclass_dummies], axis=1)

    combined.drop('Pclass', axis=1, inplace=True)

    status('pclass')


process_pclass()


def process_ticket():

    global combined

    # # extract prefix, return 'xxx' if unavailable
    def cleanTicket(ticket):
        ticket = ticket.replace('.', '')
        ticket = ticket.replace('/', '')
        ticket = ticket.split()
        ticket = map(lambda t: t.strip(), ticket)
        ticket = filter(lambda t: not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else:
            return 'XXX'


    # # dummy-fy
    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('ticket')


process_ticket()


# # process family

def process_family():

    global combined

    # # new feature: size of families inc passenger
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    # # few more features based on family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

    status('family')


process_family()


# # We do not need passenger id
combined.drop('PassengerId', inplace=True, axis=1)
# print combined.shape    # # (1309, 67)  => 67 features
# print combined.head()


# # MODELING

# # Split train-test. Apply classifier on train and eval. Then apply on test.

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score


def compute_score(clf, X, y, scoring):
    xval = cross_val_score(clf, X, y, scoring=scoring, cv=5)
    return np.mean(xval)

def recover_train_test_target():
    global combined

    train0 = pd.read_csv('./train.csv')
    targets = train0.Survived
    train = combined.head(891)
    test = combined.iloc[891:]

    return train, test, targets


train, test, targets = recover_train_test_target()


# # Feature selection

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)

features = pd.DataFrame()
features['feature'] = train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)

# features.plot(kind='barh', )
# plt.show()

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
# print train_reduced.shape   # # (891, 14) and (891, 13)  --> Fluctuates. Find out why.
# # reduced features to 13/14


# # Hyperparameters tuning

run_gs = False    # # Switch to true if you want to run grid search
# # If made true need to fix some things. Classifier part within if.

if run_gs:
    parameter_grid = {
        'max_depth': [4, 6, 8],
        'n_estimators': [50, 10],
        'max_features': ['sqrt', 'auto', 'log2'],
        'min_samples_split': [1.0, 3, 10],
        'min_samples_leaf': [1, 3, 10],
        'bootstrap': [True, False]
    }

    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(targets, n_folds=5)

    grid_search = GridSearchCV(forest, scoring='accuracy', param_grid=parameter_grid, cv=cross_validation)

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best Score: {}', format(grid_search.best_score_))
    print('Best Parameters: {}', format(grid_search.best_params_))
else:
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50,
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}

    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)


print compute_score(model, train, targets, scoring='accuracy')

output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('./test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId', 'Survived']].to_csv('./output.csv', index=False)


