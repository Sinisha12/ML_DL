import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')



def detect_strange_values(df):
    dictionary_of_unique_values = {}

    for categorical_column in df.columns:
        if(df[categorical_column].dtype == 'O'): # We assume that Object columns are the categorical (it says so in the desc.)
            # print("I'm a categorical column %s" % categorical_column)
            if(categorical_column not in dictionary_of_unique_values.keys()):
                dictionary_of_unique_values[categorical_column]=set(dataset[categorical_column].unique())
    return dictionary_of_unique_values
	
# function to check missing values (1)
def check_missing_values(df):
    plt.figure(figsize=(7, 7))
    null_in_data_set = df.isnull().sum()
    null_in_data_set = null_in_data_set[null_in_data_set > 0]
    null_in_data_set.sort_values(inplace=True)
    null_in_data_set.plot.bar()

# function to check missing values (2)
def check_missing_values_heat_map(df, legend = False):
    plt.figure(figsize=(16, 16))
    sns.heatmap(df.isnull(), cbar=legend)
	
# one more extra function for detecting missing values:
def table_view_of_missing_values(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data[missing_data['Total'] > 0]
	
	
	
# Can we create a function that returns the % of missing values in a column, given a dataset and a column name?
def return_missing_values_percent(df, column):
    return 100 * (df[column].isna().sum() / df[column].shape[0])
	

# Create a function that takes a percent P and drops all the columns of a dataframe
# that have a percentage count of missing values over P.
# There are many ways to do this, but please stick with
# DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# Please provide the percent in a decimal form, meaning, use 0.6 for 60 %
def drop_columns_from_dataset(df, p):
    return df.dropna(thresh=df.shape[0]*p, how='all', axis=1)
	
# Let us make use of the code from Data Wrangling lesson, but again, let us create reusable functions!
def list_columns_with_missing_data_with_counts_return_categorical_numerical(df):
    missing_data = df.isnull()
    numerical_columns = []
    categorical_columns = []
    for column in missing_data.columns.values.tolist():
        print('Column name:', column)
        print (missing_data[column].value_counts())
        print('Original column type:', df[column].dtype)
        print("")
        if(df[column].dtype in ['int64', 'float64']):
            numerical_columns.append(column)
        elif(df[column].dtype == 'object'):
            categorical_columns.append(column)
    return numerical_columns, categorical_columns
	
# Let us define a custom function to fill NANs
# If categorical, get the most frequent one
# If numerical, pass an aggregate function

def fill_nan_values(df, numeric, categorical, agg_function):
    for col in df.columns:
        if(col in categorical):
            df[col].replace(np.nan, df[col].value_counts().idxmax(), inplace=True)
        else:
            df[col].replace(np.nan, df[col].agg(agg_function), inplace=True)
    return df
	


		
		

# Let us create a function for binning
def binning_function(df, col_name, how_many_bins = 5, bin_name = 'class', plot = True):
    bins = np.linspace(min(dataset[col_name]), max(dataset[col_name]), how_many_bins)
    group_names = [bin_name + str(i+1) for i in np.arange(how_many_bins-1)]
    df[col_name + '_binned'] = pd.cut(df[col_name], bins, labels=group_names, include_lowest=True)
    if(plot):
        pyplot.bar(group_names, df[col_name + '_binned'].value_counts())
        # set x/y labels and plot title
        plt.pyplot.xlabel(col_name)
        plt.pyplot.ylabel("count")
        plt.pyplot.title(col_name + '- bins');
    return bins, group_names
	

# Let us define a function that does dummy encoding and appends the result dataframe to original dataframe
def dummy_encoding(df, col_name):
    df = pd.concat([df, pd.get_dummies(df[col_name])], axis=1)      
    return df
	
	
# we move on to correlation
def correlations(df):
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(), square=True, annot = True)


def find_important_correlations(df, num, target = 'SalePrice'):   
    d = {} # dictionary that holds columns
    d['strong'] = {}
    d['moderate'] = {}
    d['weak'] = {}
    d['insignificant'] = {}

    for col in num:
        if (col != target):
            pearson_coef, p_value = stats.pearsonr(df[col], df[target])
            # print("The Pearson Correlation Coefficient bettween", col.upper(), " and", target.upper() , "is", pearson_coef, "with a P-value of P =", p_value)

            # strong correlation
            if(np.abs(pearson_coef) > 0.5 and p_value < 0.001):
                print('Added to list of significant columns - strong:', col.upper())
                d['strong'][col] = {'coef': pearson_coef, 'p-value': p_value}

            # moderate correlation
            elif ((np.abs(pearson_coef) > 0.5) and ((p_value < 0.05) and (p_value > 0.001))):
                print('Added to list of significant columns - moderate:', col.upper())
                d['moderate'][col] = {'coef': pearson_coef, 'p-value': p_value}

            # weak correlation
            elif ((np.abs(pearson_coef) > 0.5) and ((p_value < 0.1) and (p_value > 0.05))):
                print('Added to list of significant columns - weak:', col.upper())
                d['weak'][col] = {'coef': pearson_coef, 'p-value': p_value}

            # insignificant correlation
            else:
                print('Added to list of significant columns - insignificant:', col.upper())
                d['insignificant'][col] = {'coef': pearson_coef, 'p-value': p_value}
    return d
	
def correlation_between_two_variables(df, v1, v2='SalePrice', plot = True, kind = 'reg'):
    sns.jointplot(df[v1], df[v2], kind=kind,  height=7, space=0)
	
	
# Let us define a function to group by a "group" in categorical variable and create AVOVA 1-way test
def anova_test(df, target = 'SalePrice', variable = 'LotShape', plot = True):
    group = df[[variable, target]].groupby([variable])
    unique_values_in_variable = df[variable].unique()
    list_for_anova = [group.get_group(x)[target] for x in unique_values_in_variable]
    f_val, p_val = stats.f_oneway(*list_for_anova)
    # f_val, p_val = stats.f_oneway(*list_for_anova) is equivalent to:
    # f_val, p_val = stats.f_oneway(list_for_anova[0], list_for_anova[1], list_for_anova[2], list_for_anova[3])
    if(plot):
        sns.boxplot(x=variable, y=target, data=df)
    print('F_VAL %0.6f, P_VAL %0.50f' % (f_val, p_val))
    return group, list_for_anova, f_val, p_val
	
	
# Can log transformation somehow help us?
def draw2by2log(arr):
    fig = plt.figure(figsize=(12, 12));
    plt.subplot(2,2,1)
    sns.distplot(arr, fit=norm); # distribution plot
    plt.subplot(2,2,3)
    stats.probplot(arr, plot=plt);
    plt.subplot(2,2,2)
    sns.distplot(np.log(arr), fit=norm);
    plt.subplot(2,2,4)
    stats.probplot(np.log(arr), plot=plt);