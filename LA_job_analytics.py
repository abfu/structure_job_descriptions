
# importing necessary libraries
#%%
import numpy as np
import pandas as pd
import os
import re
import datetime

import matplotlib.pyplot as plt
import seaborn

from wordcloud import WordCloud
import cufflinks as cf
import nltk

# File formats in the data
# csv
# pdf
# txt
# docx

# Goal: convert job postings into single structured csv file
#%%
# vars to extract from files
# File Name
# Job Class Title
# Job Class Number
# Requirement set ID
# Requirement subset
# Job Duties
# Education years
# School Type
# Education Major
# Experience Length
# Full Time Part Time
# Exp Job Class Title
# Exp Job Class Alt Resp
# Exp JOb Class Funtion
# Course Count
# Course Length
# Course Subject
# Misc Course Details
# Drivers License req
# Driv Lic Type
# Addtl Lic
# Exam Type
# Entry Salary Gen
# Entry Salary DWP
# Open Date

#%%
# Load sample .csv files
job_title_dict = pd.read_csv('CityofLA/Additional data/job_titles.csv',
                             header=None, names=['job_title'])

kaggle_data_dictionary = pd.read_csv('CityofLA/Additional data/kaggle_data_dictionary.csv')

sample_export = pd.read_csv('CityofLA/Additional data/sample job class export template.csv')

#%%
# Shape of sample export
sample_export.shape
# multiple rows for each job title
# number of rows given by required
sample_export.keys()
sample_export

# The kaggle_data_dictionary file gives the column names
# as well as the data types for each column of the target data frame

#%% md
# describe target data frame

#%%
# Creating empty target data frame
print('The data types for the columns are:\n{}'.format(pd.unique(kaggle_data_dictionary['Data Type'])))
data_type_list = list()
for i in kaggle_data_dictionary['Data Type']:
    if i == 'String':
        data_type_list.append('object')
    elif i == 'Integer':
        data_type_list.append('int64')
    elif i == 'Float':
        data_type_list.append('float64')
    elif i == 'Date':
        data_type_list.append('object')
    else:
        print('dtype not found')

df_dtype = pd.DataFrame(kaggle_data_dictionary['Field Name'])
df_dtype['dtype'] = data_type_list

dtypes = {}
for i in range(len(df_dtype)):
    dtypes[df_dtype['Field Name'][i]] = np.dtype(df_dtype['dtype'][i])

#%%
df = pd.DataFrame(columns=kaggle_data_dictionary['Field Name'])
df = df.astype(dtype=dtypes)

#%%
# Look for numbers and non-capital letters in job titles
for i in job_title_dict['job_title']:
    if re.match(r'([^A-Z])', i) is not None:
        print('Job title ' + i + ' contains numbers')



#%%

# Folder job bullitins contains txt files of job postings

# vars for job bullitin
# job title
# class code
# open date
# revised date, if revised
# annual salary

#%% md
# Get variables 'FILE_NAME' and 'JOB_CLASS_TITLE'.
# Adding a column with the full filepath.

#%%
jb_dir = 'CityofLA/Job Bulletins'

# get list of job openings

jb_files = os.listdir(jb_dir)

# Loading path to files

def get_file_path(files, folder_path):
    file_list = list()
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        file_list.append(filepath)
    return file_list

jb_path_list = get_file_path(jb_files, jb_dir)
df_jb = df
df_jb['filepath'] = jb_path_list


# Extracting job titles
jb_title = list()
for filepath in df_jb['filepath']:
    job_title = re.findall(r'\b[A-Z][A-Z]+\b', filepath)
    job_title = ' '.join(job_title)
    jb_title.append(job_title)


jb_title = [x.lower() for x in jb_title]
df_jb['FILE_NAME'] = jb_files
df_jb['JOB_CLASS_TITLE'] = jb_title

#TODO capitalize first letter

#%% md
# Clean job titles

#%%
# remove 'rev' from job title


def clean_jb_title():
    for i, k in enumerate(df_jb['job_title']):
        if re.search(r'(\srev)$', k) is not None:
            print('Removing \'rev\' from job title ' + k)
            df_jb['job_title'][i] = re.sub(r'(\srev)$', '', k)
            print(df_jb['job_title'][i])

# reverse removal of numbers in job title
df_jb['JOB_CLASS_TITLE'][0] = '311 director'

#%% md
# Extracting Class code

#%%


def get_var_class_code():
    class_code = list()
    # Selecting file
    for i in df_jb['filepath']:
        # Opening file
        open_file = open(i, 'r')
        read_file = open_file.readlines()
        # Reading the file line by line to search for class code
        for j in np.arange(len(read_file)):
            # Looking for the class code in the file
            if re.search(r'([Cc]lass\s[Cc]ode)', read_file[j]):
                if re.search(r'[0-9]{4}', read_file[j]):
                    class_code.extend(re.findall(r'[0-9]{4}', read_file[j]))
                    break
                elif re.search(r'[0-9]{4}', read_file[j+1]):
                    class_code.extend(re.findall(r'[0-9]{4}', read_file[j+1]))
                    break

        # check if a class code was appended and append 'None' if there was no class code in the file.
        if len(class_code) - 1 != df_jb.loc[df_jb['filepath'] == i].index:
            print('\nNo class code found for job title:')
            print(df_jb['JOB_CLASS_TITLE'].loc[df_jb['filepath'] == i].values)
            print('Index for file: {}'.format(df_jb.loc[df_jb['filepath'] == i].index))
            print('Appending \'0000\' for unkown class code.')
            class_code.append('0000')
        open_file.close()
    return class_code

#%%

class_code = get_var_class_code()
df_jb['JOB_CLASS_NO'] = class_code

#%%
# Function returns four variables 'class_code', 'open_date', 'rev_date' and 'rev_status'.
# I ended up not using this function, as 'rev_date' and 'rev_status' was not required at this point.


def get_vars():
    class_code = list()
    open_date = list()
    rev_date = list()
    rev_status = list()
    # Selecting file
    for i in df_jb['filepath']:
        # Opening file
        open_file = open(i, 'r')
        read_file = open_file.readlines()
        # Reading the file line by line to search for class code
        for j in np.arange(len(read_file)):
            # Looking for the class code in the file
            if re.search(r'([Cc]lass\s[Cc]ode)', read_file[j]):
                if re.search(r'[0-9]{4}', read_file[j]):
                    class_code.append(re.findall(r'[0-9]{4}', read_file[j]))
                    break
                elif re.search(r'[0-9]{4}', read_file[j+1]):
                    class_code.append(re.findall(r'[0-9]{4}', read_file[j+1]))
                    break

        # check if a class code was appended and append 'None' if there was no class code in the file.
        if len(class_code) - 1 != df_jb.loc[df_jb['filepath'] == i].index:
            print('\nNo class code found for job title:')
            print(df_jb['JOB_CLASS_TITLE'].loc[df_jb['filepath'] == i].values)
            print('Index for file: {}'.format(df_jb.loc[df_jb['filepath'] == i].index))
            print('Appending \'0000\' for unkown class code.')
            class_code.append(0000)

        # Using separate for-loop to look for 'open date'.
        for k in np.arange(len(read_file)):
            if re.search(r'([Oo]pen\s[Dd]ate)', read_file[k]):
                if re.search(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[k]):
                    open_date.extend(re.findall(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[k]))
                    break
                elif re.search(r'(\d{1,2}-\d{2}-\d{1,2}).*', read_file[k+1]):
                    open_date.extend(re.findall(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[k+1]))
                    break

        # checking if a open date was found and appended.
        if len(open_date)-1 != df_jb.loc[df_jb['filepath'] == i].index:
            print('\nNo open date found for job title:')
            print(df_jb['JOB_CLASS_TITLE'].loc[df_jb['filepath'] == i].values)
            print('Index for file: {}'.format(df_jb.loc[df_jb['filepath'] == i].index))
            print('Appending \'01-01-01\' for unkown open date.')
            open_date.append('01-01-01')

        # Looking for revision date
        for l in np.arange(len(read_file)):
            if re.search(r'(revised)', read_file[l], flags=re.IGNORECASE):
                if re.search(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[l]):
                    rev_date.append(re.findall(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[l]))
                    rev_status.append(1)
                    break
                elif re.search(r'(\d{1,2}-\d{2}-\d{1,2}).*', read_file[l+1]):
                    rev_date.append(re.findall(r'(\d{1,2}-\d{1,2}-\d{1,2}).*', read_file[l+1]))
                    rev_status.append(1)
                    break

        if len(rev_date) - 1 != df_jb.loc[df_jb['filepath'] == i].index:
            print('\nNo rev date found for job title:')
            print(df_jb['JOB_CLASS_TITLE'].loc[df_jb['filepath'] == i].values)
            print('Index for file: {}'.format(df_jb.loc[df_jb['filepath'] == i].index))
            print('Appending \'01-01-01\' for unkown open date.')
            rev_date.append('01-01-01')
            rev_status.append(0)

        open_file.close()
    return class_code, open_date, rev_date, rev_status

class_code, open_date, rev_date, rev_status = get_vars()

#%%
df_jb['OPEN_DATE'] = open_date

#%% md
# extracting requirements
#%%
def get_var_req():
    requirement = list()
    for i in df_jb['filepath']:
        # Opening file
        open_file = open(i, 'r')
        read_file = open_file.read()
        read_file = read_file.replace('\n\n', ' ')
        try:
            file_requirement = (re.search(r'(?:REQUIREMENT[S]?|REQUIREMENT[S]?\/MINIMUM QUALIFICATIONS?)(.*?)(?:NOTES|PROCESS NOTES)',
                                          read_file, flags=re.S).group(1))
            # removing typos
            file_requirement = file_requirement.replace('/MINIMUM QUALIFICATIONS', '')
            file_requirement = file_requirement.replace('/ MINIMUM QUALIFICATIONS', '')
            file_requirement = file_requirement.replace('/ MINIMUM QUALIFICATION', '')
            file_requirement = file_requirement.replace('/MINIMUM QUALIFCATIONS', '')
            file_requirement = file_requirement.replace('/MINIMUM QUALIFICATION', '')
            file_requirement = file_requirement.replace('/MINIMUM REQUIREMENTS', '')
            file_requirement = file_requirement.replace('/MIMINUMUM QUALIFICATION', '')
            file_requirement = file_requirement.replace('/MINUMUM QUALIFICATIONS', '')
            file_requirement = file_requirement.replace('/MINUMUM QUALIFICATIONS', '')
            file_requirement = file_requirement.replace('/MINIMUM QUALIFICAITON', '')
            file_requirement = file_requirement.replace('/MINUMUM QUALIFICATION', '')
            requirement.append(file_requirement)
        except:
            requirement.append(None)





    return requirement

#%%
requirement = get_var_req()
df_jb['REQUIREMENT_SET_ID'] = requirement

#%%
for i in requirement:
    count_none = 0
    if i is None:
        count_none += 1





