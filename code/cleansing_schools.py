import pandas as pd
import numpy as np
pd.set_option('expand_frame_repr',False)

public = pd.read_csv('Public_Schools.csv')
private = pd.read_csv('Private_Schools.csv')

print('Public schools:')
print('Full dataset shape:',public.shape)
number_of_public = public.shape[0]
print('Number of public schools: ',number_of_public)
print('Number of counties: ', len(public.COUNTY.unique()))

print('\nPrivate schools:')
print('Full dataset shape:',private.shape)
number_of_private = private.shape[0]
print('Number of private schools: ',number_of_private)
print('Number of counties: ', len(private.COUNTY.unique()))

# Extract info and clean data
public = public[['COUNTY','LEVEL_','ENROLLMENT','FT_TEACHER']].dropna()
# print('Shape after droping NaN:',public.shape)
print('\npublic:\n',public.head())
# public.to_csv("public_schools_clean")

private= private[['COUNTY','LEVEL_','ENROLLMENT','FT_TEACHER']].dropna()
# print('Shape after droping NaN:',private.shape)
print('\nprivate:\n',private.head())
# private.to_csv("private_schools_clean")

# Data Analysis
print('\nData Analysis:')
public_group_by_county = public.groupby(['COUNTY'])
# ft_teachers_pu = public_group_by_county['FT_TEACHER'].sum()

print('\nNumber of public schools for each county: ',public_group_by_county['COUNTY'].count())
# print('Number of full-time teachers per county: ',ft_teachers_pu)

private_group_by_county = private.groupby(['COUNTY'])
# ft_teachers_pr = private_group_by_county['FT_TEACHER'].sum()

print('\nNumber of private schools for each county: ',private_group_by_county['COUNTY'].count())
# print('Number of full-time teachers per county: ',ft_teachers_pr)


# merge public and private schools for total count
counties_pr = private_group_by_county['COUNTY'].count()
counties_pu = public_group_by_county['COUNTY'].count()
for i in counties_pr.index:
    if i in list(counties_pu.index):
        counties_pu[i] += counties_pr[i]


print('\nTotal publc and private schools for each county:\n',counties_pu)