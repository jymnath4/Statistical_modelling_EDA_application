# Importing libraries

import streamlit as st
import pandas as pd
import helper
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from streamlit_pandas_profiling import st_profile_report


# Sidebar main title
st.sidebar.title('DATA ANALYSIS APPLICATION')

# creating the csv upload element on the sidebar allowing only .csv and .xlsx

uploaded_file = st.sidebar.file_uploader("Choose a file", accept_multiple_files=False,
                                          type=['csv', 'xlsx'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

# declaring global variable

# global data

# Try and except block for file upload functionality

# if uploaded_file is not None:
#      try:
#          data = pd.read_csv(uploaded_file)
#          st.write("Data Frame (File name):", uploaded_file.name)
#          st.write('Shape of dataset:', data.shape)
#
#      except Exception as e:
#          print (e)
#          data = pd.read_excel(uploaded_file)
#          st.write("Data Frame (File name):", uploaded_file.name)
#          st.write('Shape of dataset:', data.shape)
#
# # Try and except block for displaying the uploaded file
#
# try:
#     st.write (data)
#
# except Exception as e:
#     print(e)
#     st.write("Please upload a file")

if uploaded_file is None:
    st.header('Please upload a .CSV or .XLSX file to proceed')

# Defining Head and Tail of DataFrame
if uploaded_file is not None:
    st.sidebar.subheader('Preliminary Analysis')

if uploaded_file is not None:
    preview_menu = st.sidebar.checkbox('Preview Dataset')
    if preview_menu:
        a = st.sidebar.button('Head')
        b = st.sidebar.button('Tail')

#Defineing subtags
        if a:
            st.text('Top 5 rows of dataset')
            st.write(data.head())
        if b:
            st.text('Bottom 5 rows of dataset')
            st.write(data.tail())
    
# Defining Data Types of each colum

if uploaded_file is not None:
    a = st.sidebar.checkbox('View Shape Of Dataset')
    if a:
        a = st.sidebar.button('Rows')
        b = st.sidebar.button('Columns')

# Defining subtag

        if a:
            st.text('Number of rows')
            st.write(data.shape[0])
        if b:
            st.text('Number of columns')
            st.write(data.shape[1])

# Defining Missing value analysis

if uploaded_file is not None:
    a= st.sidebar.checkbox('Missing Value Analysis')
    if a:
        m=data.isnull().values.any()
        if m == True:
            fig, ax = plt.subplots()
            sns.heatmap(data.isnull(), yticklabels = False , cmap="viridis")
            st.pyplot(fig)
            a = data.isnull().sum()
            a = pd.DataFrame(a)
            a = a.reset_index()
            a = a.rename(columns={'index': 'Column Name'})
            a = a.rename(columns={0: 'Count'})
            st.write(a)

        else:
            st.success("There are no missing values")


# Duplicate value analysis

if uploaded_file is not None:
    a = st.sidebar.checkbox("Check Duplicate Values")

    if a:
        d = data.duplicated().any()
        if d == True:
            st.warning("Dataset contains duplicate values")
            bool_series = data.duplicated()
            x = data[bool_series]
            st.write(x)
            b = st.sidebar.button('Remove Duplicates')
            if b:
                data = data.drop_duplicates()
                st.text('Duplicate values are removed')

# Data Description

if uploaded_file is not None:
    a = st.sidebar.checkbox('Data Description')
    if a:
        st.write(data.describe(include='all'))

# creating user menu on the sidebar with Main tag  and subtag

# We have 2 main tag
if uploaded_file is not None:
    st.sidebar.subheader('Advanced Analysis')

if uploaded_file is not None:
    user_menu = st.sidebar.selectbox(label='Please select an option',
                                     options={'Overall Analysis','Statistical analysis'})


    # if tag1 is true

    if user_menu == 'Overall Analysis':
        a = st.sidebar.checkbox('Dataset Pandas Profile')
        # b = st.sidebar.checkbox('Missing Value Analysis')
        c = st.sidebar.checkbox('Correlation')
        # d = st.sidebar.checkbox('Data Description')
        e = st.sidebar.checkbox('Data Types')


    # Subtag
        if a:
            pandas_prof = helper.pandas_profiling(data)
            st.title("Pandas Profile")
            st_profile_report(pandas_prof)

    # Subtag
    #     if b:
    #         st.title("Missing Value Analysis")
    #         a = pd.DataFrame(data)
    #         f, ax = plt.subplots(figsize=(15, 10))
    #         sns.heatmap(a.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    #         sns.set(font_scale=1.25)
    #         st.pyplot(f)
    #         na_values = helper.missing_values(data)
    #         st.dataframe(na_values)

    # Subtag
        if c:
            st.title("Correlation Heat Map")
            b = pd.DataFrame(data)
            b = b.corr()
            f = plt.figure(figsize=(25, 15))
            sns.heatmap(b, annot=True, cmap='cubehelix')
            sns.set(font_scale=1.25)
            st.pyplot(f)

    # Subtag
    #     if d:
    #         st.title("Data Description")
    #         a = pd.DataFrame(data)
    #         a = a.describe()
    #         st.write(a)

    # Subtag
        if e:
            st.title("Data Types")
            a = data.select_dtypes(np.number)
            st.markdown('\nNumber data types')
            st.write('Shape:', a.shape)
            st.write(a)
            b = data.select_dtypes(object)
            st.markdown('\nObject data types')
            st.write('Shape:', b.shape)
            st.write(b)
            bt= st.button('Object Feature Value Counts', key=None, help='count of each unique value',
                          on_click=None, args=None, kwargs=None, disabled=False)
            if bt:
                for i, column in enumerate(b.columns):
                    c = b[column].value_counts()
                    st.write('Column Name:', column)
                    st.write(c)


    # Second main checkbox
    # user_menu_1= st.sidebar.checkbox('Statistical analysis')

    # if tag2 is true

    if user_menu == 'Statistical analysis':
        a = st.sidebar.checkbox('Box Plot')
        b= st.sidebar.checkbox('Mean')
        c= st.sidebar.checkbox('Normality Check')
        # d= st.sidebar.checkbox('T-Test')

        if a:
            bx_plot = helper.box_plot(data)
            st.title('Box Plot')
            st.plotly_chart(bx_plot)

        if b:
            d_mean = helper.data_mean(data)
            st.title('Mean of Numerical Column')
            st.write(d_mean)

        if c:
            # Main page display title and creating a dataframe with numeric columns only

            st.title('Chi Square Test & Distribution Plot For Homogeneity Check')
            a = pd.DataFrame(data)
            a = a.select_dtypes(np.number)

    # Iterate Chi square test and Distribution plot for all numerical columns in dataset

            for i, column in enumerate(a.columns):

                # Chi_sq_test

                chi2, p = stats.normaltest(a[column])
                st.write('Column Name:', column)
                st.write('chi2= %.2f, Pvalue= %.3f' % (chi2, p))
                if p < 0.05:  # alpha value is 0.05 or 5%
                    st.write("We are rejecting null hypothesis  \nData is not normally distributed")
                else:
                    st.write("We are accepting null hypothesis  \nData is normally distributed")

    # Defining parameters for distribution plot

                n_rows = 1
                n_cols = 1
                # Create the subplots
                fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)

                Distribution_plot= sns.distplot(a[column],hist=True,
                                                hist_kws=dict(ec="k"),color='green')
                st.pyplot(fig)

        # if d:
        #     st.title('Results For One sample T Test')
        #     a = pd.DataFrame(data)
        #     a = a.select_dtypes(np.number)
        #
        #     for i, column in enumerate(a.columns):
        #
        #         mean = a.mean()
        #         sample_size = 150
        #         np.random.seed(0)
        #         sample = np.random.choice(a[column], sample_size)
        #         ttest, p_value = ttest_1samp(sample, mean)
        #
        #         st.write('Column Name:', column)
        #         st.write('P Value=' % p_value)
        #
        #         if a < 0.05:  # alpha value is 0.05 or 5%
        #             st.write("we are rejecting null hypothesis  \nData is not homogenous")
        #         else:
        #             st.write("we are accepting null hypothesis  \nData is homogenous")
