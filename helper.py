import pandas as pd
import numpy as np
import plotly.express as px
from pandas_profiling import ProfileReport

def box_plot (x):
    y = pd.DataFrame(x)
    df1 = y.select_dtypes(np.number)
    fig = px.box(df1, width=1200, boxmode="overlay", title= 'Data Distribution', height=800)
    return fig

def pandas_profiling (x):
    y = pd.DataFrame(x)
    profile = ProfileReport(y, title="Dataset Profile", dataset={
                                "description": "This profiling report was generated for the uploaded file",
                                "copyright_holder": "Jyotishman Nath",
                                "copyright_year": "2022",
                                "url": "https://jymnath4.github.io/"},explorative=True)
    return profile

def data_mean (x):
    m = x.mean()
    m = pd.DataFrame(m)
    m = m.reset_index()
    m = m.rename(columns={'index': 'Column Name'})
    m = m.rename(columns={0: 'Mean'})
    return m


def missing_values(x):
    b = x.isna().sum()/len(x)*100
    b = pd.DataFrame(b)
    b = b.reset_index()
    b = b.rename(columns={'index': 'Column Name'})
    b = b.rename(columns={0: 'Percentage of missing values'})
    return b


