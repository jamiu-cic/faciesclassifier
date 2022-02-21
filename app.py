import pickle
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

facies_colors = ['#F4D03F', '#F5B041','#DC7633','#6E2C00','#1B4F72','#2E86C1', '#AED6F1', '#A569BD', '#196F3D']

facies_dict = {0: 'BS',
               1: 'CSiS',
               2: 'D',
               3: 'FSiS', 
               4: 'MS', 
               5: 'PS',
               6: 'SS',               
               7: 'SiSh',
               8: 'WS',
               }

column_names = ['Well Name', 'Depth', 'GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'NM_M', 'RELPOS']

def get_data(filename):
    """
    Assumptions:
        1. Input data does not contain missing values or outliers
        2. Input data contains the same logs as those used in developing the model   
    """
    #assert set(column_names).issubset(set(list(df.columns)), "put your error message here"
    df = pd.read_csv(filename)    
    return df

def scale_data(df):
    features = df[['GR', 'ILD_log10', 'DeltaPHI','PHIND', 'NM_M', 'RELPOS']]
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def plotter(df, facies_colors):
    #make sure logs are sorted by depth
    df = df.sort_values(by='Depth')
    cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
    
    ztop=df['Depth'].min(); zbot=df['Depth'].max()
    
    cluster_predicted=np.repeat(np.expand_dims(df['predicted_facies'].values,1), 100, 1)
    
    fig = plt.figure(figsize=(8, 12))
    
    plt.subplot(1,6,1)
    plt.plot(df.GR, df.Depth, '-g')
    plt.xlabel("GR")
    plt.xlim(df.GR.min(),df.GR.max())
    plt.ylim(ztop,zbot)
    #plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.locator_params(axis='x', nbins=3)
        
    plt.subplot(1,6,2)
    plt.plot(df.ILD_log10, df.Depth, '-')
    plt.xlabel("ILD_log10")
    plt.xlim(df.ILD_log10.min(),df.ILD_log10.max())
    plt.ylim(ztop,zbot)
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.locator_params(axis='x', nbins=3)
        
    plt.subplot(1,6,3)
    plt.plot(df.DeltaPHI, df.Depth, '-', color='0.5')
    plt.xlabel("DeltaPHI")
    plt.xlim(df.DeltaPHI.min(),df.DeltaPHI.max())
    plt.ylim(ztop,zbot)
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.locator_params(axis='x', nbins=3)
        
    plt.subplot(1,6,4)
    plt.plot(df.PHIND, df.Depth, '-', color='r')
    plt.xlabel("PHIND")
    plt.xlim(df.PHIND.min(),df.PHIND.max())
    plt.ylim(ztop,zbot)
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.grid()
    plt.locator_params(axis='x', nbins=3)
    
    ax = plt.subplot(1,6,5)
    im = ax.imshow(cluster_predicted, interpolation='none', aspect='auto',cmap=cmap_facies,vmin=1,vmax=9)
    ax.set_xlabel('Predicted Facies')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="20%", pad=0.05)
    cbar=plt.colorbar(im, cax=cax)
    cbar.set_label((21*' ').join([' BS ', 'CSiS', ' D  ', 'FSiS', ' MS ',' PS ', ' SS ','SiSh', ' WS ']))
    cbar.set_ticks(range(0,1)); cbar.set_ticklabels('')
    #cbar.set_ticks([x for x in logs["Facies_Label"].unique()])
    #cbar.ax.set_yticklabels([x for x in facies])
    plt.tight_layout()

    fig.suptitle('Well: %s'%df.iloc[0]['Well Name'], fontsize=14,y=1.025)
 
# loading the trained model
model_in = open('facies_classifier.pkl', 'rb') 
classifier = pickle.load(model_in)
    
## Build the user interface (UI)
### headers
st.title('Facies Classifier')
st.sidebar.header('User Inputs')

st.markdown("""
This app performs facies classification from wireline logs using random forest
""")

### file loader
filename = st.sidebar.file_uploader(label = "Upload your dataset here",
                                     type=["csv"],
                                     accept_multiple_files=False,
                                     key=None,
                                     help= "Accepts only .csv",
                                     on_change=None,
                                     args=None,
                                     kwargs=None)
### UI body
if filename is not None:
    df = get_data(filename)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    view_dict = {"All": df,
                 "first 10":df.head(10),
                 "last 10":df.tail(10),
                 "sample":df.sample(10, random_state = 42).sort_index(),
                 "description": df.describe()
                }
    st.markdown("""
    #### Viewing Dataframe ####
    """)
    view = st.radio("view", list(view_dict.keys()))
    st.write(view_dict[view])
    well = st.sidebar.selectbox("Choose a well to visualize", list(df['Well Name'].unique()))
    
if st.button('Build Model'):
    prediction = classifier.predict(scale_data(df))
    df['predicted_facies'] = prediction
    st.pyplot(plotter(df[df['Well Name'] == well], facies_colors))

if filename is not None:    
    st.sidebar.download_button(
         label="Download prediction as CSV",
         data=df.to_csv().encode('utf-8'),
         file_name='data with prediction.csv',
         mime='text/csv',
     )
