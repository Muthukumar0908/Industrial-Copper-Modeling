import pandas as pd
import mysql.connector
import streamlit as st
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import pickle
import webbrowser

df=pd.read_csv(r"C:\Users\ADMIN\Videos\capstion_project\coper\coper_analysis_data.csv")
df1=pd.read_csv(r"C:\Users\ADMIN\Videos\capstion_project\coper\Copper_Set.xlsx - Result 1.csv")
df2=pd.read_csv(r"C:\Users\ADMIN\Videos\capstion_project\coper\coper_analysis2_data.csv")

icon = Image.open(r"E:\streamlit\1200px-NatCopper.jpg")
st.set_page_config(page_title="Industrial Copper Modeling",
                page_icon= icon,  
                   layout="wide", initial_sidebar_state="auto") 


with st.sidebar:
    st.sidebar.markdown("# :rainbow[Select an option to filter:]")
    selected = st.selectbox("**Menu**", ("Home","analysis","prediction"))
    
if selected=="Home":
    st.markdown('## :red[Project Title:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Industrial Copper Modeling")
    st.markdown('## :red[Skills takes away FRom This project:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Python scripting,  Data Preprocessing, EDA,  Streamlit")
    st.markdown('## :red[Domain:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Manufacturing")
    st.markdown('## :red[Problem:]')
    st.subheader("&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;1. Exploring skewness and outliers in the dataset")
    st.subheader('&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;2. Transform the data into a suitable format and perform any necessary cleaning and pre-processing steps.')
    st.subheader('&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;3. ML Regression model which predicts continuous variable ‘Selling_Price’.')
    st.subheader('&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;4. ML Classification model which predicts Status: WON or LOST.')
    st.subheader('&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;5. Creating a streamlit page where you can insert each column value and you will get the Selling_Price predicted value or Status(Won/Lost)')

    link="https://github.com/Muthukumar0908/Industrial-Copper-Modeling.git"
    link1='https://www.linkedin.com/in/muthukumar-r-44848416b/'
    colum3,colum4,colum5= st.columns([0.015,0.020,0.1])
    with colum3:
        if st.button('GidHub'):
            webbrowser.open_new_tab(link)
    with colum4:
        if st.button("LinkedIn"):
            webbrowser.open_new_tab(link1)
if selected=="prediction":
    select = st.sidebar.selectbox("**Type**",("Regression","Classifier"))
    with open(r"C:\Users\ADMIN\Videos\capstion_project\coper\bagging_regression.pkl",'rb') as file:
        Result = pickle.load(file)            
    if select =='Regression':
        lis=[]
        column1,column2 = st.columns([2,2], gap = 'small')
        with column1:
                min_ton = df2['quantity tons'].min()
                max_ton = df2['quantity tons'].max()
                min_thick = df2['thickness'].min()
                max_thick = df2['thickness'].max()
                min_width = df2['width'].min()
                max_width = df2['width'].max()
                min_customer = df2['customer'].min()
                max_customer = df2['customer'].max()
                Quantity_ton = st.number_input(f'**Enter Quantity Tons (Minimum : {min_ton}, Maximun : {max_ton})**')
                thickness = st.number_input(f'**Enter Thickness (Minimum : {min_thick}, Maximun : {max_thick})**')
                width = st.number_input(f'**Enter Width (Minimum : {min_width}, Maximun : {max_width})**')
                customer = st.selectbox("**Select a Customer Id**",options = df['customer'].unique())
        with column2:
            country=st.selectbox("**Select a country**",options=df2['country'].unique())
            status=st.selectbox(" **Select a status** ",options=df2['status'].unique())
            item_type=st.selectbox(" **Select a item type** ",options=df2['item type'].unique())
            product_ref=st.selectbox(" **Select a product_ref** ",options=df2['product_ref'].unique())
            
            lis.append(Quantity_ton)
            lis.append(customer)
            lis.append(country)
            a,b,c,d,e,f,g,h=6.00,0.00,1.00,4.00,5.00,3.00,2.00,7.00
            if status=="Won":
                lis.append(a)
            elif status=="Lost":
                lis.append(b)
            elif status=="Not lost for AM":
                lis.append(c)
            elif status=="Revised":
                lis.append(d)
            elif status=="To be approved":
                lis.append(e)
            elif status=="Offered":
                lis.append(f)
            elif status=="Offerable":
                lis.append(g)
            elif status=="Wonderful":
                lis.append(h)

            i,j,k,l,m,n,o=5.00,3.00,2.00,1.00,6.00,0.00,4.00
            if item_type=="W":
                lis.append(i)
            elif item_type=="S":
                lis.append(j)
            elif item_type=="PL":
                lis.append(k)
            elif item_type=="Others":   
                lis.append(l)
            elif item_type=="WL":
                lis.append(m)
            elif item_type=="IPL":
                lis.append(n)
            elif item_type=="SLAWR":
                lis.append(o)
            lis.append(thickness)
            lis.append(width)
            lis.append(product_ref)    
            # st.write(lis)
        if st.button("submit"): 
            y=np.array([lis])
            k=Result.predict(y)
            
            st.markdown(f"# :green[Prediction of selling price is :red['{k}']]")      

    
    if select =='Classifier':
        with open(r"C:\Users\ADMIN\Videos\capstion_project\coper\randomFormest_classffier.pkl",'rb') as file:
            Result = pickle.load(file)  
        df4=df2[(df2['status']=='Won')|(df2['status']=="Lost")]
        # st.write(df4.shape)
        lis1=[]
        column1,column2 = st.columns([2,2], gap = 'small')
        with column1:
            min_ton = df4['quantity tons'].min()
            max_ton = df4['quantity tons'].max()
            min_thick = df4['thickness'].min()
            max_thick = df4['thickness'].max()
            min_width = df4['width'].min()
            max_width = df4['width'].max()
            min_customer = df4['customer'].min()
            max_customer = df4['customer'].max()
            Quantity_ton = st.number_input(f'**Enter Quantity Tons (Minimum : {min_ton}, Maximun : {max_ton})**')
            thickness = st.number_input(f'**Enter Thickness (Minimum : {min_thick}, Maximun : {max_thick})**')
            width = st.number_input(f'**Enter Width (Minimum : {min_width}, Maximun : {max_width})**')
            customer = st.selectbox("**Select a Customer Id**",options = df['customer'].unique())
            country=st.selectbox("**Select a country**",options=df4['country'].unique())
        with column2:
            min_price = df4['selling_price'].min()
            max_price = df4['selling_price'].max()            

            item_type=st.selectbox(" **Select a item type** ",options=df4['item type'].unique())
            product_ref=st.selectbox(" **Select a product_ref** ",options=df4['product_ref'].unique())
            application=st.selectbox(" **Select a application** ",options=df4['application'].unique())
            selling_price = st.number_input(f'**Enter selling Price (Minimum : {min_price}, Maximun : {max_price})**')
            lis1.append(Quantity_ton)
            lis1.append(customer)
            lis1.append(country)   
            i,j,k,l,m,n,o=5.00,3.00,2.00,1.00,6.00,0.00,4.00
            if item_type=="W":
                lis1.append(i)
            elif item_type=="S":
                lis1.append(j)
            elif item_type=="PL":
                lis1.append(k)
            elif item_type=="Others":   
                lis1.append(l)
            elif item_type=="WL":
                lis1.append(m)
            elif item_type=="IPL":
                lis1.append(n)
            elif item_type=="SLAWR":
                lis1.append(o)
            lis1.append(application)
            lis1.append(thickness)
            lis1.append(width) 
            lis1.append(product_ref)
            lis1.append(selling_price) 
            # st.write(lis1)
        if st.button("submit"): 
            y=np.array([lis1])
            k=Result.predict(y)
            # st.write(k)
            if k== 6:
                st.markdown('# :white[The model predicts Status is :green[WIN]]')
            else:
                st.markdown("# :white[The model predicts Status is :red[LOST]]")
if selected=="analysis":
    # pass
    data = df.groupby(['country'])[['selling_price']].sum("selling_price").sort_values('selling_price',ascending=False).head(100)
    with st.expander(""):
         st.dataframe(data.style.background_gradient(cmap='Greens'),use_container_width=False)
    # st.dataframe(data)
    fig = px.bar(data, y='selling_price', x=data.index,title="which country highest selling_price?", width = 1200, height = 600)
    st.plotly_chart(fig)
    # fig.show()
    ###################################################
    df['customer'] = df['customer'].astype(int)
    data = df[['selling_price','customer','country']].sort_values('selling_price',ignore_index=True,ascending=False).head(20)
    # data
    with st.expander(""):
         st.dataframe(data.style.background_gradient(cmap='Greens'),use_container_width=False)
    # st.dataframe(data)
    colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']
    fig = px.pie(data, names = 'customer', values = 'selling_price', hover_data = ['country'],hole=0.3,
                color_discrete_sequence=px.colors.sequential.YlOrRd_r,title="which customer buy in highest price", width = 1000, height = 600)
    fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                    marker=dict(colors=colors, line=dict(color='#000000', width=2)))
    st.plotly_chart(fig) 
    # fig.show()
    ####################################################

    colum1,column2=st.columns([0.5,0.5])
    data=df.groupby(['item type'])[['quantity tons']].sum("quantity tons").sort_values("quantity tons").head(200)
    # data
    data1=df.groupby(['item type'])[['selling_price']].sum("selling_price").sort_values("selling_price").head(200)
    data['selling_price']=data1['selling_price']
    data['selling_price']=data['selling_price'].astype('str')
    data['quantity tons']=data['quantity tons'].astype('str')
    with colum1:
        # d=['selling_price',"quantity tons"]
        # for i in d:
        #     fig = px.bar(data, y=i, x=data.index,title=f"top total '{i}' in item_type ")
        #     fig.show()
        fig = px.bar(data, y='quantity tons', x=data.index,title="top total quantity tons in item_type ")
        # fig.show()
        st.plotly_chart(fig)    
    with column2:
        fig=px.bar(data, y='selling_price', x=data.index,title='top total selling_price in item_type ')
        # fig.show()
        st.plotly_chart(fig)    
    ##################################################
    # sns.boxplot(df1['selling_price'])
    date=df.copy()
    date['item_date'].astype('str')

    d=date['item_date'].str.split('-', n=2, expand=True)
    date['item_year']=d[0]+"-"+d[1]
    data=date.groupby("item_year")[["selling_price"]].sum('selling_price').head(10)
    # data 
    with st.expander("",expanded=False):
        st.dataframe(data.style.background_gradient(cmap='Greens'),use_container_width=False)
    fig = px.line(data, y='selling_price', x=data.index,title="which month purchese in highest selling price?",orientation ="h", width = 1000, height = 600)
    st.plotly_chart(fig)
    # fig.show()
    ####################################################

    data=df.groupby(['country'])[['quantity tons']].sum("quantity tons").sort_values('quantity tons',ascending=False).head(2000)
    with st.expander("",expanded=False):
        st.dataframe(data.style.background_gradient(cmap='Greens'),use_container_width=False)
    fig = px.bar(data, y='quantity tons', x=data.index,title="which country purchese in highest quantity tons?", width = 1000, height = 600)
    st.plotly_chart(fig)
    # pie.show()
    ####################################################

    data=df.sort_values(['selling_price',"quantity tons"],ascending=False).groupby(['item type']).head(10)
    # data
    fig=px.scatter(data,x="quantity tons",y="selling_price",color="item type",width = 1000, height = 600,title="top 10 selling price for using item_type")
    # fig.show()
    st.plotly_chart(fig)
    # ####################################################

    colum1,column2=st.columns([0.5,0.5])
    with colum1:
        # sns.boxplot(df['selling_price'])
        data=df1['selling_price']

        fig = px.box(data, y="selling_price",title="change the unstable data to stable data for selling _price")
        # fig.show()
        st.plotly_chart(fig)        
    with column2:
        data=df['selling_price']

        fig = px.box(data, y="selling_price")
        # fig.show()        
        st.plotly_chart(fig)  
    
    # colum1,column2=st.columns([0.5,0.5])
    # with colum1:
    #     data=df1['quantity tons']

    #     # fig = px.histogram(data, y="quantity tons")
    #     # fig.show()
    #     fig = px.histogram(data, x="quantity tons",
    #                     marginal="box")
    #     # fig.show()  
    #     st.plotly_chart(fig)  
    # with column2:
    #     data=df['quantity tons']
    #     fig = px.histogram(data, x="quantity tons",
    #                     marginal="violin",title="change the unstable data to stable data for quantity tons")
    #     # fig.show()  
    #     st.plotly_chart(fig) 
