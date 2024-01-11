#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Mall_Customers.csv')

# Encode the 'Gender' column
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Feature matrix
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']]

# K-means clustering
k_optimal = 6
kmeans_optimal = KMeans(n_clusters=k_optimal, init='k-means++', random_state=42)
labels_optimal = kmeans_optimal.fit_predict(X)

# Add cluster labels to the original dataframe
df['Cluster_Optimal'] = labels_optimal

# Function to generate strategy for a given cluster
def generate_strategy(cluster):
    st.subheader(f'Marketing Strategy for Cluster {cluster}')

    if cluster == 0:
        st.subheader("these contains  Mainly older customers (average age 56) with moderate income and spending")
        st.write("Target older customers with moderate income and spending.")
        st.write("Create special deals and promotions for loyalty.")

    elif cluster == 1:
        st.subheader("This cluster mainly contains  Younger customers (average age 42) with higher income but lower spending")
        st.write("Attract younger customers with higher income but lower spending.")
        st.write("Offer discounts and suggest additional products.")

    # Repeat for other clusters...\
    elif cluster == 2:
        st.subheader("This cluster mainly contains young customers (average age 25) with lower income but high spending")
        st.write("Focus on trendy and fashionable products.")
        st.write("Use social media for marketing to reach a younger audience.")
    elif cluster == 3:
        st.subheader("This cluster contains mainly Customers with moderate income and spending (average age 27)")
        st.write("Targeted advertising for customers with average incomes.")
        st.write("Introduce bundles of products to promote increased spending.")
    elif cluster == 4:
        st.subheader("This Cluster mainly contains the Customers with higher income and high spending (average age 33)")
        st.write("Emphasize premium and high-quality products")
        st.write("Host exclusive events for this cluster")


    elif cluster == 5:
        st.subheader("Older customers (average age 44) with lower income and spending")
        
        st.write("Focus on affordable and value-based products.")
        st.write("Educate customers about product value.")
   

    else:
        st.write("Strategy not defined for this cluster.")

# Streamlit app
st.title('Cluster Analytics App')

# Sidebar
selected_cluster = st.sidebar.selectbox('Select Cluster', ['Visualize All'] + list(range(6)))

# Placeholder for new data input
new_data_placeholder = st.empty()

# Visualization based on user selection
if selected_cluster == 'Visualize All':
    # 3D Scatter plot with different colors for different clusters
    fig = px.scatter_3d(df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)', color='Cluster_Optimal',
                        size_max=10, opacity=0.8, title='Optimal K-means Clustering',
                        color_discrete_map={i: f'cluster {i}' for i in range(k_optimal)})
    fig.update_layout(scene=dict(aspectmode="cube"))
    st.plotly_chart(fig)

    # Display new data input fields
    st.sidebar.subheader('Check for New Data')
    new_data_age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
    new_data_income = st.sidebar.number_input('Annual Income (k$)', min_value=0, max_value=200, value=50)
    new_data_spending = st.sidebar.number_input('Spending Score (1-100)', min_value=0, max_value=100, value=50)
    new_data_gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

    # Convert gender to numerical using LabelEncoder
    new_data_gender_encoded = le.transform([new_data_gender])[0]

    # Check for cluster button
    if st.sidebar.button('Check for Cluster'):
        new_data = np.array([[new_data_age, new_data_income, new_data_spending, new_data_gender_encoded]])
        predicted_cluster = kmeans_optimal.predict(new_data)[0]

        # Display the cluster for the new data
        st.sidebar.subheader(f'The new data belongs to Cluster {predicted_cluster}')

        # Generate strategy for the new data cluster
        generate_strategy(predicted_cluster)

else:
    # Clear the placeholder when Visualize All is not selected
    new_data_placeholder.empty()
    # Filter data for the selected cluster
    selected_cluster_data = df[df['Cluster_Optimal'] == selected_cluster]

    # Display 3D Scatter plot for the selected cluster
    fig_selected_cluster = px.scatter_3d(selected_cluster_data, x='Age', y='Annual Income (k$)',
                                        z='Spending Score (1-100)', title=f'Cluster {selected_cluster} Visualization')
    fig_selected_cluster.update_layout(scene=dict(aspectmode="cube"))
    st.plotly_chart(fig_selected_cluster)

    # Display statistics for the selected cluster
    st.subheader(f'Statistics for Cluster {selected_cluster}')
    st.write(selected_cluster_data.describe())

    # Generate strategy for the selected cluster
    generate_strategy(selected_cluster)


# In[ ]:





# In[ ]:




