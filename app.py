import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.title("🚀 Advanced Customer Segmentation Tool")
st.write("Upload your dataset (CSV file) to perform customer segmentation.")

file_path = "customer_segmentation_dataset (2).csv"

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        st.download_button("Download Sample Dataset", f, file_name="customer_segmentation_dataset (2).csv", mime="text/csv")
else:
    st.error("Sample dataset not found. Please upload it manually.")

uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

if uploaded_file is not None:
  
    try:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset successfully loaded!")
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")
        st.stop()

  
    st.write("### Preview of the Dataset")
    st.write(df.head())

    
    df.fillna(df.median(numeric_only=True), inplace=True)

    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        st.write("### Categorical Columns Encoded")
    else:
        st.warning("No categorical columns found. Proceeding without encoding.")

   
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    
    pca = PCA()
    df_pca = pca.fit_transform(df)

    
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(explained_variance >= 0.95) + 1
    st.write(f"Retaining {n_components} components to explain 95% of the variance.")

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df)

   
    input_dim = df_pca.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

   
    split_idx = int(0.8 * len(df_pca))
    X_train, X_val = df_pca[:split_idx], df_pca[split_idx:]

    
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_data=(X_val, X_val), verbose=1)

   
    encoder = Model(input_layer, encoded)
    X_encoded = encoder.predict(df_pca)

    
    n_clusters = st.number_input("Enter the number of clusters for K-Means", min_value=2, max_value=10, value=5)

    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_encoded)

 
    autoencoder.save("autoencoder_model.keras")  
    pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    pickle.dump(pca, open("pca.pkl", "wb")) 

    
    df['Cluster'] = kmeans.labels_

    st.write("## Customer Segmentation Results")

   
    st.write("### Customer Group Distribution")
    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Customer Group', 'Number of Customers']
    fig1 = px.bar(cluster_counts, x='Customer Group', y='Number of Customers', 
                title="How Many Customers Are in Each Group?")
    st.plotly_chart(fig1)
    st.caption("This shows the number of customers belonging to each group.")

   
    st.write("### Customer Group Visualization (2D)")
    pca_2d = PCA(n_components=2)
    df_pca_2d = pca_2d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_2d = pd.DataFrame(df_pca_2d, columns=['Key Feature 1', 'Key Feature 2'])
    df_pca_2d['Customer Group'] = df['Cluster']
    fig2 = px.scatter(df_pca_2d, x='Key Feature 1', y='Key Feature 2', 
                    color='Customer Group', 
                    title="How Customer Groups Are Spread Out")
    st.plotly_chart(fig2)
    st.caption("Each dot represents a customer. Similar customers are grouped closer together.")

   
    st.write("### Key Metrics Across Customer Groups")
    feature_importance = df.groupby('Cluster').mean().reset_index()
    fig3 = px.box(feature_importance, x='Cluster', y=numerical_cols, 
                title="What Makes Each Group Different?")
    fig3.update_xaxes(title_text="Customer Group")
    fig3.update_yaxes(title_text="Metric Value (Scaled)")
    st.plotly_chart(fig3)
    st.caption("Higher/lower values show what metrics are important for each group.")

   
    st.write("### Relationships Between Customer Metrics")
    fig4 = px.scatter_matrix(df, dimensions=numerical_cols[:5], 
                            color='Cluster', 
                            title="How Customer Metrics Interact with Each Other")
    st.plotly_chart(fig4)
    st.caption("Explore how different customer metrics relate to each other.")

 
    st.write("### Customer Group Visualization (3D)")
    pca_3d = PCA(n_components=3)
    df_pca_3d = pca_3d.fit_transform(df.drop(columns=['Cluster']))
    df_pca_3d = pd.DataFrame(df_pca_3d, columns=['Key Feature 1', 'Key Feature 2', 'Key Feature 3'])
    df_pca_3d['Customer Group'] = df['Cluster']
    fig5 = px.scatter_3d(df_pca_3d, x='Key Feature 1', y='Key Feature 2', z='Key Feature 3',
                        color='Customer Group', 
                        title="3D View of Customer Groups")
    st.plotly_chart(fig5)
    st.caption("Rotate this 3D view to explore customer groups from different angles.")

    
    st.write("### Group Strengths and Weaknesses")
    cluster_means = df.groupby('Cluster').mean().reset_index()
    categories = numerical_cols[:5] 

    for cluster in cluster_means['Cluster']:
        fig6 = go.Figure()
        fig6.add_trace(go.Scatterpolar(
            r=cluster_means[cluster_means['Cluster'] == cluster][categories].values.flatten(),
            theta=[col.replace("_", " ").title() for col in categories], 
            fill='toself',
            name=f'Group {cluster}'
        ))
        fig6.update_layout(
            polar=dict(radialaxis=dict(visible=True, title="Metric Strength")),
            title=f"What Makes Group {cluster} Unique?"
        )
        st.plotly_chart(fig6)
    st.caption("Larger areas show where a group is stronger compared to others.")

   
    st.write("### Group Comparison Dashboard")
    cluster_characteristics = df.groupby('Cluster').mean()
    fig7 = px.imshow(cluster_characteristics, 
                    labels=dict(x="Customer Metrics", y="Customer Group", color="Strength"),
                    title="Compare Groups Across All Metrics")
    fig7.update_xaxes(side="top")
    st.plotly_chart(fig7)
    st.caption("Warmer colors = higher values, cooler colors = lower values for each metric.")

else:
    st.info("Please upload a CSV file to get started.")