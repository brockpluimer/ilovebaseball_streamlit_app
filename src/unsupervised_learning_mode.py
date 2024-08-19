import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
import umap
import plotly.express as px
from load_data import load_and_prepare_data

def unsupervised_learning_mode():
    st.title("Unsupervised Learning Mode")

    st.write("""
    Unsupervised learning algorithms are used to find patterns in data without pre-existing labels. 
    These methods can help discover hidden structures in the data, group similar players together, 
    or reduce the dimensionality of the data for visualization and further analysis.
    """)

    # Load data
    data_type = st.sidebar.radio("Select Player Type", ["Hitters", "Pitchers"])
    data_df, player_years = load_and_prepare_data("Hitter" if data_type == "Hitters" else "Pitcher")

    # Select features for analysis
    feature_columns = [col for col in data_df.columns if col not in ['IDfg', 'Name', 'Team', 'year']]
    selected_features = st.multiselect("Select Features for Analysis", feature_columns, default=feature_columns[:5])

    # Prepare data
    X = data_df[selected_features]

    # Data cleaning options
    cleaning_method = st.radio("Choose data cleaning method", 
                               ["Drop rows with missing values", 
                                "Impute missing values with mean", 
                                "Impute missing values with median"])

    st.write("""
    Data cleaning is crucial for unsupervised learning. Missing values can significantly affect the results:
    - Dropping rows removes incomplete data but may reduce the dataset size.
    - Imputing with mean or median fills in missing values with average values, preserving data size but potentially introducing bias.
    """)

    # Handle missing data
    if cleaning_method == "Drop rows with missing values":
        X = X.dropna()
        data_df = data_df.loc[X.index]
    else:
        imputer = SimpleImputer(strategy='mean' if cleaning_method == "Impute missing values with mean" else 'median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select algorithm
    algorithm = st.selectbox("Select Unsupervised Learning Algorithm", 
                             ["K-Means Clustering", 
                              "Principal Component Analysis (PCA)", 
                              "t-SNE", 
                              "Non-Negative Matrix Factorization (NMF)", 
                              "UMAP"])

    if algorithm == "K-Means Clustering":
        kmeans_analysis(X_scaled, X, data_df)
    elif algorithm == "Principal Component Analysis (PCA)":
        pca_analysis(X_scaled, X, data_df)
    elif algorithm == "t-SNE":
        tsne_analysis(X_scaled, X, data_df)
    elif algorithm == "Non-Negative Matrix Factorization (NMF)":
        nmf_analysis(X_scaled, X, data_df)
    else:  # UMAP
        umap_analysis(X_scaled, X, data_df)

def kmeans_analysis(X_scaled, X, data_df):
    st.write("""
    K-Means Clustering is an algorithm that groups similar data points together. It attempts to find k clusters 
    in the data, where each cluster is represented by its center (centroid).

    Parameter:
    - Number of clusters: Determines how many groups the algorithm will try to find in the data. 
      More clusters can capture finer distinctions but may lead to overfitting.
    """)

    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters,
                     hover_data={'Name': data_df['Name'], 'Year': data_df['year']},
                     labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                     title=f'K-Means Clustering (n_clusters={n_clusters})')
    st.plotly_chart(fig)

def pca_analysis(X_scaled, X, data_df):
    st.write("""
    Principal Component Analysis (PCA) is a dimensionality reduction technique. It finds the directions 
    (principal components) along which the data varies the most.

    Parameter:
    - Number of components: Determines how many principal components to compute. More components 
      retain more information but may be harder to visualize.
    """)

    n_components = st.slider("Select number of components", 2, min(X.shape[1], 10), 2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig = px.line(x=range(1, n_components+1), y=cumulative_variance_ratio,
                  labels={'x': 'Number of Components', 'y': 'Cumulative Explained Variance Ratio'},
                  title='PCA: Cumulative Explained Variance Ratio')
    st.plotly_chart(fig)
    
    if n_components >= 2:
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1],
                         hover_data={'Name': data_df['Name'], 'Year': data_df['year']},
                         labels={'x': 'First Principal Component', 'y': 'Second Principal Component'},
                         title='PCA: First Two Principal Components')
        st.plotly_chart(fig)

def tsne_analysis(X_scaled, X, data_df):
    st.write("""
    t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for dimensionality reduction 
    that is particularly well suited for the visualization of high-dimensional datasets.

    Parameter:
    - Perplexity: Balances attention between local and global aspects of data. Lower values 
      focus on local structure, while higher values focus on global structure.
    """)

    perplexity = st.slider("Select perplexity", 5, 50, 30)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1],
                     hover_data={'Name': data_df['Name'], 'Year': data_df['year']},
                     labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                     title=f't-SNE (perplexity={perplexity})')
    st.plotly_chart(fig)

def nmf_analysis(X_scaled, X, data_df):
    st.write("""
    Non-Negative Matrix Factorization (NMF) is an algorithm for parts-based learning. It decomposes 
    data into two non-negative matrices, which can be interpreted as a parts-based representation.

    Parameter:
    - Number of components: Determines the number of parts or features to extract. More components 
      can capture more detailed structure but may be harder to interpret.
    """)

    n_components = st.slider("Select number of components", 2, min(X.shape[1], 10), 2)
    nmf = NMF(n_components=n_components, random_state=42)
    X_nmf = nmf.fit_transform(X_scaled - X_scaled.min())  # NMF requires non-negative input
    
    if n_components >= 2:
        fig = px.scatter(x=X_nmf[:, 0], y=X_nmf[:, 1],
                         hover_data={'Name': data_df['Name'], 'Year': data_df['year']},
                         labels={'x': 'First NMF Component', 'y': 'Second NMF Component'},
                         title='NMF: First Two Components')
        st.plotly_chart(fig)

def umap_analysis(X_scaled, X, data_df):
    st.write("""
    UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique 
    that can be used for visualization similarly to t-SNE, but also for general non-linear 
    dimensionality reduction.

    Parameters:
    - Number of neighbors: Balances local versus global structure in the data. Lower values 
      capture more local structure, higher values more global structure.
    - Minimum distance: Controls how tightly the embedding is allowed to pack points together. 
      Smaller values result in tighter clusters.
    """)

    n_neighbors = st.slider("Select number of neighbors", 2, 100, 15)
    min_dist = st.slider("Select minimum distance", 0.0, 1.0, 0.1, 0.05)
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    
    fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1],
                     hover_data={'Name': data_df['Name'], 'Year': data_df['year']},
                     labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
                     title=f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
    st.plotly_chart(fig)

if __name__ == "__main__":
    unsupervised_learning_mode()