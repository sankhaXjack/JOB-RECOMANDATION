import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Step 1: Load the data
jobs_df = pd.read_csv('jobs_large.csv')
candidates_df = pd.read_csv('candidates.csv')

# Step 2: Preprocess Data (One-hot encoding for categorical variables)
jobs_df_encoded = pd.get_dummies(jobs_df, columns=['location', 'job_type', 'skills'])
candidates_df_encoded = pd.get_dummies(candidates_df, columns=['location', 'skills'])

# Step 3: Standardize the features
scaler = StandardScaler()
jobs_df_scaled = scaler.fit_transform(jobs_df_encoded)

# Step 4: K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Increased to 4 clusters
kmeans.fit(jobs_df_scaled)

# Assign clusters to the jobs
jobs_df['Cluster'] = kmeans.labels_

# Step 5: Dimensionality Reduction using PCA for 2D visualization
pca = PCA(n_components=2)
jobs_pca = pca.fit_transform(jobs_df_scaled)

# Plot the clusters in 2D
plt.figure(figsize=(10, 6))
plt.scatter(jobs_pca[:, 0], jobs_pca[:, 1], c=jobs_df['Cluster'], cmap='viridis', marker='o', edgecolor='k')
plt.title('Job Clusters (PCA Reduced to 2D)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.show()


# Step 6: MLE Visualization for one candidate
def recommend_jobs_for_candidate(candidate_id):
    candidate = candidates_df_encoded.loc[candidates_df_encoded['candidate_id'] == candidate_id].iloc[0]
    candidate_exp = candidates_df.loc[candidates_df['candidate_id'] == candidate_id, 'years_of_experience'].values[0]

    likelihoods = []

    plt.figure(figsize=(10, 6))

    for cluster in range(kmeans.n_clusters):
        cluster_jobs = jobs_df[jobs_df['Cluster'] == cluster]
        cluster_exp = cluster_jobs['years_of_experience'].values

        if len(cluster_exp) > 1:
            # Fit a normal distribution to the experience in this cluster
            mu, std = norm.fit(cluster_exp)
            likelihood = norm.pdf(candidate_exp, mu, std)
            likelihoods.append(likelihood)

            # Plot the distribution of years of experience for the cluster
            x_vals = np.linspace(min(cluster_exp) - 1, max(cluster_exp) + 1, 100)
            y_vals = norm.pdf(x_vals, mu, std)
            plt.plot(x_vals, y_vals, label=f'Cluster {cluster} (μ={mu:.2f}, σ={std:.2f})')

            # Mark the candidate's experience on the graph
            plt.axvline(candidate_exp, color='red', linestyle='--',
                        label=f'Candidate {candidate_id} (Experience={candidate_exp} years)' if cluster == 0 else "")

    plt.title(f'MLE Likelihood for Candidate {candidate_id}')
    plt.xlabel('Years of Experience')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

    # Find the best cluster based on likelihoods
    best_cluster = np.argmax(likelihoods)
    print(f'Candidate {candidate_id} is most likely to belong to Cluster {best_cluster}.')

    # Recommend jobs from the best matching cluster
    recommended_jobs = jobs_df[jobs_df['Cluster'] == best_cluster]
    print(f"Recommended jobs for Candidate {candidate_id}:")
    print(recommended_jobs[['job_id', 'location', 'job_type', 'skills']])
    print("\n")


# Step 7: Apply the recommendation and plot for a candidate (Example: Candidate 101)
recommend_jobs_for_candidate(101)
