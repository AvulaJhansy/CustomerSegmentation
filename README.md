# Mall Customer Segmentation Project

## Problem Definition
The goal of this project is to understand customer segmentation in the mall based on various attributes such as age, gender, annual income, and spending score. Insights derived from this analysis can help in devising targeted marketing strategies and improving customer satisfaction.

## Data Collection and Preprocessing
- The mall customer dataset was collected and preprocessed.
- Preprocessing steps included handling null values, removing duplicates, detecting outliers, and encoding the 'gender' attribute as 0 and 1.
- A column named 'customer id' was dropped as it was deemed irrelevant.

## Data Visualization
- Matplotlib and Seaborn libraries were utilized for visualizations.
- Visualizations included distributions of age and gender, distribution of gender with respect to spending score, and others.
- Observations from visualizations included insights on gender distribution, income differences between genders, and spending behavior.

## Model Implementation
- Determined the number of clusters (k) using methods like the Elbow Method and Silhouette Score, ultimately selecting k=6.
- Applied the K-Means clustering algorithm on selected features such as age, annual income, spending score, and gender, and assigned labels to each data point.

## Model Efficiency
- Evaluated the quality of clustering using the Davies-Bouldin Index, which measures compactness and separation between clusters.
- The obtained Davies-Bouldin Index value was 0.7475215820921529, indicating a medium quality of clustering.

## Deployment using Streamlit
- Developed a user-friendly website with Streamlit to showcase interactive visualizations using Plotly.
- The website provides insights into customer segmentation into clusters and tailors marketing strategies for each cluster.
- Users can input new data to determine which cluster the data point falls under.

## Conclusion
This project demonstrates the use of data science techniques to analyze customer data and derive actionable insights for marketing strategies and customer segmentation.
