# Spotify Multidimensional Analysis

A comprehensive machine learning project that leverages multidimensional clustering to predict song success on Spotify charts. This work explores how different dimensions of metadata, from temporal patterns to sonic characteristics, can inform predictions about a track's chart performance and longevity.

## Project Overview

Understanding what makes a song successful on streaming platforms is a complex challenge. This project tackles it by first identifying meaningful patterns across multiple dimensions of the data through unsupervised clustering, then using those discovered patterns as features for downstream prediction tasks.

The analysis covers over 650,000 chart entries representing more than 9,000 unique songs from 2017 to 2023. By examining songs through lenses like their temporal release patterns, performance trajectories, audio characteristics, and artist collaboration networks, we create a rich feature space that captures the multifaceted nature of music success.

## Dataset

The project works with Spotify chart data that has been enriched with audio features and metadata. The main dataset files include song identifiers, chart rankings over time, audio features like danceability and energy, artist information, and geographic data. After running the clustering pipeline, an augmented dataset is generated with cluster assignments across seven different dimensions.

## Methodology

The approach unfolds in two main phases. First, we perform clustering across multiple dimensions to discover natural groupings in the data. These dimensions include temporal patterns capturing when songs enter charts and seasonal trends, performance clusters based on chart trajectory and longevity, sonic profiles derived from audio features, artist tiers reflecting popularity and reach, geographic patterns in chart performance across regions, collaboration structures among artists, and evolutionary patterns tracking how songs change position over time.

Once these clusters are identified and songs are tagged with their membership across all dimensions, we use this enriched metadata for various prediction tasks. The cluster assignments serve as powerful meta-features that capture complex patterns difficult to model directly from raw features alone.

## Repository Structure

The codebase is organized to facilitate reproducible analysis. The notebooks directory contains all analytical workflows, while the outputs directory stores generated visualizations, trained clustering models, and summary reports. Model artifacts including trained predictors, scalers, and encoders live at the root level. The data directory holds processed datasets, and various mapping files translate cluster IDs into human-readable labels.

## Getting Started

To run this analysis, you'll need a Python environment with standard data science libraries. Install pandas, numpy, scikit-learn, matplotlib, seaborn, and jupyter notebook. For specific model implementations, you may also need joblib and other supporting packages that are imported within the notebooks.

## Running the Analysis

The notebooks must be executed in a specific order to ensure proper data flow. Start by running All_Clusters.ipynb, which performs all multidimensional clustering operations and appends cluster assignments to the original dataset. This step is crucial because all subsequent analyses depend on these cluster features.

After clustering is complete, you can run the other notebooks in any order based on your analysis goals. The Binary_Classification notebook builds models to predict whether songs achieve sustained success, defined as staying in the Top 200 for at least eight weeks. For a more nuanced view, multiclass_prediction explores four success categories ranging from brief appearances to lasting hits, while multiclass_prediction_3_class offers a simplified three-category version.

If you're interested in predicting the exact duration of chart presence, regression_v6 models the number of weeks a song will remain in the Top 200. The Ablation_Test notebook conducts ablation studies to understand which feature groups contribute most to model performance. For exploratory analysis, eda_test provides visualizations and statistical summaries of the dataset. Finally, Plot_Merge is a utility notebook that combines multiple cluster visualization images into unified figures for presentation.

## Key Findings

The clustering analysis reveals six distinct temporal patterns in when songs enter and perform on charts, with Spring Friday releases forming the largest group. Performance clustering identifies four main trajectories, from brief chart visitors to mid-chart mainstays. The combination of temporal, performance, and sonic clusters as features significantly improves prediction accuracy compared to using raw features alone. Artist tier and collaboration patterns also emerge as strong indicators of sustained success.

## Model Outputs

The trained models and artifacts are saved for reuse. Final classification models exist in both binary and multiclass variants, each with their own label encoders and feature scalers. PCA transformers enable dimensionality reduction for visualization and modeling. The clustering models themselves, including temporal, performance, sonic, geographic, and collaboration clusterers, are preserved along with their fitted scalers. Model metadata files document training parameters and performance metrics for reproducibility.

## Visualizations

The outputs directory contains rich visualizations. Cluster distributions show the makeup of each discovered group. Elbow plots and silhouette scores document the optimal number of clusters chosen for each dimension. PCA projections visualize high-dimensional clusters in two dimensions. Cross-tabulation heatmaps reveal relationships between different clustering dimensions. Performance distribution plots compare chart trajectories across categories, and temporal trend visualizations show how chart patterns evolve over time and seasons.

## Future Directions

There are several promising avenues for extending this work. Additional clustering dimensions like lyrical themes or production quality could be incorporated. Deep learning approaches might better capture complex interactions between features. Incorporating real-time streaming data and social media signals could enable dynamic prediction as songs are released. Transfer learning could allow the methodology to generalize to other music platforms or markets.

## Contributing

This project welcomes contributions and suggestions for improvement. Whether you have ideas for new clustering dimensions, alternative modeling approaches, or enhanced visualizations, feel free to explore and extend the analysis.
