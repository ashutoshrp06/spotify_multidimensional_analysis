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

```
spotify_multidimensional_analysis/
│
├── data/
│   └── processed/
│       └── clustering_summary_report.txt            # Summary statistics of clustering results
│
├── outputs/
│   ├── clustering_summary_report.txt                # Summary statistics (duplicate location)
│   │
│   ├── models/
│   │   ├── feature_scaler.pkl                       # Feature scaler for final model
│   │   ├── feature_scaler_3class.pkl                # Feature scaler for 3-class model
│   │   ├── final_model.pkl                          # Trained final classification model
│   │   ├── final_model_3class.pkl                   # Trained 3-class classification model
│   │   ├── geo_kmeans5.joblib                       # Geographic clustering model
│   │   ├── geo_scaler.joblib                        # Scaler for geographic features
│   │   ├── imputer.joblib                           # Missing value imputer
│   │   ├── kmeans_k3.joblib                         # K-means clustering model (k=3)
│   │   ├── label_encoder.pkl                        # Label encoder for final model
│   │   ├── model_metadata.json                      # Metadata for final model
│   │   ├── model_metadata_3class.json               # Metadata for 3-class model
│   │   ├── pca_transformer.pkl                      # PCA transformer for final model
│   │   ├── pca_transformer_3class.pkl               # PCA transformer for 3-class model
│   │   └── scaler.joblib                            # General feature scaler
│   │
│   └── figures/
│       ├── Audio Evolution Clusters.png
│       ├── all_models_comparison_final_3class.png
│       ├── all_models_final_comparison.png
│       ├── artist_cluster_elbow_plot.png
│       ├── binary_ablation_study_results.svg
│       ├── cluster_crosstab_heatmap.png
│       ├── cluster_type_importance_3class.png
│       ├── combined_pca_cluster_plots.svg
│       ├── confusion_matrices_all_models.png
│       ├── confusion_matrices_all_models_3class.png
│       ├── eda_audio_features.png
│       ├── eda_audio_features_3class.png
│       ├── eda_cluster_distributions.png
│       ├── eda_cluster_distributions_3class.png
│       ├── eda_correlation_matrix.png
│       ├── eda_correlation_matrix_3class.png
│       ├── eda_target_analysis.png
│       ├── eda_target_analysis_3class.png
│       ├── feature_importance_by_type.png
│       ├── feature_importance_by_type_3class.png
│       ├── feature_importance_final_3class.png
│       ├── feature_importance_final_model.png
│       ├── final_model_confusion_matrix.png
│       ├── final_model_confusion_matrix_3class.png
│       ├── granular_ablation_study_results.png
│       ├── heatmap_average_rank.png
│       ├── heatmap_count.png
│       ├── model_comparison.png
│       ├── model_comparison_3class.png
│       ├── pca_sonic_clusters.png
│       ├── pca_variance_analysis.png
│       ├── pca_variance_analysis_3class.png
│       ├── per_class_performance.png
│       ├── per_class_performance_3class.png
│       ├── prediction_confidence_3class.png
│       ├── prediction_confidence_analysis.png
│       ├── smote_comparison_3class.png
│       ├── success_category_distribution.png
│       ├── success_category_distribution_3class.png
│       ├── sustained_success_distributions.png
│       ├── sustained_success_distributions.svg
│       ├── temporal_cluster_optimization.png
│       ├── temporal_clusters_visualization.png
│       ├── temporal_eda.png
│       ├── temporal_trends.png
│       ├── tuned_models_confusion_matrices.png
│       ├── tuned_models_confusion_matrices_3class.png
│       ├── tuning_comparison.png
│       ├── tuning_comparison_3class.png
│       ├── unique_songs_cluster_plot.png
│       ├── unique_songs_elbow_plot.png
│       └── unique_songs_silhouette_plot.png
│
├── notebooks/
│   ├── Ablation_Test.ipynb                          # Feature ablation analysis
│   ├── All_Clusters.ipynb                           # Main clustering pipeline
│   ├── Binary_Classification.ipynb                  # Binary success prediction
│   ├── Plot_Merge.ipynb                             # Visualization merging utility
│   ├── Regression.ipynb                             # Chart duration regression
│   ├── eda_test.ipynb                               # Exploratory data analysis
│   ├── multiclass_prediction.ipynb                  # 4-class success prediction
│   └── multiclass_prediction_3_class.ipynb          # 3-class success prediction
│
├── report/
│   ├── 30_Final_report.pdf                          # Final project report
│   ├── template.tex                                 # LaTeX template
│   ├── refs.bib                                     # Bibliography
│   ├── nips_2016.sty                                # NIPS style file
│   ├── lineno.sty                                   # Line numbering style
│   ├── nicefrac.sty                                 # Nice fractions style
│   └── figs/                                        # Report figures
│       ├── PCA_regression.png
│       ├── actual_vs_pred_regression.png
│       ├── all_models_comparison_final_3class.png
│       ├── combined_pca_cluster_plots_final.png
│       ├── feature_importance.svg
│       ├── feature_importance_by_type_3class.png
│       ├── feature_importance_final_3class.png
│       ├── final_model_confusion_matrix_3class.png
│       ├── learning_curves_regression.png
│       ├── per_class_performance_3class.png
│       ├── precision_recall_curve.svg
│       ├── prediction_confidence_3class.png
│       ├── reg_res_1.png
│       ├── reg_res_2.png
│       ├── reg_res_3.png
│       ├── regression_target_skew.png
│       ├── roc_curve_comparison.svg
│       └── sustained.svg
```

## Getting Started

To run this analysis, you'll need a Python environment with standard data science libraries. Install pandas, numpy, scikit-learn, matplotlib, seaborn, and jupyter notebook. For specific model implementations, you may also need joblib and other supporting packages that are imported within the notebooks. For a complete list of libraries and their versions, refer to `requirements.txt`. Install all dependencies using:

```bash
pip install -r requirements.txt
```

## Running the Analysis

### Step 1: Run Clustering (Required First)
- **Execute `All_Clusters.ipynb`** - This performs all multidimensional clustering operations and appends cluster assignments to the dataset
- **Critical**: All subsequent analyses depend on the cluster features generated in this step

### Step 2: Run Analysis Notebooks (Any Order)

**Classification Tasks:**
- `Binary_Classification.ipynb` - Predicts sustained success (songs staying in Top 200 for ≥8 weeks)
- `multiclass_prediction.ipynb` - Predicts four success categories (brief to lasting hits)
- `multiclass_prediction_3_class.ipynb` - Simplified three-category success prediction

**Regression Task:**
- `regression_v6.ipynb` - Predicts exact number of weeks a song will remain in Top 200

**Analysis & Evaluation:**
- `Ablation_Test.ipynb` - Tests which feature groups contribute most to model performance
- `eda_test.ipynb` - Exploratory data analysis with visualizations and statistical summaries

**Utilities:**
- `Plot_Merge.ipynb` - Combines multiple cluster visualization images into unified figures

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
