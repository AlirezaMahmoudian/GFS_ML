# GFS_ML
# GFS_ML: Genetic-Based Feature Selection for Predicting the Punching Shear Strength of Fiber-Reinforced Concrete Slabs Using Tree Ensemble Machine Learning Models

This software provides a comprehensive platform for performing feature selection in tree-based machine learning models using genetic algorithms. The software streamlines the processes of importing, preprocessing, and analyzing datasets, as well as training machine learning models, tuning hyperparameters, and evaluating their performance.

## Features

- **Dataset Importation**: Supports CSV and Excel files for easy data import.
- **Data Visualization**: Generate scatter plots and correlation matrices to understand feature relationships.
- **Data Preprocessing**: Convert text data to numerical format with label encoding.
- **Model Selection**: Choose from various tree-based models like Decision Trees, Random Forests, Gradient Boosting, Extra Trees, AdaBoost, XGBoost, CatBoost, and LightGBM.
- **Feature Selection using Genetic Algorithm**: Optimize feature selection using genetic algorithms by setting parameters such as initial population size, number of genes, mutation rate, and number of epochs.
- **Hyperparameter Tuning**: Use grid search to find the best hyperparameters for the selected model.
- **SHAP Values for Model Interpretation**: Calculate and display SHAP values to understand the contribution of each feature to the model's predictions.
- **Real-time Predictions**: Input feature values and get real-time predictions from the trained model.

## Installation

To install and run the software, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```sh
    python main.py
    ```

## Usage

1. **Import Dataset**:
   - Click on the "Import Dataset" button and select your CSV or Excel file.
   - The dataset will be displayed in a table for inspection.

2. **Data Visualization**:
   - Use the "Scatter Plot" and "Heatmap Correlation" buttons to visualize the dataset.
   - Preprocess the data using the "Preprocess Data" button to convert text data to numerical format.

3. **Model Selection and Feature Selection**:
   - Select your desired tree-based machine learning model.
   - Set parameters for the genetic algorithm (initial population size, number of genes, mutation rate, and number of epochs).
   - Click on "Find the Best Features" to perform feature selection using the genetic algorithm.

4. **Hyperparameter Tuning**:
   - Use the "Tune Hyperparameters" option to perform grid search and find the best hyperparameters for the selected model.

5. **Model Interpretation with SHAP Values**:
   - Click on "Show SHAP Values" to calculate and display SHAP values, providing insights into feature contributions.

6. **Real-time Predictions**:
   - Enter values for the selected features and click on the "Predict" button to see the model's prediction.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [your-email@example.com](mailto:your-email@example.com).

---

**Note**: Make sure to update the repository URL, email address, and any other placeholders with your actual information before uploading the README file to GitHub.
