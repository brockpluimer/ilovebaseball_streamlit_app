import streamlit as st
import pandas as pd
import numpy as np
from load_data import load_and_prepare_data, load_hof_data
import os
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

@st.cache_data
def load_and_prepare_data_cached(data_type):
    data_df, player_years = load_and_prepare_data(data_type)
    hof_data = load_hof_data()
    merged_data = pd.merge(data_df, hof_data[['IDfg', 'inducted']], on='IDfg', how='left')
    merged_data['inducted'] = merged_data['inducted'].fillna('N')
    merged_data['inducted'] = (merged_data['inducted'] == 'Y').astype(int)
    return merged_data, player_years

def create_pipeline(is_binary, imbalance_method, selected_model, model_options):
    if is_binary:
        if imbalance_method == "SMOTE (Oversampling)":
            return ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('model', model_options[selected_model])
            ])
        elif imbalance_method == "Random Undersampling":
            return ImbPipeline([
                ('scaler', StandardScaler()),
                ('undersampler', RandomUnderSampler(random_state=42)),
                ('model', model_options[selected_model])
            ])
        elif imbalance_method == "Class Weights":
            if hasattr(model_options[selected_model], 'class_weight'):
                model_options[selected_model].set_params(class_weight='balanced')
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_options[selected_model])
            ])
        else:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('model', model_options[selected_model])
            ])
    else:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', model_options[selected_model])
        ])

def train_model(X, y, _pipeline):
    return _pipeline.fit(X, y)

@st.cache_data
def make_predictions(_model, X):
    return _model.predict(X)

def supervised_learning_mode():
    st.title("Supervised Learning Mode")

    data_type = st.sidebar.radio("Select Player Type", ["Hitters", "Pitchers"])
    merged_data, player_years = load_and_prepare_data_cached("Hitter" if data_type == "Hitters" else "Pitcher")

    all_columns = merged_data.columns.tolist()
    target_options = ['Hall of Fame?'] + [col for col in all_columns if col not in ['IDfg', 'Name', 'Team', 'year', 'inducted']]
    
    target_variable = st.selectbox("Select Target Variable", target_options)
    
    if target_variable == 'Hall of Fame?':
        target_variable = 'inducted'

    is_binary = merged_data[target_variable].nunique() == 2

    feature_columns = [col for col in all_columns if col not in ['IDfg', 'Name', 'Team', 'year', target_variable]]
    
    if data_type == "Hitters":
        default_features = ['HR', 'RBI', 'AVG', 'OPS']
    else:  # Pitchers
        default_features = ['W', 'ERA', 'WHIP', 'SO']
    
    default_features = [feat for feat in default_features if feat in feature_columns]
    
    selected_features = st.multiselect("Select Features", feature_columns, default=default_features)

    cleaning_method = st.radio("Choose data cleaning method", 
                               ["Drop rows with missing values", 
                                "Impute missing values with mean", 
                                "Impute missing values with median"])

    imbalance_method = "None"
    if is_binary:
        st.subheader("Class Imbalance Handling")
        imbalance_method = st.selectbox("Choose imbalance handling method", 
                                        ["None", "SMOTE (Oversampling)", "Random Undersampling", "Class Weights"])

    if is_binary:
        model_options = {
            "Support Vector Machine": SVC(probability=True),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
    else:
        model_options = {
            "Support Vector Machine": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge()
        }

    selected_model = st.selectbox("Select Model", list(model_options.keys()))

    if st.button("Run Model"):
        X = merged_data[selected_features]
        y = merged_data[target_variable]

        if cleaning_method == "Drop rows with missing values":
            X = X.dropna()
            y = y[X.index]
        else:
            imputer = SimpleImputer(strategy='mean' if cleaning_method == "Impute missing values with mean" else 'median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_binary else None)

        pipeline = create_pipeline(is_binary, imbalance_method, selected_model, model_options)
        trained_model = train_model(X_train, y_train, pipeline)

        y_pred = make_predictions(trained_model, X_test)

        if is_binary:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, trained_model.predict_proba(X_test)[:, 1])
            
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
            st.write(f"AUC-ROC: {auc_roc:.2f}")
            
            plot_confusion_matrix(y_test, y_pred)
            plot_roc_curve(trained_model, X_test, y_test)
        else:
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared: {r2:.2f}")
            
            plot_actual_vs_predicted(y_test, y_pred)

        if selected_model == "Support Vector Machine" and len(selected_features) == 2:
            plot_svm_decision_boundary(trained_model, X_test, y_test, selected_features)
        elif "Nearest Neighbors" in selected_model and len(selected_features) == 2:
            plot_knn_decision_boundary(trained_model, X_test, y_test, selected_features)
        elif selected_model in ["Logistic Regression", "Linear Regression", "Lasso Regression", "Ridge Regression"]:
            plot_coefficient_importance(trained_model.named_steps['model'], selected_features)
        elif "Random Forest" in selected_model:
            plot_feature_importance(trained_model.named_steps['model'], selected_features)
            plot_tree_visualization(trained_model.named_steps['model'], selected_features)

        plot_feature_selection(X_train, y_train, selected_features)
        plot_learning_curve(trained_model, X_train, y_train)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, text_auto=True, title="Confusion Matrix")
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    st.plotly_chart(fig)

def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC curve (AUC = {roc_auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    st.plotly_chart(fig)

def plot_actual_vs_predicted(y_true, y_pred):
    fig = px.scatter(x=y_true, y=y_pred, title="Actual vs Predicted")
    fig.add_trace(go.Scatter(x=[min(y_true), max(y_true)], y=[min(y_true), max(y_true)], mode='lines', name='Ideal'))
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    st.plotly_chart(fig)

def plot_svm_decision_boundary(model, X, y, feature_names):
    if len(feature_names) != 2:
        st.write("SVM decision boundary can only be plotted for 2 features.")
        return

    x0, x1 = X.iloc[:, 0], X.iloc[:, 1]
    xx, yy = np.meshgrid(np.linspace(x0.min()-1, x0.max()+1, 100),
                         np.linspace(x1.min()-1, x1.max()+1, 100))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=[
        go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdBu', opacity=0.8, name='Decision Boundary'),
        go.Scatter(x=x0[y==0], y=x1[y==0], mode='markers', name='Class 0'),
        go.Scatter(x=x0[y==1], y=x1[y==1], mode='markers', name='Class 1')
    ])
    fig.update_layout(title='SVM Decision Boundary',
                      xaxis_title=feature_names[0],
                      yaxis_title=feature_names[1])
    st.plotly_chart(fig)

def plot_knn_decision_boundary(pipeline, X, y, feature_names):
    if len(feature_names) != 2:
        st.write("KNN decision boundary can only be plotted for 2 features.")
        return

    # Extract feature names
    feature1, feature2 = feature_names

    # Create a mesh grid
    x0, x1 = X[feature1], X[feature2]
    xx, yy = np.meshgrid(np.linspace(x0.min()-1, x0.max()+1, 100),
                         np.linspace(x1.min()-1, x1.max()+1, 100))
    
    # Create a DataFrame with the mesh grid points
    mesh_data = pd.DataFrame({feature1: xx.ravel(), feature2: yy.ravel()})
    
    # Predict using the pipeline
    Z = pipeline.predict(mesh_data)
    Z = Z.reshape(xx.shape)

    fig = go.Figure(data=[
        go.Contour(x=xx[0], y=yy[:, 0], z=Z, colorscale='RdYlBu', opacity=0.8, name='Decision Boundary'),
        go.Scatter(x=x0[y==0], y=x1[y==0], mode='markers', name='Class 0'),
        go.Scatter(x=x0[y==1], y=x1[y==1], mode='markers', name='Class 1')
    ])
    fig.update_layout(title='KNN Decision Boundary',
                      xaxis_title=feature1,
                      yaxis_title=feature2)
    st.plotly_chart(fig)

def plot_coefficient_importance(model, feature_names):
    coefficients = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    importance = np.abs(coefficients)
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    fig = px.bar(feature_importance, x='feature', y='importance', title='Feature Importance (Coefficient Magnitude)')
    st.plotly_chart(fig)

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig = px.bar(x=[feature_names[i] for i in indices], y=importances[indices], title="Feature Importance")
    fig.update_xaxes(title="Features")
    fig.update_yaxes(title="Importance")
    st.plotly_chart(fig)

def plot_tree_visualization(model, feature_names):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model.estimators_[0], feature_names=feature_names, filled=True, ax=ax)
    st.pyplot(fig)

def plot_feature_selection(X, y, feature_names):
    estimator = SVR(kernel="linear")
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)

    fig = go.Figure(data=[
        go.Scatter(x=range(1, len(selector.grid_scores_) + 1), y=selector.grid_scores_,
                   mode='lines+markers')
    ])
    fig.update_layout(title='Recursive Feature Elimination with Cross-Validation',
                      xaxis_title='Number of Features',
                      yaxis_title='Cross-Validation Score')
    st.plotly_chart(fig)

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_sizes, y=train_scores_mean,
                             mode='lines+markers', name='Training score',
                             error_y=dict(type='data', array=train_scores_std, visible=True)))
    fig.add_trace(go.Scatter(x=train_sizes, y=test_scores_mean,
                             mode='lines+markers', name='Cross-validation score',
                             error_y=dict(type='data', array=test_scores_std, visible=True)))
    fig.update_layout(title='Learning Curve',
                      xaxis_title='Training Examples',
                      yaxis_title='Score')
    st.plotly_chart(fig)

if __name__ == "__main__":
    supervised_learning_mode()