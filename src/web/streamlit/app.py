import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import requests
import sys
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, accuracy_score, precision_score, 
    recall_score, f1_score, mean_absolute_error, mean_squared_error
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.utils.common import load_model
from src.components.data_preprocessing import TextPreprocessor

# Page config
st.set_page_config(
    page_title="Review Analysis Dashboard",
    page_icon="https://media.getredy.id/images/users/58915/18956148381625815185.png",
    layout="wide"
)

# Add logo to sidebar
st.sidebar.image("https://media.getredy.id/images/users/58915/18956148381625815185.png", width=200)

# Navigation
pages = ["Summary", "Prediction", "Model Evaluation"]
choice = st.sidebar.selectbox("Navigate", pages)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/Womens Clothing E-Commerce Reviews.csv")
        df = df.dropna(subset=['Review Text', 'Rating', 'Recommended IND'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def display_summary_page():
    st.title("E-Commerce Reviews Analysis")
    
    df = load_data()
    if df is not None:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Recommendation Rate", f"{(df['Recommended IND'].mean()*100):.1f}%")
        with col3:
            st.metric("Average Rating", f"{df['Rating'].mean():.1f}")
        
        # Basic Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                df, 
                x="Rating", 
                title="Distribution of Ratings",
                color_discrete_sequence=['#3498db']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.pie(
                df, 
                names="Recommended IND", 
                title="Recommendation Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.subheader("Additional Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating vs Recommendation
            fig = px.box(
                df, 
                x='Recommended IND', 
                y='Rating',
                title="Rating Distribution by Recommendation",
                color='Recommended IND',
                labels={'Recommended IND': 'Recommended', 'Rating': 'Rating'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Department distribution
            dept_counts = df['Department Name'].value_counts()
            fig = px.bar(
                x=dept_counts.index,
                y=dept_counts.values,
                title="Reviews by Department",
                labels={'x': 'Department', 'y': 'Number of Reviews'}
            )
            st.plotly_chart(fig, use_container_width=True)

def display_prediction_page():
    st.title("Review Analysis Prediction")
    
    user_input = st.text_area("Enter your review text:", height=150)
    
    col1, col2 = st.columns(2)
    
    if st.button("Analyze Review"):
        if user_input:
            try:
                with st.spinner("Analyzing review..."):
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={"text": user_input}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Recommendation prediction
                        with col1:
                            st.write("### Recommendation Prediction")
                            if result["recommendation"]:
                                st.success("✨ This review is predicted to be RECOMMENDED")
                            else:
                                st.warning("⚠️ This review is predicted to be NOT RECOMMENDED")
                            st.info(f"Confidence: {result['recommendation_confidence']:.2%}")
                        
                        # Sentiment prediction
                        with col2:
                            st.write("### Sentiment Prediction")
                            sentiment_color = {
                                1: "🔴",
                                2: "🟠",
                                3: "🟡",
                                4: "🟢",
                                5: "💚"
                            }
                            
                            st.write(f"### Predicted Rating: {sentiment_color[result['sentiment']]} {result['sentiment']}/5")
                            st.info(f"Confidence: {result['sentiment_confidence']:.2%}")
                        
                        # Text analysis
                        with st.expander("View Text Analysis"):
                            st.write("**Original Text:**")
                            st.write(user_input)
                            st.write("**Processed Text:**")
                            st.code(result["cleaned_text"])
                            
                        # Sentiment gauge
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = result['sentiment'],
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Sentiment Rating"},
                            gauge = {
                                'axis': {'range': [1, 5]},
                                'steps': [
                                    {'range': [1, 2], 'color': "lightgray"},
                                    {'range': [2, 3], 'color': "gray"},
                                    {'range': [3, 4], 'color': "lightgreen"},
                                    {'range': [4, 5], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': result['sentiment']
                                }
                            }
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
        else:
            st.warning("Please enter some text to analyze")
    

def display_recommendation_metrics(df, y_true, y_pred, y_prob):
    st.subheader("Recommendation Model Performance")
    
    # Basic metrics
    metrics = classification_report(y_true, y_pred, output_dict=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{metrics['weighted avg']['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{metrics['weighted avg']['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['weighted avg']['f1-score']:.3f}")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm / cm.sum() * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Not Recommended', 'Recommended'],
        y=['Not Recommended', 'Recommended'],
        colorscale='RdBu',
        text=[[f'Count: {val}<br>Percentage: {percent:.1f}%' 
              for val, percent in zip(row, row_percent)] 
             for row, row_percent in zip(cm, cm_percent)],
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROC and PR Curves
    col1, col2 = st.columns(2)
    
    with col1:
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f'ROC curve (AUC = {roc_auc:.3f})',
            mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            name=f'PR curve (AP = {pr_auc:.3f})',
            mode='lines'
        ))
        
        fig.update_layout(
            title='Precision-Recall Curve',
            xaxis_title='Recall',
            yaxis_title='Precision'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_sentiment_metrics(df, y_true, y_pred, y_prob):
    st.subheader("Sentiment Model Performance")
    
    # Sentiment metrics
    sent_metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{sent_metrics['accuracy']:.3f}")
    with col2:
        st.metric("MAE", f"{sent_metrics['mae']:.3f}")
    with col3:
        st.metric("RMSE", f"{sent_metrics['rmse']:.3f}")
    
    # Sentiment Confusion Matrix
    st.subheader("Sentiment Confusion Matrix")
    sent_cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=sent_cm,
        x=[str(i) for i in range(1, 6)],
        y=[str(i) for i in range(1, 6)],
        colorscale='RdBu',
        text=sent_cm,
        texttemplate="%{text}",
    ))
    
    fig.update_layout(
        title='Sentiment Confusion Matrix',
        xaxis_title='Predicted Rating',
        yaxis_title='True Rating'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating distributions
    col1, col2 = st.columns(2)
    
    with col1:
        # True vs Predicted Distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=y_true,
            name='True Ratings',
            opacity=0.75,
            nbinsx=5
        ))
        fig.add_trace(go.Histogram(
            x=y_pred,
            name='Predicted Ratings',
            opacity=0.75,
            nbinsx=5
        ))
        
        fig.update_layout(
            title='Distribution of True vs Predicted Ratings',
            xaxis_title='Rating',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction Error Distribution
        df['rating_error'] = y_pred - y_true
        
        fig = px.histogram(
            df,
            x='rating_error',
            title='Distribution of Rating Prediction Errors',
            labels={'rating_error': 'Prediction Error (Predicted - True)',
                   'count': 'Number of Reviews'},
            nbins=9
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_error_analysis(df, y_rec_true, y_rec_pred, y_sent_true, y_sent_pred,
                         y_rec_prob, y_sent_prob):
    st.subheader("Error Analysis")
    
    # Add prediction and confidence columns
    df['predicted_rec'] = y_rec_pred
    df['rec_confidence'] = np.max(y_rec_prob, axis=1)
    df['predicted_rating'] = y_sent_pred
    df['rating_confidence'] = np.max(y_sent_prob, axis=1)
    
    # Error statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Recommendation Errors")
        misclassified_rec = df[df['Recommended IND'] != df['predicted_rec']]
        
        st.metric("Total Misclassified", len(misclassified_rec))
        st.metric("Error Rate", f"{(len(misclassified_rec)/len(df))*100:.2f}%")
        
        false_positives = len(misclassified_rec[
            (misclassified_rec['predicted_rec'] == 1) & 
            (misclassified_rec['Recommended IND'] == 0)
        ])
        false_negatives = len(misclassified_rec[
            (misclassified_rec['predicted_rec'] == 0) & 
            (misclassified_rec['Recommended IND'] == 1)
        ])
        
        st.metric("False Positives", false_positives)
        st.metric("False Negatives", false_negatives)
    
    with col2:
        st.write("### Rating Errors")
        df['rating_error'] = df['predicted_rating'] - df['Rating']
        
        st.metric("Mean Absolute Error", f"{mean_absolute_error(y_sent_true, y_sent_pred):.3f}")
        st.metric("Root Mean Squared Error", f"{np.sqrt(mean_squared_error(y_sent_true, y_sent_pred)):.3f}")
        
        one_off_accuracy = np.mean(abs(df['rating_error']) <= 1)
        st.metric("Within ±1 Rating", f"{one_off_accuracy:.1%}")
    
    # Error Analysis by Confidence
    st.subheader("Error Analysis by Confidence")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df[df['Recommended IND'] != df['predicted_rec']],
            x='rec_confidence',
            title="Confidence Distribution of Misclassified Recommendations",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(
            df,
            x='rating_confidence',
            color=abs(df['rating_error']) > 0,
            title="Confidence Distribution by Rating Error",
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample misclassified examples
    st.subheader("Sample Misclassified Examples")
    
    tab1, tab2 = st.tabs(["Recommendation Errors", "Rating Errors"])
    
    with tab1:
        confidence_threshold = st.slider(
            "Filter by confidence threshold (Recommendation)",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="rec_conf"
        )
        
        high_conf_errors = misclassified_rec[
            misclassified_rec['rec_confidence'] >= confidence_threshold
        ].sort_values('rec_confidence', ascending=False)
        
        if len(high_conf_errors) > 0:
            for _, row in high_conf_errors.head().iterrows():
                with st.expander(f"Confidence: {row['rec_confidence']:.2f}"):
                    st.write("**Original Text:**", row['Review Text'])
                    st.write("**True Label:**", "Recommended" if row['Recommended IND'] == 1 else "Not Recommended")
                    st.write("**Predicted Label:**", "Recommended" if row['predicted_rec'] == 1 else "Not Recommended")
        else:
            st.info("No examples found with the current confidence threshold.")
    
    with tab2:
        rating_threshold = st.slider(
            "Filter by confidence threshold (Rating)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            key="rating_conf"
        )
        
        error_threshold = st.slider(
            "Minimum rating error",
            min_value=1,
            max_value=4,
            value=2,
            key="error_threshold"
        )
        
        high_error_samples = df[
            (df['rating_confidence'] >= rating_threshold) &
            (abs(df['rating_error']) >= error_threshold)
        ].sort_values('rating_confidence', ascending=False)
        
        if len(high_error_samples) > 0:
            for _, row in high_error_samples.head().iterrows():
                with st.expander(f"Confidence: {row['rating_confidence']:.2f}"):
                    st.write("**Original Text:**", row['Review Text'])
                    st.write("**True Rating:**", row['Rating'])
                    st.write("**Predicted Rating:**", row['predicted_rating'])
                    st.write("**Rating Error:**", row['rating_error'])
        else:
            st.info("No examples found with the current thresholds.")

def display_feature_importance(vectorizer, rec_model, sent_model):
    st.subheader("Feature Importance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        try :
            importance = rec_model.feature_importances_
        except:
            importance = rec_model.coef_[0]

        st.write("### Recommendation Model Features")
        rec_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        top_n_rec = st.slider("Number of top features (Recommendation)", 5, 30, 15)
        
        fig = px.bar(
            rec_importance.head(top_n_rec),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n_rec} Features for Recommendation'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Show feature importance table (Recommendation)"):
            st.dataframe(
                rec_importance.head(top_n_rec).style.format({
                    'importance': '{:.4f}'
                })
            )
    
    with col2:
        st.write("### Sentiment Model Features")
        try:
            sentiment_importance = sent_model.feature_importances_
        except:
            sentiment_importance = sent_model.coef_[0]

        sent_importance = pd.DataFrame({
            'feature': vectorizer.get_feature_names_out(),
            'importance': sentiment_importance
        }).sort_values('importance', ascending=False)
        
        top_n_sent = st.slider("Number of top features (Sentiment)", 5, 30, 15)
        
        fig = px.bar(
            sent_importance.head(top_n_sent),
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n_sent} Features for Sentiment'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        if st.checkbox("Show feature importance table (Sentiment)"):
            st.dataframe(
                sent_importance.head(top_n_sent).style.format({
                    'importance': '{:.4f}'
                })
            )
    
    # Feature overlap analysis
    st.subheader("Feature Overlap Analysis")
    
    top_n_overlap = st.slider("Number of features to compare", 10, 50, 20)
    
    top_rec_features = set(rec_importance.head(top_n_overlap)['feature'])
    top_sent_features = set(sent_importance.head(top_n_overlap)['feature'])
    
    common_features = top_rec_features.intersection(top_sent_features)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("### Recommendation-Specific")
        st.write(sorted(list(top_rec_features - common_features)))
    
    with col2:
        st.write("### Common Features")
        st.write(sorted(list(common_features)))
    
    with col3:
        st.write("### Sentiment-Specific")
        st.write(sorted(list(top_sent_features - common_features)))

def display_advanced_analysis(df):
    st.subheader("Advanced Analysis")
    
    # Attribute Analysis
    st.write("### Review Attributes Analysis")
    
    attributes = {
        'size': ['size', 'fit', 'small', 'large', 'tight', 'loose'],
        'quality': ['quality', 'material', 'fabric', 'stitch', 'durable', 'cheap'],
        'comfort': ['comfortable', 'uncomfortable', 'soft', 'wear', 'feel'],
        'style': ['style', 'design', 'fashion', 'look', 'beautiful', 'ugly'],
        'price': ['price', 'expensive', 'cheap', 'cost', 'worth', 'value'],
        'shipping': ['shipping', 'delivery', 'package', 'arrived', 'fast', 'slow']
    }
    
    @st.cache_data
    def extract_attributes(text, attributes):
        result = {attr: 0 for attr in attributes}
        for attr, keywords in attributes.items():
            for keyword in keywords:
                if keyword in text.lower():
                    result[attr] += 1
        return result
    
    # Apply attribute extraction
    df['attributes_dict'] = df['cleaned_text'].apply(
        lambda x: extract_attributes(x, attributes)
    )
    
    # Convert to DataFrame
    attributes_df = pd.DataFrame(df['attributes_dict'].tolist())
    
    # Attribute Summary
    tab1, tab2, tab3 = st.tabs([
        "Attribute Mentions", 
        "Correlation Analysis",
        "Category Analysis"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=attributes_df.sum().index,
                y=attributes_df.sum().values,
                title="Attribute Mentions in Reviews",
                labels={'x': 'Attribute', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attribute mentions by recommendation
            attr_by_rec = pd.DataFrame()
            for attr in attributes_df.columns:
                attr_by_rec.loc[attr, 'Recommended'] = attributes_df[attr][df['Recommended IND'] == 1].mean()
                attr_by_rec.loc[attr, 'Not Recommended'] = attributes_df[attr][df['Recommended IND'] == 0].mean()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Recommended',
                x=attr_by_rec.index,
                y=attr_by_rec['Recommended']
            ))
            fig.add_trace(go.Bar(
                name='Not Recommended',
                x=attr_by_rec.index,
                y=attr_by_rec['Not Recommended']
            ))
            
            fig.update_layout(
                title="Attribute Mentions by Recommendation",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Correlation heatmap
        corr_matrix = attributes_df.corr()
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu",
            title="Attribute Correlation Matrix"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with target variables
        col1, col2 = st.columns(2)
        
        with col1:
            rec_corr = pd.DataFrame()
            for col in attributes_df.columns:
                rec_corr.loc[col, 'correlation'] = attributes_df[col].corr(df['Recommended IND'])
            
            fig = px.bar(
                x=rec_corr.index,
                y=rec_corr['correlation'],
                title="Attribute Correlation with Recommendations",
                labels={'x': 'Attribute', 'y': 'Correlation'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            rating_corr = pd.DataFrame()
            for col in attributes_df.columns:
                rating_corr.loc[col, 'correlation'] = attributes_df[col].corr(df['Rating'])
            
            fig = px.bar(
                x=rating_corr.index,
                y=rating_corr['correlation'],
                title="Attribute Correlation with Ratings",
                labels={'x': 'Attribute', 'y': 'Correlation'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Category-specific analysis
        selected_category = st.selectbox(
            "Select Department",
            options=df['Department Name'].unique()
        )
        
        category_df = df[df['Department Name'] == selected_category]
        category_attrs = pd.DataFrame(category_df['attributes_dict'].tolist())
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                x=category_attrs.sum().index,
                y=category_attrs.sum().values,
                title=f"Attribute Mentions in {selected_category}",
                labels={'x': 'Attribute', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            category_corr = pd.DataFrame()
            for col in category_attrs.columns:
                category_corr.loc[col, 'correlation'] = category_attrs[col].corr(category_df['Rating'])
            
            fig = px.bar(
                x=category_corr.index,
                y=category_corr['correlation'],
                title=f"Attribute Correlation with Ratings in {selected_category}",
                labels={'x': 'Attribute', 'y': 'Correlation'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # Topic Modeling Section
    st.write("### Topic Modeling (LDA)")
    
    num_topics = st.slider("Number of Topics", 3, 10, 5)
    num_words = st.slider("Number of Words per Topic", 5, 15, 10)
    
    # Perform LDA
    lda_vectorizer = TfidfVectorizer(
        max_features=1000, 
        stop_words='english'
    )
    lda_matrix = lda_vectorizer.fit_transform(df['cleaned_text'])
    
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(lda_matrix)
    
    # Display topics
    feature_names = lda_vectorizer.get_feature_names_out()
    
    tab1, tab2 = st.tabs(["Topic Details", "Topic Distribution"])
    
    with tab1:
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [
                feature_names[i] 
                for i in topic.argsort()[:-num_words-1:-1]
            ]
            with st.expander(f"Topic {topic_idx + 1}"):
                word_freq = dict(zip(
                    top_words,
                    sorted(topic[topic.argsort()[:-num_words-1:-1]], reverse=True)
                ))
                
                fig = px.bar(
                    x=list(word_freq.keys()),
                    y=list(word_freq.values()),
                    title=f"Top Words in Topic {topic_idx + 1}"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.write("Top words and their importance scores:")
                for word, score in word_freq.items():
                    st.write(f"- {word}: {score:.4f}")
    
    with tab2:
        # Get document-topic distribution
        doc_topics = lda.transform(lda_matrix)
        
        # Overall topic distribution
        topic_dist = doc_topics.mean(axis=0)
        fig = px.bar(
            x=[f"Topic {i+1}" for i in range(num_topics)],
            y=topic_dist,
            title="Overall Topic Distribution in Reviews",
            labels={'x': 'Topic', 'y': 'Average Probability'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic distribution by rating
        st.write("### Topic Distribution by Rating")
        
        topic_by_rating = {}
        for rating in range(1, 6):
            mask = df['Rating'] == rating
            if mask.any():
                topic_by_rating[f"Rating {rating}"] = doc_topics[mask].mean(axis=0)
        
        topic_dist_df = pd.DataFrame(topic_by_rating).T
        topic_dist_df.columns = [f"Topic {i+1}" for i in range(num_topics)]
        
        fig = px.imshow(
            topic_dist_df,
            labels=dict(x="Topic", y="Rating", color="Probability"),
            color_continuous_scale="Viridis",
            title="Topic Distribution Heatmap by Rating"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_model_evaluation():
    st.title("Model Performance Evaluation")
    
    try:
        # Load models and data
        recommendation_model, sentiment_model, vectorizer = load_model()
        if None in (recommendation_model, sentiment_model, vectorizer):
            st.error("Models not found. Please train the models first.")
            st.stop()
            
        df = load_data()
        if df is None:
            st.error("Could not load data.")
            st.stop()
            
        # Preprocess data
        preprocessor = TextPreprocessor()
        
        with st.spinner("Processing data..."):
            df['cleaned_text'] = df['Review Text'].apply(preprocessor.clean_text)
            df = df[df['cleaned_text'] != ""].reset_index(drop=True)
            
            # Get predictions for both models
            X = vectorizer.transform(df['cleaned_text'])
            
            # Recommendation predictions
            y_rec_true = df['Recommended IND']
            y_rec_pred = recommendation_model.predict(X)
            y_rec_prob = recommendation_model.predict_proba(X)
            
            # Sentiment predictions
            y_sent_true = df['Rating']
            y_sent_pred = sentiment_model.predict(X)
            y_sent_prob = sentiment_model.predict_proba(X)

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Recommendation Metrics", 
            "⭐ Sentiment Metrics",
            "🔍 Error Analysis",
            "📈 Feature Importance",
            "🔬 Advanced Analysis"
        ])
        
        with tab1:
            display_recommendation_metrics(df, y_rec_true, y_rec_pred, y_rec_prob)
            
        with tab2:
            display_sentiment_metrics(df, y_sent_true, y_sent_pred, y_sent_prob)
            
        with tab3:
            display_error_analysis(df, y_rec_true, y_rec_pred, y_sent_true, y_sent_pred,
                               y_rec_prob, y_sent_prob)
            
        with tab4:
            display_feature_importance(vectorizer, recommendation_model, sentiment_model)
            
        with tab5:
            display_advanced_analysis(df)

    except Exception as e:
        st.error(f"Error in model evaluation: {str(e)}")
        st.info("Please make sure the model is trained and saved in the 'models' directory.")

# Main app initialization and logic
def main():
    if choice == "Summary":
        display_summary_page()
    elif choice == "Prediction":
        display_prediction_page()
    elif choice == "Model Evaluation":
        display_model_evaluation()

if __name__ == "__main__":
    main()