import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Page configuration
st.set_page_config(page_title="Linear Regression Calculator", layout="wide")

# Title
st.title("📊 Multiple Linear Regression Calculator")
st.markdown("### Analyze relationships between variables using regression analysis")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose Analysis Type", 
                            ["Simple Linear Regression", 
                             "Multiple Linear Regression",
                             "Make Predictions"])

# ============= SIMPLE LINEAR REGRESSION =============
if page == "Simple Linear Regression":
    st.header("Simple Linear Regression")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file (2 columns: X, Y)", type=['csv'])
    
    if uploaded_file is not None:
        # Read data
        try:
            data = pd.read_csv(uploaded_file, sep=';')
            if len(data.columns) == 1:
                data = pd.read_csv(uploaded_file)
        except:
            data = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(data.head())
        
        # Select columns
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X variable (Independent)", data.columns)
        with col2:
            y_col = st.selectbox("Select Y variable (Dependent)", data.columns)
        
        if st.button("Run Simple Regression"):
            X = data[x_col].values.reshape(-1, 1)
            Y = data[y_col].values
            
            # Calculate regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            model = LinearRegression()
            model.fit(X, Y)
            
            slope = model.coef_[0]
            intercept = model.intercept_
            Y_pred = model.predict(X)
            
            # Calculate R²
            r2 = r2_score(Y, Y_pred)
            
            # Display results
            st.success("✅ Regression Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Slope (b)", f"{slope:.4f}")
            with col2:
                st.metric("Intercept (a)", f"{intercept:.4f}")
            with col3:
                st.metric("R² Score", f"{r2:.4f}")
            
            st.info(f"**Equation:** Y = {intercept:.2f} + {slope:.4f}X")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, Y, alpha=0.6, label='Actual Data')
            ax.plot(X, Y_pred, color='red', linewidth=2, label='Regression Line')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title('Simple Linear Regression')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

# ============= MULTIPLE LINEAR REGRESSION =============
elif page == "Multiple Linear Regression":
    st.header("Multiple Linear Regression")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Detect delimiter
        try:
            data = pd.read_csv(uploaded_file, sep=';')
            if len(data.columns) == 1:
                data = pd.read_csv(uploaded_file)
        except:
            data = pd.read_csv(uploaded_file)
        
        st.subheader("Data Preview")
        st.dataframe(data.head(10))
        
        st.subheader("Select Variables")
        
        # Select Y variable
        y_col = st.selectbox("Select Y variable (Dependent)", data.columns)
        
        # Select X variables
        x_cols = st.multiselect("Select X variables (Independent)", 
                                [col for col in data.columns if col != y_col],
                                default=[col for col in data.columns if col != y_col][:2])
        
        if len(x_cols) >= 2 and st.button("Run Multiple Regression"):
            # Extract data
            Y = data[y_col].values
            n = len(data)
            
            # Manual calculation for 2 variables using DEVIATION FORMULAS
            if len(x_cols) == 2:
                X1 = data[x_cols[0]].values
                X2 = data[x_cols[1]].values
                
                # Calculate RAW sums (Capital Sigma)
                SUM_X1 = np.sum(X1)
                SUM_X2 = np.sum(X2)
                SUM_Y = np.sum(Y)
                SUM_X1_squared = np.sum(X1**2)
                SUM_X2_squared = np.sum(X2**2)
                SUM_Y_squared = np.sum(Y**2)
                SUM_X1X2 = np.sum(X1 * X2)
                SUM_X1Y = np.sum(X1 * Y)
                SUM_X2Y = np.sum(X2 * Y)
                
                # Calculate DEVIATION sums (lowercase sigma)
                sum_x1_squared = SUM_X1_squared - (SUM_X1**2 / n)
                sum_x2_squared = SUM_X2_squared - (SUM_X2**2 / n)
                sum_y_squared = SUM_Y_squared - (SUM_Y**2 / n)
                sum_x1y = SUM_X1Y - (SUM_X1 * SUM_Y / n)
                sum_x2y = SUM_X2Y - (SUM_X2 * SUM_Y / n)
                sum_x1x2 = SUM_X1X2 - (SUM_X1 * SUM_X2 / n)
                
                # Calculate coefficients using normal equations with DEVIATION sums
                numerator_b1 = (sum_x2_squared * sum_x1y) - (sum_x2y * sum_x1x2)
                denominator = (sum_x1_squared * sum_x2_squared) - (sum_x1x2**2)
                b1 = numerator_b1 / denominator
                
                numerator_b2 = (sum_x1_squared * sum_x2y) - (sum_x1y * sum_x1x2)
                b2 = numerator_b2 / denominator
                
                a = (SUM_Y - (b1 * SUM_X1) - (b2 * SUM_X2)) / n
                
                # Calculate R² using DEVIATION sums
                R2 = (b1 * sum_x1y + b2 * sum_x2y) / sum_y_squared
                
                # Predictions
                Y_pred = a + b1 * X1 + b2 * X2
                
                # Display results
                st.success("✅ Multiple Regression Complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Intercept (a)", f"{a:.4f}")
                with col2:
                    st.metric(f"b₁ ({x_cols[0]})", f"{b1:.6f}")
                with col3:
                    st.metric(f"b₂ ({x_cols[1]})", f"{b2:.6f}")
                with col4:
                    st.metric("R² Score", f"{R2:.4f}")
                
                st.info(f"**Equation:** Y = {a:.2f} + {b1:.6f}X₁ + {b2:.6f}X₂")
                st.write(f"The model explains **{R2*100:.1f}%** of the variance in {y_col}")
                
                # Show detailed calculation
                with st.expander("📊 View Detailed Calculations"):
                    st.write("**Raw Sums (Capital Σ):**")
                    calc_data = pd.DataFrame({
                        'Variable': ['ΣX₁', 'ΣX₂', 'ΣY', 'ΣX₁²', 'ΣX₂²', 'ΣY²', 'ΣX₁Y', 'ΣX₂Y', 'ΣX₁X₂'],
                        'Value': [f"{SUM_X1:.2f}", f"{SUM_X2:.2f}", f"{SUM_Y:.2f}", 
                                 f"{SUM_X1_squared:.2f}", f"{SUM_X2_squared:.2f}", f"{SUM_Y_squared:.2f}",
                                 f"{SUM_X1Y:.2f}", f"{SUM_X2Y:.2f}", f"{SUM_X1X2:.2f}"]
                    })
                    st.dataframe(calc_data)
                    
                    st.write("**Deviation Sums (lowercase Σ):**")
                    dev_data = pd.DataFrame({
                        'Variable': ['Σx₁²', 'Σx₂²', 'Σy²', 'Σx₁y', 'Σx₂y', 'Σx₁x₂'],
                        'Formula': ['ΣX₁² - (ΣX₁)²/n', 'ΣX₂² - (ΣX₂)²/n', 'ΣY² - (ΣY)²/n',
                                   'ΣX₁Y - (ΣX₁)(ΣY)/n', 'ΣX₂Y - (ΣX₂)(ΣY)/n', 'ΣX₁X₂ - (ΣX₁)(ΣX₂)/n'],
                        'Value': [f"{sum_x1_squared:.4f}", f"{sum_x2_squared:.4f}", f"{sum_y_squared:.4f}",
                                 f"{sum_x1y:.4f}", f"{sum_x2y:.4f}", f"{sum_x1x2:.4f}"]
                    })
                    st.dataframe(dev_data)
                    
                    st.write("**Coefficient Calculations:**")
                    st.latex(r'b_1 = \frac{(\Sigma x_2^2 \times \Sigma x_1y) - (\Sigma x_2y \times \Sigma x_1x_2)}{(\Sigma x_1^2 \times \Sigma x_2^2) - (\Sigma x_1x_2)^2}')
                    st.latex(r'b_2 = \frac{(\Sigma x_1^2 \times \Sigma x_2y) - (\Sigma x_1y \times \Sigma x_1x_2)}{(\Sigma x_1^2 \times \Sigma x_2^2) - (\Sigma x_1x_2)^2}')
                    st.latex(r'a = \frac{\Sigma Y - (b_1 \times \Sigma X_1) - (b_2 \times \Sigma X_2)}{n}')
                    
                    st.write("**R² Calculation:**")
                    st.latex(r'R^2 = \frac{b_1 \Sigma x_1y + b_2 \Sigma x_2y}{\Sigma y^2}')
                    st.write(f"R² = ({b1:.6f} × {sum_x1y:.4f} + {b2:.6f} × {sum_x2y:.4f}) / {sum_y_squared:.4f}")
                    st.write(f"R² = {R2:.4f}")
                
                # Store in session state for predictions
                st.session_state['model_params'] = {
                    'a': a, 'b1': b1, 'b2': b2,
                    'x1_name': x_cols[0], 'x2_name': x_cols[1], 'y_name': y_col,
                    'R2': R2
                }
                
                # Visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Actual vs Predicted", "Residual Plot", "3D Visualization", "Data Table"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(Y, Y_pred, alpha=0.6)
                    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Perfect Prediction')
                    ax.set_xlabel(f'Actual {y_col}')
                    ax.set_ylabel(f'Predicted {y_col}')
                    ax.set_title('Actual vs Predicted Values')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with tab2:
                    residuals = Y - Y_pred
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(Y_pred, residuals, alpha=0.6, color='purple')
                    ax.axhline(y=0, color='r', linestyle='--', lw=2)
                    ax.set_xlabel('Predicted Values')
                    ax.set_ylabel('Residuals')
                    ax.set_title('Residual Plot')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                with tab3:
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(X1, X2, Y, c='blue', marker='o', alpha=0.6, label='Actual Data')
                    
                    # Regression plane
                    X1_range = np.linspace(X1.min(), X1.max(), 10)
                    X2_range = np.linspace(X2.min(), X2.max(), 10)
                    X1_mesh, X2_mesh = np.meshgrid(X1_range, X2_range)
                    Y_mesh = a + b1 * X1_mesh + b2 * X2_mesh
                    ax.plot_surface(X1_mesh, X2_mesh, Y_mesh, alpha=0.3, color='red')
                    
                    ax.set_xlabel(x_cols[0])
                    ax.set_ylabel(x_cols[1])
                    ax.set_zlabel(y_col)
                    ax.set_title('3D Regression Plane')
                    ax.legend()
                    st.pyplot(fig)
                
                with tab4:
                    results_table = pd.DataFrame({
                        x_cols[0]: X1,
                        x_cols[1]: X2,
                        f'Actual {y_col}': Y,
                        f'Predicted {y_col}': Y_pred,
                        'Residuals': residuals
                    })
                    st.dataframe(results_table)
                    
                    # Download button
                    csv = results_table.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="regression_results.csv",
                        mime="text/csv"
                    )

# ============= MAKE PREDICTIONS =============
elif page == "Make Predictions":
    st.header("Make Predictions")
    
    if 'model_params' in st.session_state:
        params = st.session_state['model_params']
        
        st.write(f"**Current Model:**")
        st.latex(f"{params['y_name']} = {params['a']:.2f} + {params['b1']:.6f} \\times {params['x1_name']} + {params['b2']:.6f} \\times {params['x2_name']}")
        st.write(f"**R² = {params['R2']:.4f}** ({params['R2']*100:.1f}% of variance explained)")
        
        st.divider()
        
        st.subheader("Enter Values for Prediction")
        
        col1, col2 = st.columns(2)
        with col1:
            x1_input = st.number_input(f"Enter {params['x1_name']} value", value=5.0, step=0.1)
        with col2:
            x2_input = st.number_input(f"Enter {params['x2_name']} value", value=1000.0, step=10.0)
        
        if st.button("🔮 Predict", type="primary"):
            prediction = params['a'] + params['b1'] * x1_input + params['b2'] * x2_input
            
            st.success(f"### Predicted {params['y_name']}: **{prediction:.2f}**")
            
            # Show calculation
            with st.expander("View Calculation Steps"):
                st.write(f"{params['y_name']} = {params['a']:.2f} + {params['b1']:.6f} × {x1_input} + {params['b2']:.6f} × {x2_input}")
                st.write(f"{params['y_name']} = {params['a']:.2f} + {params['b1'] * x1_input:.2f} + {params['b2'] * x2_input:.2f}")
                st.write(f"{params['y_name']} = {prediction:.2f}")
        
        st.divider()
        
        # Batch predictions
        st.subheader("Batch Predictions")
        st.write("Upload a CSV file with columns matching your independent variables")
        
        batch_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'], key='batch')
        
        if batch_file is not None:
            try:
                batch_data = pd.read_csv(batch_file, sep=';')
                if len(batch_data.columns) == 1:
                    batch_data = pd.read_csv(batch_file)
                
                if params['x1_name'] in batch_data.columns and params['x2_name'] in batch_data.columns:
                    predictions = params['a'] + params['b1'] * batch_data[params['x1_name']] + params['b2'] * batch_data[params['x2_name']]
                    batch_data[f'Predicted_{params["y_name"]}'] = predictions
                    
                    st.dataframe(batch_data)
                    
                    csv = batch_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"CSV must contain columns: {params['x1_name']} and {params['x2_name']}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    else:
        st.warning("⚠️ Please run a Multiple Linear Regression first!")
        st.info("Go to the 'Multiple Linear Regression' page and analyze your data to create a model.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**About This App**

This calculator uses the deviation formula method for multiple linear regression:
- Accurate coefficient calculation
- Proper R² computation
- Interactive visualizations
- Batch predictions

Built with Streamlit 🎈
""")