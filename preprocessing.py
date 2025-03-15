import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import logging

def configure_logging():
    """
    Configure logging for the preprocessing pipeline
    
    Returns:
    --------
    logging.Logger: Configured logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def preprocess_data_for_rl(
    file_path, 
    test_size=0.2, 
    val_size=0.25, 
    random_state=42
):
    """
    Preprocess agricultural data for reinforcement learning water allocation model
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    test_size : float, default=0.2
        Proportion of data for testing
    val_size : float, default=0.25
        Proportion of remaining data for validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict: Processed datasets and preprocessing metadata
    """
    # Configure logging
    logger = configure_logging()
    logger.info("Starting advanced data preprocessing...")
    
    # Step 1: Load the data
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise
    
    # Step 2: Drop the label column
    if 'label' in df.columns:
        logger.info("Dropping 'label' column")
        df = df.drop(columns=['label'])
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Step 3: Handle missing values
    logger.info("Handling missing values...")
    
    # Numerical imputation with median
    num_imputer = SimpleImputer(strategy='median')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    
    # Categorical imputation with most frequent
    cat_imputer = None
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    
    # Step 4: Encode categorical variables
    logger.info("Processing categorical features...")
    categorical_to_encode = []
    
    # Custom encoding mappings
    encoding_maps = {
        'soil_type': {1: 'sandy', 2: 'loamy', 3: 'clay'},
        'growth_stage': {1: 'seedling', 2: 'vegetative', 3: 'flowering'},
        'water_source_type': {1: 'river', 2: 'groundwater', 3: 'recycled'}
    }
    
    for col, mapping in encoding_maps.items():
        if col in df.columns:
            if df[col].dtype != 'object':
                df[col] = df[col].map(mapping)
            categorical_to_encode.append(col)
    
    # One-hot encoding
    encoded_features = {}
    if categorical_to_encode:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded_cats = encoder.fit_transform(df[categorical_to_encode])
        
        # Create DataFrame with encoded categories
        encoded_cols = encoder.get_feature_names_out(categorical_to_encode)
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df.index)
        
        # Store encoder information
        encoded_features = {
            'encoder': encoder,
            'categorical_cols': categorical_to_encode,
            'encoded_cols': encoded_cols.tolist()
        }
        
        # Drop original categorical columns and add encoded ones
        df = df.drop(columns=categorical_to_encode)
        df = pd.concat([df, encoded_df], axis=1)
    
    # Step 5: Scale features
    logger.info("Scaling numerical features...")
    
    # Identify current numerical columns
    current_numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Use StandardScaler for features with potentially different scales
    scaler = StandardScaler()
    df[current_numerical_cols] = scaler.fit_transform(df[current_numerical_cols])
    
    # Step 6: Split data
    logger.info("Splitting data into train, validation, and test sets...")
    
    # First split out the test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Then split the remaining data into training and validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=random_state
    )
    
    logger.info(f"Train set: {train_df.shape[0]} samples")
    logger.info(f"Validation set: {val_df.shape[0]} samples")
    logger.info(f"Test set: {test_df.shape[0]} samples")
    
    # Prepare RL-specific data structure
    rl_data = {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'feature_names': df.columns.tolist(),
        'scaler': scaler,
        'categorical_info': encoded_features,
        'preprocessing_info': {
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer
        }
    }
    
    logger.info("Preprocessing complete!")
    logger.info(f"Final feature set includes {len(df.columns)} features")
    
    return rl_data

def prepare_rl_environment_data(rl_data, state_features=None):
    """
    Prepare data specifically for the reinforcement learning environment
    
    Parameters:
    -----------
    rl_data : dict
        Output from the preprocess_data_for_rl function
    state_features : list, optional
        Specific features to include in the state representation
    
    Returns:
    --------
    dict: Data prepared for RL environment
    """
    logger = configure_logging()
    logger.info("Preparing data structure for RL environment...")
    
    # Define core environmental features
    core_features = [
        'soil_moisture', 'rainfall', 'water_usage_efficiency', 
        'temperature', 'humidity', 'ph'
    ]
    
    # Identify categorical encoded features
    soil_features = [col for col in rl_data['feature_names'] if 'soil_type' in col]
    growth_features = [col for col in rl_data['feature_names'] if 'growth_stage' in col]
    water_source_features = [col for col in rl_data['feature_names'] if 'water_source_type' in col]
    
    # Additional environmental factors
    env_features = [
        'sunlight_exposure', 'wind_speed', 'co2_concentration', 
        'organic_matter', 'frost_risk'
    ]
    
    # Nutrient features
    nutrient_features = ['N', 'P', 'K']
    
    # Combine and filter existing features
    if state_features is None:
        state_features = list(set(
            core_features + 
            soil_features + 
            growth_features + 
            water_source_features + 
            [f for f in env_features if f in rl_data['feature_names']] +
            [f for f in nutrient_features if f in rl_data['feature_names']]
        ))
    
    # Filter features that exist in the data
    state_features = [f for f in state_features if f in rl_data['feature_names']]
    
    logger.info("Selected features for RL state representation:")
    for feature in state_features:
        logger.info(f"  - {feature}")
    
    # Create filtered datasets
    env_data = {
        'train': rl_data['train'][state_features],
        'val': rl_data['val'][state_features],
        'test': rl_data['test'][state_features],
        'state_features': state_features,
        'scaler': rl_data['scaler'],
        'categorical_info': rl_data['categorical_info']
    }
    
    # Convert to numpy arrays for faster access
    env_data['train_array'] = env_data['train'].values
    env_data['val_array'] = env_data['val'].values
    env_data['test_array'] = env_data['test'].values
    
    return env_data