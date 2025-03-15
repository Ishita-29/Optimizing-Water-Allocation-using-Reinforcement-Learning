import numpy as np
import logging
import random

def configure_logging():
    """
    Configure logging for the environment
    
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

class WaterAllocationEnvironment:
    """
    Reinforcement Learning Environment for Water Allocation - Improved Version
    """
    def __init__(self, env_data):
        """
        Initialize the environment
        
        Parameters:
        -----------
        env_data : dict
            Preprocessed environment data
        """
        # Configure logging
        self.logger = configure_logging()
        
        # Training data
        self.train_data = env_data['train_array']
        self.state_features = env_data['state_features']
        
        # Environment parameters
        self.current_step = 0
        self.max_steps = len(self.train_data)
        self.episode_count = 0
        
        # Track cumulative soil moisture for realistic simulation
        self.current_soil_moisture = 0.0
        
        # Define action space (water allocation levels)
        self.action_space = [
            0.0,   # No water
            0.25,  # Low water
            0.5,   # Medium water
            0.75,  # High water
            1.0    # Full water
        ]
        
        # State dimensions
        self.state_dim = self.train_data.shape[1]
        
        # Find key feature indices
        self.feature_indices = {}
        for feature in ['soil_moisture', 'water_usage_efficiency', 'temperature', 
                        'humidity', 'rainfall', 'N', 'P', 'K', 'sunlight_exposure',
                        'wind_speed', 'ph']:
            if feature in self.state_features:
                self.feature_indices[feature] = self.state_features.index(feature)
            else:
                self.logger.warning(f"Feature {feature} not found in state features. Environment might not work correctly.")
        
        # Initialize environment dynamics parameters
        self.water_effect_factors = {
            'soil_moisture': 0.7,     # Direct effect
            'temperature': -0.2,      # Negative effect (cooling)
            'humidity': 0.3,          # Positive effect (increases humidity)
        }
        
        # Soil type effects
        self.soil_type_indices = [i for i, feat in enumerate(self.state_features) if 'soil_type' in feat]
        
        # Historical data for evaluation
        self.performance_history = []
        
        self.logger.info(f"Improved environment initialized with {self.max_steps} steps")
        self.logger.info(f"State dimension: {self.state_dim}")
        self.logger.info(f"Found key features: {list(self.feature_indices.keys())}")
    
    def reset(self):
        """
        Reset the environment to initial state with randomization
        
        Returns:
        --------
        numpy.ndarray: Initial state
        """
        # Increment episode counter
        self.episode_count += 1
        
        # Start at a random position in the dataset for more variety
        self.current_step = random.randint(0, max(0, self.max_steps - 100))
        
        # Get initial state with small noise for variance
        initial_state = self.train_data[self.current_step].copy()
        
        # Add small noise to create variability (1-5% of the original value)
        noise_factor = 0.01 + (0.04 * random.random())  # Random between 1% and 5%
        noise = noise_factor * np.random.normal(0, 1, self.state_dim)
        initial_state = initial_state + noise
        
        # Set initial soil moisture based on rainfall and baseline
        if 'soil_moisture' in self.feature_indices and 'rainfall' in self.feature_indices:
            self.current_soil_moisture = initial_state[self.feature_indices['soil_moisture']]
        else:
            self.current_soil_moisture = 0.5  # Default value
            
        return initial_state
    
    def step(self, action_index):
        """
        Take a step in the environment with realistic dynamics
        
        Parameters:
        -----------
        action_index : int
            Index of the selected action
        
        Returns:
        --------
        tuple: (next_state, reward, done, info)
        """
        # Get water allocation amount
        water_amount = self.action_space[action_index]
        
        # Save current state
        current_state = self.train_data[self.current_step].copy()
        
        # Advance to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps - 1
        
        if done:
            return current_state, 0, done, {}
        
        # Get next state
        next_state = self.train_data[self.current_step].copy()
        
        # Apply water effects with realistic dynamics
        next_state = self._apply_water_effects(next_state, water_amount, current_state)
        
        # Calculate reward with multiple factors
        reward = self._calculate_reward(current_state, next_state, water_amount)
        
        # Add small random noise to next state for increased variance
        next_state += np.random.normal(0, 0.02, self.state_dim)
        
        # Info dictionary for debugging
        info = {
            'water_amount': water_amount,
            'soil_moisture': next_state[self.feature_indices['soil_moisture']] if 'soil_moisture' in self.feature_indices else None,
        }
        
        return next_state, reward, done, info
    
    def _apply_water_effects(self, next_state, water_amount, current_state):
        """
        Apply the effects of water allocation to the next state
        
        Parameters:
        -----------
        next_state : numpy.ndarray
            The next state to modify
        water_amount : float
            Amount of water allocated
        current_state : numpy.ndarray
            Current state for reference
            
        Returns:
        --------
        numpy.ndarray: Modified next state
        """
        # Update soil moisture based on:
        # 1. Current soil moisture
        # 2. Water allocation
        # 3. Natural factors (rainfall, evaporation from temperature)
        if 'soil_moisture' in self.feature_indices:
            soil_moisture_idx = self.feature_indices['soil_moisture']
            
            # Base soil moisture from dataset
            base_soil_moisture = next_state[soil_moisture_idx]
            
            # Calculate evaporation based on temperature
            evaporation = 0.0
            if 'temperature' in self.feature_indices:
                temp_idx = self.feature_indices['temperature']
                temperature = current_state[temp_idx]
                evaporation = max(0, 0.05 * (temperature - 0.5))  # Higher temps cause more evaporation
            
            # Calculate water retention based on soil type
            retention_factor = 0.7  # Default
            for soil_idx in self.soil_type_indices:
                if current_state[soil_idx] > 0.5:  # If soil type is present
                    if 'sandy' in self.state_features[soil_idx]:
                        retention_factor = 0.5  # Sandy soil retains less water
                    elif 'clay' in self.state_features[soil_idx]:
                        retention_factor = 0.9  # Clay soil retains more water
                    elif 'loamy' in self.state_features[soil_idx]:
                        retention_factor = 0.7  # Loamy soil has medium retention
            
            # Calculate rainfall contribution
            rainfall_factor = 0.0
            if 'rainfall' in self.feature_indices:
                rainfall_idx = self.feature_indices['rainfall']
                rainfall = current_state[rainfall_idx]
                rainfall_factor = rainfall * 0.3  # Rainfall contributes to soil moisture
            
            # Calculate new soil moisture with all factors
            new_soil_moisture = base_soil_moisture + (water_amount * retention_factor) + rainfall_factor - evaporation
            
            # Apply physical limits
            next_state[soil_moisture_idx] = max(0.0, min(1.0, new_soil_moisture))
            
            # Update current soil moisture
            self.current_soil_moisture = next_state[soil_moisture_idx]
        
        # Apply other effects of water
        for feature, factor in self.water_effect_factors.items():
            if feature in self.feature_indices:
                idx = self.feature_indices[feature]
                if feature != 'soil_moisture':  # Already handled above
                    effect = water_amount * factor
                    next_state[idx] = max(0.0, min(1.0, next_state[idx] + effect))
        
        return next_state
    
    def _calculate_reward(self, current_state, next_state, water_amount):
        """
        Calculate reward based on multiple agricultural factors
        
        Parameters:
        -----------
        current_state : numpy.ndarray
            Current state
        next_state : numpy.ndarray
            Next state
        water_amount : float
            Amount of water allocated
            
        Returns:
        --------
        float: Calculated reward
        """
        reward = 0.0
        
        # Factor 1: Soil moisture optimization
        # Optimal soil moisture depends on crop type, temperature, etc.
        if 'soil_moisture' in self.feature_indices:
            soil_moisture_idx = self.feature_indices['soil_moisture']
            current_moisture = current_state[soil_moisture_idx]
            next_moisture = next_state[soil_moisture_idx]
            
            # Define optimal moisture range (should depend on crop type but simplifying here)
            optimal_low = 0.4
            optimal_high = 0.7
            
            # Reward for being in optimal range
            if optimal_low <= next_moisture <= optimal_high:
                reward += 2.0
            else:
                # Penalty for being outside optimal range, proportional to distance
                distance = min(abs(next_moisture - optimal_low), abs(next_moisture - optimal_high))
                reward -= distance * 3.0
            
            # Reward improvement toward optimal range
            if abs(next_moisture - 0.55) < abs(current_moisture - 0.55):
                reward += 1.0
        
        # Factor 2: Water efficiency
        # Reward for using less water if moisture is already adequate
        if 'soil_moisture' in self.feature_indices and 'water_usage_efficiency' in self.feature_indices:
            soil_moisture_idx = self.feature_indices['soil_moisture']
            efficiency_idx = self.feature_indices['water_usage_efficiency']
            
            current_moisture = current_state[soil_moisture_idx]
            efficiency = current_state[efficiency_idx]
            
            # If soil already has good moisture, reward water conservation
            if current_moisture > 0.6 and water_amount < 0.5:
                reward += 2.0 * efficiency * (1 - water_amount)
            
            # If soil is dry, reward appropriate watering
            elif current_moisture < 0.3 and water_amount > 0.5:
                reward += 2.0 * efficiency * water_amount
            
            # Penalize wasting water when soil is already wet
            elif current_moisture > 0.8 and water_amount > 0.5:
                reward -= 3.0 * water_amount
        
        # Factor 3: Nutrient balance
        # N, P, K balance is important for crop health
        if all(f in self.feature_indices for f in ['N', 'P', 'K']):
            n_idx = self.feature_indices['N']
            p_idx = self.feature_indices['P']
            k_idx = self.feature_indices['K']
            
            n_level = next_state[n_idx]
            p_level = next_state[p_idx]
            k_level = next_state[k_idx]
            
            # Calculate nutrient balance (simplified)
            balance = 1.0 - (abs(n_level - p_level) + abs(n_level - k_level) + abs(p_level - k_level)) / 3.0
            reward += balance * 2.0
            
            # Water impacts nutrient availability
            if 'soil_moisture' in self.feature_indices:
                moisture = next_state[self.feature_indices['soil_moisture']]
                if 0.3 <= moisture <= 0.8:  # Good moisture range for nutrient uptake
                    reward += 1.0
                else:
                    reward -= 1.0  # Poor moisture affects nutrient availability
        
        # Factor 4: Environmental conditions
        # Temperature, humidity, pH affect crop growth
        env_factor = 1.0
        
        if 'temperature' in self.feature_indices:
            temp_idx = self.feature_indices['temperature']
            temp = next_state[temp_idx]
            
            # Normalized temp assuming 0.5 is optimal (scale of 0-1)
            temp_effect = 1.0 - abs(temp - 0.5) * 2.0
            env_factor *= (0.5 + temp_effect / 2.0)  # Dampen the effect
        
        if 'humidity' in self.feature_indices:
            humid_idx = self.feature_indices['humidity']
            humidity = next_state[humid_idx]
            
            # Normalized humidity assuming 0.6 is optimal
            humidity_effect = 1.0 - abs(humidity - 0.6) * 2.0
            env_factor *= (0.5 + humidity_effect / 2.0)
        
        if 'ph' in self.feature_indices:
            ph_idx = self.feature_indices['ph']
            ph = next_state[ph_idx]
            
            # Normalized pH assuming 0.5 is optimal (neutral)
            ph_effect = 1.0 - abs(ph - 0.5) * 2.0
            env_factor *= (0.5 + ph_effect / 2.0)
        
        # Apply environmental factor
        reward *= max(0.5, env_factor)
        
        # Factor 5: Water conservation bonus
        # Reward for using less water overall
        water_conservation_bonus = (1.0 - water_amount) * 0.5
        reward += water_conservation_bonus
        
        return reward
    
    def render(self):
        """
        Optional visualization method
        
        Prints current state information
        """
        current_state = self.train_data[self.current_step].copy()
        self.logger.info("Current Environment State:")
        for feature, index in self.feature_indices.items():
            self.logger.info(f"{feature}: {current_state[index]}")
            
        # Print action space
        self.logger.info(f"Available water allocations: {self.action_space}")
        
    def seed(self, seed=None):
        """
        Set random seed for environment
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
            
        Returns:
        --------
        int: The seed used
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return seed