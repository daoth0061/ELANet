"""
Configuration management for ELANet
Handles loading and validation of configuration parameters
"""

import yaml
import os
from typing import Dict, Any


class Config:
    """Configuration class for ELANet"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        return config
    
    def _validate_config(self):
        """Validate configuration parameters"""
        required_sections = ['dataset', 'model', 'training', 'checkpoints']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate dataset configuration
        if not os.path.exists(self.config['dataset']['base_url']):
            print(f"Warning: Dataset path does not exist: {self.config['dataset']['base_url']}")
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = self.config['checkpoints']['save_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def get(self, key_path: str, default=None):
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to configuration value (e.g., 'model.haft.num_levels')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getattr__(self, name):
        """Allow direct access to top-level configuration sections"""
        try:
            config = object.__getattribute__(self, 'config')
            if name in config:
                return config[name]
        except AttributeError:
            raise AttributeError(f"Configuration section '{name}' not found")
        # raise AttributeError(f"Configuration section '{name}' not found")
    
    def save(self, save_path: str = None):
        """Save current configuration to file"""
        if save_path is None:
            save_path = self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)
    
    def update(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def print_config(self):
        """Print current configuration"""
        print("Current Configuration:")
        print(yaml.dump(self.config, default_flow_style=False, indent=2))


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    return Config(config_path)
