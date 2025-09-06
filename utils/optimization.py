"""
Utility functions for optimization and performance
"""

import torch
import torch.nn as nn


def setup_optimization(config, device='cuda'):
    """
    Setup optimization settings based on configuration
    
    Args:
        config: Configuration object
        device: Device to run on
        
    Returns:
        Dictionary with optimization settings applied
    """
    optimization_info = {}
    
    # Enable cuDNN benchmarking for consistent input sizes
    if config.get('optimization.cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Slightly faster
        optimization_info['cudnn_benchmark'] = True
    
    # Enable TF32 for Ampere GPUs
    if config.get('optimization.allow_tf32', True):
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        optimization_info['tf32_enabled'] = True
    
    # Check CUDA capability
    cuda_capability = None
    if torch.cuda.is_available():
        cuda_capability = torch.cuda.get_device_capability()
        optimization_info['cuda_capability'] = cuda_capability
        print(f"🔍 CUDA Capability: {cuda_capability}")
    
    # Model compilation settings
    compile_model = config.get('optimization.compile_model', True)
    min_cuda_capability = config.get('optimization.min_cuda_capability', 7.0)
    
    optimization_info['should_compile'] = False
    if compile_model and cuda_capability and cuda_capability[0] >= min_cuda_capability:
        optimization_info['should_compile'] = True
        optimization_info['compile_mode'] = config.get('optimization.compile_mode', 'max-autotune')
        print(f"✅ Model compilation enabled with mode: {optimization_info['compile_mode']}")
    else:
        if cuda_capability:
            print(f"⚠️ Skipping model compilation - CUDA capability {cuda_capability} < {min_cuda_capability}")
        print("🔧 Running in standard mode")
    
    # Memory format optimization
    if config.get('optimization.channels_last', True):
        optimization_info['use_channels_last'] = True
        print("✅ Channels-last memory format enabled")
    
    return optimization_info


def optimize_model(model, config, device='cuda'):
    """
    Apply optimizations to the model
    
    Args:
        model: PyTorch model
        config: Configuration object
        device: Device to run on
        
    Returns:
        Optimized model
    """
    optimization_info = setup_optimization(config, device)
    
    # Model compilation for PyTorch 2.0+
    if optimization_info.get('should_compile', False):
        try:
            compile_mode = optimization_info.get('compile_mode', 'max-autotune')
            model = torch.compile(model, mode=compile_mode)
            print(f"✅ Model compiled with {compile_mode} mode for ultra speed!")
        except Exception as e:
            print(f"⚠️ Model compilation failed: {e}")
            print("🔧 Using standard mode")
    
    # Convert model to optimal memory format
    if optimization_info.get('use_channels_last', False):
        model = model.to(memory_format=torch.channels_last)
        print("✅ Model converted to channels-last memory format")
    
    return model


def count_parameters(model):
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def print_model_info(model):
    """
    Print detailed model information
    
    Args:
        model: PyTorch model
    """
    param_info = count_parameters(model)
    
    print(f"\n{'='*60}")
    print("MODEL INFORMATION")
    print(f"{'='*60}")
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Non-trainable parameters: {param_info['non_trainable_parameters']:,}")
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = (param_size + buffer_size) / 1024 / 1024
    print(f"Model size: {model_size:.2f} MB")
    print(f"{'='*60}\n")


def setup_device(config):
    """
    Setup and configure device
    
    Args:
        config: Configuration object
        
    Returns:
        torch.device object
    """
    use_cuda = config.get('device.use_cuda', True)
    device_id = config.get('device.device_id', 0)
    
    if use_cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"🖥️ Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(device_id)}")
        print(f"Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"🖥️ Using device: {device}")
        if use_cuda:
            print("⚠️ CUDA requested but not available, falling back to CPU")
    
    return device
