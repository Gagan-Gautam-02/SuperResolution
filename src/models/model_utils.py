import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """Get model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    @staticmethod
    def initialize_weights(model: nn.Module, init_type: str = 'xavier'):
        """Initialize model weights"""
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                if init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(m.weight, 0, 0.02)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @staticmethod
    def save_checkpoint(model: nn.Module, optimizer, epoch: int, 
                       loss: float, filepath: str, **kwargs):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            **kwargs
        }
        torch.save(checkpoint, filepath)
    
    @staticmethod
    def load_checkpoint(filepath: str, model: nn.Module, 
                       optimizer=None, device='cpu') -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint
    
    @staticmethod
    def model_summary(model: nn.Module, input_size: tuple):
        """Print model summary"""
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = f"{class_name}-{module_idx+1}"
                summary[m_key] = {}
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = -1
                
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = -1
                
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
            
            if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
                hooks.append(module.register_forward_hook(hook))
        
        device = next(model.parameters()).device
        summary = {}
        hooks = []
        
        model.apply(register_hook)
        
        # Create dummy input
        if len(input_size) == 3:
            x = torch.rand(2, *input_size).to(device)
            model(x, x)  # For dual input models
        else:
            x = torch.rand(1, *input_size).to(device)
            model(x)
        
        for h in hooks:
            h.remove()
        
        print("=" * 70)
        print(f"{'Layer (type)':<25} {'Output Shape':<25} {'Param #':<15}")
        print("=" * 70)
        
        total_params = 0
        total_output = 0
        trainable_params = 0
        
        for layer in summary:
            line_new = f"{layer:<25} {str(summary[layer]['output_shape']):<25} {summary[layer]['nb_params']:<15}"
            total_params += summary[layer]["nb_params"]
            
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"]:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)
        
        print("=" * 70)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 70)
