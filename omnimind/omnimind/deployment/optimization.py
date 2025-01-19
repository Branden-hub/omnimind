import torch
import torch.nn as nn
import torch.jit
import onnx
import tensorrt as trt
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from pathlib import Path
import json

class TorchScriptCompiler:
    """Compiles models to TorchScript for deployment."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats = {}
    
    def compile_model(self,
                     model: nn.Module,
                     example_inputs: Optional[torch.Tensor] = None,
                     optimize: bool = True) -> torch.jit.ScriptModule:
        """Compile model to TorchScript."""
        try:
            # Record original model size
            original_size = self._get_model_size(model)
            
            # Compile model
            if example_inputs is not None:
                # Use tracing
                traced_model = torch.jit.trace(model, example_inputs)
            else:
                # Use scripting
                traced_model = torch.jit.script(model)
            
            # Optimize if requested
            if optimize:
                traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Update statistics
            compiled_size = self._get_model_size(traced_model)
            self.stats.update({
                'original_size': original_size,
                'compiled_size': compiled_size,
                'compilation_method': 'trace' if example_inputs else 'script'
            })
            
            return traced_model
            
        except Exception as e:
            self.logger.error(f"Error compiling model: {e}")
            raise
    
    def save_model(self,
                  model: torch.jit.ScriptModule,
                  path: Union[str, Path]) -> None:
        """Save compiled model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save(str(path))
        
        # Save statistics
        stats_path = path.with_suffix('.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def load_model(self,
                  path: Union[str, Path]) -> torch.jit.ScriptModule:
        """Load compiled model."""
        path = Path(path)
        
        # Load model
        model = torch.jit.load(str(path))
        
        # Load statistics
        stats_path = path.with_suffix('.json')
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        
        return model
    
    def _get_model_size(self, model: Union[nn.Module, torch.jit.ScriptModule]) -> int:
        """Get model size in bytes."""
        return sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
        )

class ONNXExporter:
    """Exports models to ONNX format."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.stats = {}
    
    def export_model(self,
                    model: nn.Module,
                    example_inputs: torch.Tensor,
                    path: Union[str, Path],
                    input_names: Optional[List[str]] = None,
                    output_names: Optional[List[str]] = None,
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> None:
        """Export model to ONNX."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Export model
            torch.onnx.export(
                model,
                example_inputs,
                str(path),
                input_names=input_names or ['input'],
                output_names=output_names or ['output'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=12
            )
            
            # Verify exported model
            onnx_model = onnx.load(str(path))
            onnx.checker.check_model(onnx_model)
            
            # Update statistics
            self.stats.update({
                'model_size': path.stat().st_size,
                'opset_version': 12,
                'input_names': input_names or ['input'],
                'output_names': output_names or ['output']
            })
            
            # Save statistics
            stats_path = path.with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            raise
    
    def optimize_model(self,
                      path: Union[str, Path]) -> None:
        """Optimize ONNX model."""
        path = Path(path)
        
        try:
            # Load model
            model = onnx.load(str(path))
            
            # Optimize
            optimized_model = onnx.optimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, str(path))
            
            # Update statistics
            self.stats['optimized'] = True
            
        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            raise

class TensorRTOptimizer:
    """Optimizes models using TensorRT."""
    
    def __init__(self,
                 max_workspace_size: int = 1 << 30,
                 precision: str = 'fp32'):
        self.logger = logging.getLogger(__name__)
        self.max_workspace_size = max_workspace_size
        self.precision = precision
        self.stats = {}
        
        # Initialize TensorRT
        self.logger.info("Initializing TensorRT...")
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = max_workspace_size
        
        # Set precision
        if precision == 'fp16':
            self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            self.config.set_flag(trt.BuilderFlag.INT8)
    
    def optimize_model(self,
                      onnx_path: Union[str, Path],
                      output_path: Union[str, Path],
                      input_shape: Tuple[int, ...]) -> None:
        """Optimize ONNX model using TensorRT."""
        onnx_path = Path(onnx_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Parse ONNX
            network = self.builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, self.trt_logger)
            
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        self.logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
            
            # Set input shape
            network.get_input(0).shape = input_shape
            
            # Build engine
            engine = self.builder.build_engine(network, self.config)
            
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            
            # Update statistics
            self.stats.update({
                'precision': self.precision,
                'max_workspace_size': self.max_workspace_size,
                'input_shape': input_shape,
                'engine_size': output_path.stat().st_size
            })
            
            # Save statistics
            stats_path = output_path.with_suffix('.json')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error optimizing model: {e}")
            raise
    
    def load_engine(self,
                   path: Union[str, Path]) -> trt.ICudaEngine:
        """Load TensorRT engine."""
        path = Path(path)
        
        try:
            # Load engine
            runtime = trt.Runtime(self.trt_logger)
            with open(path, 'rb') as f:
                engine_data = f.read()
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            return engine
            
        except Exception as e:
            self.logger.error(f"Error loading engine: {e}")
            raise
