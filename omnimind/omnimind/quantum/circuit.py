import numpy as np
from typing import List, Dict, Optional, Union
from .gates import QuantumGates
from .registers import QuantumRegisters
from .measurement import QuantumMeasurement

class QuantumCircuit:
    """Simulates a quantum circuit for quantum computations."""
    
    def __init__(self):
        self.gates = QuantumGates()
        self.registers = QuantumRegisters()
        self.measurement = QuantumMeasurement()
        self.num_qubits = 0
        self.state_vector = None
        self.circuit_operations = []
        
    def initialize(self, num_qubits: int) -> None:
        """Initialize the quantum circuit with specified number of qubits."""
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=np.complex128)
        self.state_vector[0] = 1.0  # Initialize to |0...0⟩
        
    def apply_gate(self, gate_name: str, target_qubit: int,
                  control_qubit: Optional[int] = None) -> None:
        """Apply a quantum gate to the specified qubit(s)."""
        gate_matrix = self.gates.get_gate(gate_name)
        
        if control_qubit is not None:
            # Controlled operation
            self._apply_controlled_gate(gate_matrix, control_qubit, target_qubit)
        else:
            # Single qubit operation
            self._apply_single_qubit_gate(gate_matrix, target_qubit)
            
        self.circuit_operations.append({
            'gate': gate_name,
            'target': target_qubit,
            'control': control_qubit
        })
        
    def measure(self, qubit: int) -> int:
        """Measure the specified qubit and collapse its state."""
        return self.measurement.measure_qubit(self.state_vector, qubit)
    
    def measure_all(self) -> List[int]:
        """Measure all qubits in the circuit."""
        return self.measurement.measure_all(self.state_vector)
    
    def get_state(self) -> np.ndarray:
        """Get the current state vector of the circuit."""
        return self.state_vector.copy()
    
    def _apply_single_qubit_gate(self, gate_matrix: np.ndarray,
                                target_qubit: int) -> None:
        """Apply a single qubit gate."""
        n = self.num_qubits
        target = target_qubit
        
        # Create the full operation matrix
        operation = np.eye(1)
        
        for i in range(n):
            if i == target:
                operation = np.kron(operation, gate_matrix)
            else:
                operation = np.kron(operation, np.eye(2))
                
        self.state_vector = operation @ self.state_vector
        
    def _apply_controlled_gate(self, gate_matrix: np.ndarray,
                             control_qubit: int, target_qubit: int) -> None:
        """Apply a controlled quantum gate."""
        n = self.num_qubits
        control = control_qubit
        target = target_qubit
        
        # Create the projection operators
        p0 = np.array([[1, 0], [0, 0]])
        p1 = np.array([[0, 0], [0, 1]])
        
        # Create the full operation matrix
        operation = np.zeros((2**n, 2**n), dtype=np.complex128)
        
        # Add the identity operation when control qubit is |0⟩
        term0 = np.eye(1)
        for i in range(n):
            if i == control:
                term0 = np.kron(term0, p0)
            elif i == target:
                term0 = np.kron(term0, np.eye(2))
            else:
                term0 = np.kron(term0, np.eye(2))
                
        # Add the gate operation when control qubit is |1⟩
        term1 = np.eye(1)
        for i in range(n):
            if i == control:
                term1 = np.kron(term1, p1)
            elif i == target:
                term1 = np.kron(term1, gate_matrix)
            else:
                term1 = np.kron(term1, np.eye(2))
                
        operation = term0 + term1
        self.state_vector = operation @ self.state_vector
