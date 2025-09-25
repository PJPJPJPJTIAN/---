import os
import numpy as np
import ast
from src.problems.base.env import BaseEnv
from src.problems.psp.components import Solution


class Env(BaseEnv):
    """Port Scheduling env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "psp")
        self.construction_steps = self.instance_data["vessel_num"]
        self.key_item = "total_scheduling_cost"
        self.compare = lambda x, y: y - x  # Lower cost is better

    @property
    def is_complete_solution(self) -> bool:
        # A solution is complete when all vessels have been considered (assigned or unassigned)
        assigned_vessels = set(self.current_solution.vessel_assignments.keys())
        total_vessels = set(range(self.instance_data["vessel_num"]))
        return assigned_vessels == total_vessels

    def load_data(self, data_path: str) -> dict:
        """Load port scheduling problem instance data from text file."""
        instance_data = {}
        
        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse parameter = value format
                if '=' in line:
                    param_name, param_value = line.split('=', 1)
                    param_name = param_name.strip()
                    param_value = param_value.strip()
                    
                    # Try to evaluate the value
                    try:
                        # Handle list format
                        if param_value.startswith('[') and param_value.endswith(']'):
                            value = ast.literal_eval(param_value)
                            instance_data[param_name] = np.array(value)
                        else:
                            # Handle single values
                            try:
                                # Try to parse as float first
                                value = float(param_value)
                                # Convert to int if it's a whole number
                                if value == int(value):
                                    value = int(value)
                                instance_data[param_name] = value
                            except ValueError:
                                # Keep as string if can't convert to number
                                instance_data[param_name] = param_value
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse value for {param_name}: {param_value}")
                        instance_data[param_name] = param_value
        
        return instance_data

    def init_solution(self) -> Solution:
        """Initialize an empty solution."""
        vessel_assignments = {i: None for i in range(self.instance_data["vessel_num"])}
        tugboat_inbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        tugboat_outbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        
        return Solution(
            vessel_assignments=vessel_assignments,
            tugboat_inbound_assignments=tugboat_inbound_assignments,
            tugboat_outbound_assignments=tugboat_outbound_assignments
        )

    def get_key_value(self, solution: Solution = None) -> float:
        """Calculate the total scheduling cost of the solution based on the model."""
        if solution is None:
            solution = self.current_solution
        
        total_cost = 0.0
        
        # Z1: Unserved vessel penalty
        # Z1 = Σ_i M * α_i * (1 - Σ_j Σ_t x_ijt)
        unserved_penalty = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            if solution.vessel_assignments.get(vessel_id) is None:
                unserved_penalty += (self.instance_data["penalty_parameter"] * 
                                   self.instance_data["vessel_priority_weights"][vessel_id])
        
        # Z2: Total port time cost 
        # Z2 = Σ_i α_i * β_i * [Σ_k Σ_t (t + τ^out_i) * y^out_ikt - Σ_k Σ_t t * y^in_ikt]
        port_time_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                
                if inbound_services and outbound_services:
                    # Get inbound service start time
                    inbound_start = inbound_services[0][1]  # Single tugboat service
                    
                    # Get outbound service start time and add service duration
                    outbound_start = outbound_services[0][1]  # Single tugboat service
                    outbound_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                    outbound_end = outbound_start + outbound_duration
                    
                    # Calculate total port time: from inbound start to outbound completion
                    total_port_time = outbound_end - inbound_start
                    
                    port_time_cost += (self.instance_data["vessel_priority_weights"][vessel_id] * 
                                     self.instance_data["vessel_waiting_costs"][vessel_id] * 
                                     total_port_time)
        
        # Z3: ETA deviation cost
        # Z3 = Σ_i α_i * γ_i * (u^early_i + u^late_i)
        # Based on constraint: Σ_k Σ_t t * y^in_ikt = ETA_i + u^late_i - u^early_i
        eta_deviation_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            if inbound_services:
                inbound_start = inbound_services[0][1]  # Single tugboat service
                eta = self.instance_data["vessel_etas"][vessel_id]
                
                # Calculate ETA deviations
                if inbound_start < eta:
                    early_deviation = eta - inbound_start
                    late_deviation = 0
                else:
                    early_deviation = 0
                    late_deviation = inbound_start - eta
                
                eta_deviation_cost += (self.instance_data["vessel_priority_weights"][vessel_id] * 
                                     self.instance_data["vessel_jit_costs"][vessel_id] * 
                                     (early_deviation + late_deviation))
        
        # Z4: Tugboat utilization cost
        # Z4 = Σ_k Σ_i Σ_t c_k * (τ^in_i * y^in_ikt + τ^out_i * y^out_ikt)
        tugboat_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Inbound tugboat services
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                tugboat_unit_cost = self.instance_data["tugboat_costs"][tugboat_id]
                tugboat_cost += tugboat_unit_cost * service_duration
            
            # Outbound tugboat services
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                tugboat_unit_cost = self.instance_data["tugboat_costs"][tugboat_id]
                tugboat_cost += tugboat_unit_cost * service_duration
        
        # Total weighted cost
        # Z = λ_1 * Z_1 + λ_2 * Z_2 + λ_3 * Z_3 + λ_4 * Z_4
        total_cost = (self.instance_data["objective_weights"][0] * unserved_penalty +
                     self.instance_data["objective_weights"][1] * port_time_cost +
                     self.instance_data["objective_weights"][2] * eta_deviation_cost +
                     self.instance_data["objective_weights"][3] * tugboat_cost)
        
        return total_cost

    def validation_solution(self, solution: Solution = None) -> bool:
        """
        Check the validation of the solution following the mathematical model constraint order:
        Constraint (1): Each vessel can be assigned at most once
        Constraint (2): Inbound tugboat service coupling
        Constraint (3): Outbound tugboat service coupling  
        Constraint (4): Vessel-berth compatibility constraints
        Constraint (5): Vessel-tugboat inbound compatibility
        Constraint (6): Vessel-tugboat outbound compatibility
        Constraint (7): Single inbound service per vessel
        Constraint (8): Single outbound service per vessel
        Constraint (9): Inbound-berthing timing sequence constraints
        Constraint (10): Berthing-outbound timing sequence constraints
        Constraint (11): Berth occupation constraints
        Constraint (12): Tugboat occupation constraints
        Constraint (13): Inbound service time window constraints
        Constraint (14): ETA deviation linearization constraints
        Constraint (15): Variable domain constraints
        """
        if solution is None:
            solution = self.current_solution
    
        # Basic solution format validation
        if not isinstance(solution, Solution):
            return False
    
        # Constraint (1): Each vessel can be assigned at most once
        # ∑_j ∑_t x_ijt ≤ 1, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment_count = 1 if solution.vessel_assignments.get(vessel_id) is not None else 0
            if assignment_count > 1:
                return False
    
        # Constraint (2): Inbound tugboat service coupling
        # ∑_k ∑_t y^in_ikt = ∑_j ∑_t x_ijt, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            
            if assignment is not None:
                # Assigned vessels must have exactly one inbound service
                if len(inbound_services) != 1:
                    return False
            else:
                # Unassigned vessels should not have inbound service
                if inbound_services:
                    return False
    
        # Constraint (3): Outbound tugboat service coupling
        # ∑_k ∑_t y^out_ikt = ∑_j ∑_t x_ijt, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            
            if assignment is not None:
                # Assigned vessels must have exactly one outbound service
                if len(outbound_services) != 1:
                    return False
            else:
                # Unassigned vessels should not have outbound service
                if outbound_services:
                    return False
    
        # Constraint (4): Vessel-berth compatibility constraints
        # x_ijt = 0, ∀i ∈ I, j ∈ J, t ∈ T : C_j < S_i
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                
                # Check if berth exists
                if not (0 <= berth_id < self.instance_data["berth_num"]):
                    return False
                
                # Check vessel-berth level compatibility
                vessel_level = self.instance_data["vessel_sizes"][vessel_id]
                berth_capacity = self.instance_data["berth_capacities"][berth_id]
                if berth_capacity < vessel_level:
                    return False
    
        # Constraint (5): Vessel-tugboat inbound compatibility
        # y^in_ikt = 0, ∀i ∈ I, k ∈ K, t ∈ T : G_k < S_i
        for vessel_id in range(self.instance_data["vessel_num"]):
            vessel_level = self.instance_data["vessel_sizes"][vessel_id]
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            
            for tugboat_id, _ in inbound_services:
                if not (0 <= tugboat_id < self.instance_data["tugboat_num"]):
                    return False
                tugboat_capacity = self.instance_data["tugboat_capacities"][tugboat_id]
                if tugboat_capacity < vessel_level:
                    return False
    
        # Constraint (6): Vessel-tugboat outbound compatibility
        # y^out_ikt = 0, ∀i ∈ I, k ∈ K, t ∈ T : G_k < S_i
        for vessel_id in range(self.instance_data["vessel_num"]):
            vessel_level = self.instance_data["vessel_sizes"][vessel_id]
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            
            for tugboat_id, _ in outbound_services:
                if not (0 <= tugboat_id < self.instance_data["tugboat_num"]):
                    return False
                tugboat_capacity = self.instance_data["tugboat_capacities"][tugboat_id]
                if tugboat_capacity < vessel_level:
                    return False
    
        # Constraint (7): Single inbound service per vessel
        # ∑_k ∑_t y^in_ikt ≤ 1, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            if len(inbound_services) > 1:
                return False
    
        # Constraint (8): Single outbound service per vessel
        # ∑_k ∑_t y^out_ikt ≤ 1, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            if len(outbound_services) > 1:
                return False
    
        # Constraint (9): Inbound-berthing timing sequence constraints
        # 0 ≤ ∑_j ∑_t t*x_ijt - ∑_k ∑_t (t + τ^in_i)*y^in_ikt ≤ ε_time * ∑_j ∑_t x_ijt, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                _, berth_start = assignment
                
                # Check inbound service sequence
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                for tugboat_id, service_start in inbound_services:
                    service_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                    service_end = service_start + service_duration
                    
                    # Inbound service must complete before or at berth start
                    if service_end > berth_start:
                        return False
                    
                    # Check time tolerance
                    time_gap = berth_start - service_end
                    if time_gap > self.instance_data["time_constraint_tolerance"]:
                        return False
    
        # Constraint (10): Berthing-outbound timing sequence constraints
        # 0 ≤ ∑_k ∑_t t*y^out_ikt - ∑_j ∑_t (t + D_i)*x_ijt ≤ ε_time * ∑_j ∑_t x_ijt, ∀i ∈ I
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                _, berth_start = assignment
                berth_duration = self.instance_data["vessel_durations"][vessel_id]
                berth_end = berth_start + berth_duration
                
                # Check outbound service sequence
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                for tugboat_id, service_start in outbound_services:
                    # Outbound service must start after berth ends
                    if service_start < berth_end:
                        return False
                    
                    # Check time tolerance
                    time_gap = service_start - berth_end
                    if time_gap > self.instance_data["time_constraint_tolerance"]:
                        return False
    
        # Constraint (11): Berth occupation constraints
        # ∑_i ∑_{τ=max(1,t-D_i+1)}^t x_ijτ ≤ 1, ∀j ∈ J, t ∈ T
        berth_schedules = {}
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                duration = self.instance_data["vessel_durations"][vessel_id]
                
                if berth_id not in berth_schedules:
                    berth_schedules[berth_id] = []
                
                # Check for time overlapping with existing assignments
                for existing_start, existing_end in berth_schedules[berth_id]:
                    if not (start_time + duration <= existing_start or start_time >= existing_end):
                        return False
                
                berth_schedules[berth_id].append((start_time, start_time + duration))
    
        # Constraint (12): Tugboat occupation constraints
        # ∑_i [∑_{τ=max(1,t-τ^in_i-ρ^in+1)}^t y^in_ikτ + ∑_{τ=max(1,t-τ^out_i-ρ^out+1)}^t y^out_ikτ] ≤ 1, ∀k ∈ K, t ∈ T
        tugboat_schedules = {k: [] for k in range(self.instance_data["tugboat_num"])}
        
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Process inbound occupation
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                prep_time = self.instance_data["inbound_preparation_time"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts with existing tugboat schedule
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))
            
            # Process outbound occupation
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                prep_time = self.instance_data["outbound_preparation_time"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts with existing tugboat schedule
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))
    
        # Constraint (13): Inbound service time window constraints
        # y^in_ikt = 0, ∀i ∈ I, k ∈ K, t < ETA_i - Δ^early_i or t > ETA_i + Δ^late_i
        for vessel_id in range(self.instance_data["vessel_num"]):
            eta = self.instance_data["vessel_etas"][vessel_id]
            early_limit = self.instance_data["vessel_early_limits"][vessel_id]
            late_limit = self.instance_data["vessel_late_limits"][vessel_id]
            
            # Check inbound service time windows
            for tugboat_id, service_start in solution.tugboat_inbound_assignments.get(vessel_id, []):
                if not (eta - early_limit <= service_start <= eta + late_limit):
                    return False
    
        # Constraint (14): ETA deviation linearization constraints
        # ∑_k ∑_t t*y^in_ikt = ETA_i + u^late_i - u^early_i, ∀i ∈ I
        # This constraint is used for objective function calculation and is implicitly handled
        # The actual inbound start time is validated through time window constraints above
    
        # Constraint (15): Variable domain constraints
        # All variables are properly constrained by their definitions and bounds checks above
        # Additional domain checks
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                # Check time period bounds
                if not (1 <= start_time <= self.instance_data["time_periods"]):
                    return False
        
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Check inbound service time bounds
            for tugboat_id, service_start in solution.tugboat_inbound_assignments.get(vessel_id, []):
                if not (1 <= service_start <= self.instance_data["time_periods"]):
                    return False
            
            # Check outbound service time bounds
            for tugboat_id, service_start in solution.tugboat_outbound_assignments.get(vessel_id, []):
                if not (1 <= service_start <= self.instance_data["time_periods"]):
                    return False
    
        return True

    def get_unassigned_vessels(self, solution: Solution = None) -> list:
        """Get list of unassigned vessel IDs."""
        if solution is None:
            solution = self.current_solution
            
        unassigned = []
        for vessel_id in range(self.instance_data['vessel_num']):
            if solution.vessel_assignments.get(vessel_id) is None:
                unassigned.append(vessel_id)
        
        return unassigned
    
    def get_vessel_time_window(self, vessel_id: int) -> tuple:
        """Get the feasible time window for vessel's inbound service start."""
        eta = self.instance_data['vessel_etas'][vessel_id]
        early_limit = self.instance_data['vessel_early_limits'][vessel_id]
        late_limit = self.instance_data['vessel_late_limits'][vessel_id]
        
        earliest_start = max(1, int(eta - early_limit))
        latest_start = min(self.instance_data['time_periods'], int(eta + late_limit))
        
        return earliest_start, latest_start   
    
    def get_compatible_berths(self, vessel_id: int) -> list:
        """Get list of berths compatible with vessel level requirements."""
        vessel_level = self.instance_data['vessel_sizes'][vessel_id]
        compatible_berths = []
        
        for berth_id in range(self.instance_data['berth_num']):
            if self.instance_data['berth_capacities'][berth_id] >= vessel_level:
                compatible_berths.append(berth_id)
        
        return compatible_berths
    
    def get_compatible_tugboats(self, vessel_id: int) -> list:
        """Get list of tugboats compatible with vessel level requirements."""
        vessel_level = self.instance_data['vessel_sizes'][vessel_id]
        compatible_tugboats = []
        
        for tugboat_id in range(self.instance_data['tugboat_num']):
            if self.instance_data['tugboat_capacities'][tugboat_id] >= vessel_level:
                compatible_tugboats.append(tugboat_id)
        
        return compatible_tugboats
    
    def is_berth_available(self, berth_id: int, start_time: int, duration: int, solution: Solution = None) -> bool:
        """Check if berth is available for the given time period."""
        if solution is None:
            solution = self.current_solution
            
        end_time = start_time + duration
        
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                assigned_berth, assigned_start = assignment
                if assigned_berth == berth_id:
                    assigned_duration = self.instance_data['vessel_durations'][vessel_id]
                    assigned_end = assigned_start + assigned_duration
                    if not (end_time <= assigned_start or start_time >= assigned_end):
                        return False
        return True
    
    def is_tugboat_available(self, tugboat_id: int, start_time: int, duration: int, 
                           prep_time: int = 0, solution: Solution = None) -> bool:
        """Check if tugboat is available for the given service period including prep time."""
        if solution is None:
            solution = self.current_solution
            
        service_end = start_time + duration + prep_time
        
        for vessel_id in range(self.instance_data['vessel_num']):
            # Check inbound assignments
            for assigned_tug, assigned_start in solution.tugboat_inbound_assignments.get(vessel_id, []):
                if assigned_tug == tugboat_id:
                    assigned_duration = self.instance_data['vessel_inbound_service_times'][vessel_id]
                    assigned_end = assigned_start + assigned_duration + self.instance_data['inbound_preparation_time']
                    if not (service_end <= assigned_start or start_time >= assigned_end):
                        return False
            
            # Check outbound assignments
            for assigned_tug, assigned_start in solution.tugboat_outbound_assignments.get(vessel_id, []):
                if assigned_tug == tugboat_id:
                    assigned_duration = self.instance_data['vessel_outbound_service_times'][vessel_id]
                    assigned_end = assigned_start + assigned_duration + self.instance_data['outbound_preparation_time']
                    if not (service_end <= assigned_start or start_time >= assigned_end):
                        return False
        return True
    
    def find_feasible_assignments(self, vessel_id: int, max_results: int = 3, 
                                 solution: Solution = None) -> list:
        """
        Find feasible complete assignments strictly following the mathematical model constraints.
        
        Mathematical Model Validation:
        - Constraint (4): Vessel-berth compatibility (C_j >= S_i)
        - Constraint (5,6): Vessel-tugboat compatibility (G_k >= S_i) 
        - Constraint (9): Inbound-berthing timing sequence
        - Constraint (10): Berthing-outbound timing sequence
        - Constraint (11): Berth occupation constraints
        - Constraint (12): Tugboat occupation constraints  
        - Constraint (13): Inbound service time window
        - Constraint (14): ETA deviation linearization
        """
        if solution is None:
            solution = self.current_solution
        
        assignments = []
        
        # Get vessel parameters
        vessel_level = self.instance_data['vessel_sizes'][vessel_id]
        eta_i = self.instance_data['vessel_etas'][vessel_id] 
        duration_i = int(self.instance_data['vessel_durations'][vessel_id])
        tau_in_i = int(self.instance_data['vessel_inbound_service_times'][vessel_id])
        tau_out_i = int(self.instance_data['vessel_outbound_service_times'][vessel_id])
        delta_early_i = self.instance_data['vessel_early_limits'][vessel_id]
        delta_late_i = self.instance_data['vessel_late_limits'][vessel_id]
        epsilon_time = self.instance_data['time_constraint_tolerance']
        prep_in = self.instance_data['inbound_preparation_time']
        prep_out = self.instance_data['outbound_preparation_time']
        T = self.instance_data['time_periods']
        
        # Constraint (13): Inbound service time window
        # y^in_ikt = 0, ∀i ∈ I, k ∈ K, t < ETA_i - Δ^early_i or t > ETA_i + Δ^late_i
        earliest_inbound = max(1, int(eta_i - delta_early_i))
        latest_inbound = min(T, int(eta_i + delta_late_i))
        
        if earliest_inbound > latest_inbound:
            return []  # No valid time window
        
        # Get compatible resources - Constraints (4), (5), (6)
        compatible_berths = []
        for berth_id in range(self.instance_data['berth_num']):
            if self.instance_data['berth_capacities'][berth_id] >= vessel_level:  # C_j >= S_i
                compatible_berths.append(berth_id)
        
        compatible_tugboats = []
        for tug_id in range(self.instance_data['tugboat_num']):
            if self.instance_data['tugboat_capacities'][tug_id] >= vessel_level:  # G_k >= S_i
                compatible_tugboats.append(tug_id)
        
        if not compatible_berths or not compatible_tugboats:
            return []
        
        # Enumerate all feasible timing combinations
        for t_in in range(earliest_inbound, latest_inbound + 1):  # Inbound start time
            # Constraint (9): Inbound-berthing timing sequence
            # 0 ≤ ∑_j ∑_t t*x_ijt - ∑_k ∑_t (t + τ^in_i)*y^in_ikt ≤ ε_time * ∑_j ∑_t x_ijt
            # Simplified: 0 ≤ t_berth - (t_in + τ^in_i) ≤ ε_time
            
            inbound_end = t_in + tau_in_i
            min_berth_start = inbound_end  # Gap ≥ 0
            max_berth_start = min(T, inbound_end + int(epsilon_time))  # Gap ≤ ε_time
            
            for t_berth in range(min_berth_start, max_berth_start + 1):  # Berth start time
                berth_end = t_berth + duration_i
                
                if berth_end > T:  # Check time bounds
                    continue
                    
                # Constraint (10): Berthing-outbound timing sequence  
                # 0 ≤ ∑_k ∑_t t*y^out_ikt - ∑_j ∑_t (t + D_i)*x_ijt ≤ ε_time * ∑_j ∑_t x_ijt
                # Simplified: 0 ≤ t_out - (t_berth + D_i) ≤ ε_time
                
                min_outbound_start = berth_end  # Gap ≥ 0
                max_outbound_start = min(T, berth_end + int(epsilon_time))  # Gap ≤ ε_time
                
                for t_out in range(min_outbound_start, max_outbound_start + 1):  # Outbound start time
                    outbound_end = t_out + tau_out_i
                    
                    if outbound_end > T:  # Check time bounds
                        continue
                    
                    # Find feasible berth assignment
                    feasible_berth = None
                    for berth_id in compatible_berths:
                        # Constraint (11): Berth occupation constraints
                        if self.is_berth_available(berth_id, t_berth, duration_i, solution):
                            feasible_berth = berth_id
                            break
                    
                    if feasible_berth is None:
                        continue
                    
                    # Find feasible tugboat for inbound
                    feasible_inbound_tug = None
                    min_inbound_cost = float('inf')
                    for tug_id in compatible_tugboats:
                        # Constraint (12): Tugboat occupation constraints
                        if self.is_tugboat_available(tug_id, t_in, tau_in_i, prep_in, solution):
                            tug_cost = self.instance_data['tugboat_costs'][tug_id]
                            if tug_cost < min_inbound_cost:
                                min_inbound_cost = tug_cost
                                feasible_inbound_tug = (tug_id, t_in)
                    
                    if feasible_inbound_tug is None:
                        continue
                    
                    # Find feasible tugboat for outbound  
                    feasible_outbound_tug = None
                    min_outbound_cost = float('inf')
                    for tug_id in compatible_tugboats:
                        # Constraint (12): Tugboat occupation constraints
                        if self.is_tugboat_available(tug_id, t_out, tau_out_i, prep_out, solution):
                            tug_cost = self.instance_data['tugboat_costs'][tug_id]
                            if tug_cost < min_outbound_cost:
                                min_outbound_cost = tug_cost
                                feasible_outbound_tug = (tug_id, t_out)
                    
                    if feasible_outbound_tug is None:
                        continue
                    
                    # Calculate total assignment cost
                    try:
                        total_cost = self.calculate_assignment_cost(
                            vessel_id, feasible_berth, t_berth, 
                            feasible_inbound_tug, feasible_outbound_tug)
                    except Exception:
                        continue
                    
                    # Create mathematically valid assignment
                    assignment = {
                        'berth_id': feasible_berth,
                        'berth_start_time': t_berth,
                        'inbound_tugboats': [feasible_inbound_tug],   # Single service per constraints (7)
                        'outbound_tugboats': [feasible_outbound_tug], # Single service per constraints (8)
                        'total_cost': total_cost
                    }
                    
                    assignments.append(assignment)
                    
                    # Early termination for efficiency
                    if len(assignments) >= max_results:
                        return sorted(assignments, key=lambda x: x['total_cost'])
        
        return sorted(assignments, key=lambda x: x['total_cost'])

    def calculate_assignment_cost(self, vessel_id: int, berth_id: int, berth_start_time: int,
                                inbound_tugboat: tuple, outbound_tugboat: tuple) -> float:
        """Calculate the cost components for a specific vessel assignment.
        
        Args:
            vessel_id: ID of the vessel
            berth_id: ID of the berth
            berth_start_time: Start time of berthing
            inbound_tugboat: (tugboat_id, start_time) tuple
            outbound_tugboat: (tugboat_id, start_time) tuple
        """
        if inbound_tugboat is None or outbound_tugboat is None:
            return float('inf')  # Invalid assignment
        
        _, inbound_start = inbound_tugboat
        _, outbound_start = outbound_tugboat
        
        # Calculate port time cost (from inbound start to outbound completion)
        outbound_duration = self.instance_data['vessel_outbound_service_times'][vessel_id]
        total_port_time = (outbound_start + outbound_duration) - inbound_start
        port_time_cost = (self.instance_data['vessel_priority_weights'][vessel_id] * 
                         self.instance_data['vessel_waiting_costs'][vessel_id] * total_port_time)
        
        # Calculate ETA deviation cost
        eta = self.instance_data['vessel_etas'][vessel_id]
        eta_deviation = abs(inbound_start - eta)
        eta_cost = (self.instance_data['vessel_priority_weights'][vessel_id] * 
                   self.instance_data['vessel_jit_costs'][vessel_id] * eta_deviation)
        
        # Calculate tugboat utilization cost
        inbound_tugboat_id, _ = inbound_tugboat
        outbound_tugboat_id, _ = outbound_tugboat
        
        inbound_cost = (self.instance_data['tugboat_costs'][inbound_tugboat_id] * 
                       self.instance_data['vessel_inbound_service_times'][vessel_id])
        outbound_cost = (self.instance_data['tugboat_costs'][outbound_tugboat_id] * 
                        self.instance_data['vessel_outbound_service_times'][vessel_id])
        tugboat_cost = inbound_cost + outbound_cost
        
        return port_time_cost + eta_cost + tugboat_cost
    
    def helper_function(self) -> dict:
        """Return essential helper functions only."""
        return {
            # Core validation and state
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            
            # Basic queries
            "get_unassigned_vessels": self.get_unassigned_vessels,
            "get_vessel_time_window": self.get_vessel_time_window,
            "get_compatible_berths": self.get_compatible_berths,
            "get_compatible_tugboats": self.get_compatible_tugboats,
            
            # Availability checks
            "is_berth_available": self.is_berth_available,
            "is_tugboat_available": self.is_tugboat_available,
            
            # Assignment functions
            "find_feasible_assignments": self.find_feasible_assignments,  
            "calculate_assignment_cost": self.calculate_assignment_cost,
        }