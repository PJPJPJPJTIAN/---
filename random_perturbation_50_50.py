from src.problems.psp.components import *
import random
import numpy as np

def random_perturbation_50_50(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[BaseOperator, dict]:
    """
    Random perturbation heuristic with 50/50 probability strategy:
    - 50% chance: Select an unassigned vessel and try to assign it using find_feasible_assignments
    - 50% chance: Select an assigned vessel, unassign it, and try to reassign with find_feasible_assignments
    
    This creates diversification by either expanding the solution (assign unassigned) or 
    disrupting existing assignments (reassign assigned). The random selection from feasible
    assignments introduces stochasticity while maintaining constraint feasibility.
    
    Hyper-parameters in kwargs:
        - seed (int, optional): Random seed for reproducible vessel selection and assignment choice.
        - max_feasible_results (int, default=5): Maximum feasible assignments to generate for random selection.
        - force_mode (str, default=None): Override 50/50 probability - 'assign' forces unassigned selection, 'reassign' forces assigned selection.
    
    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - vessel_num (int): Total number of vessels.
            - current_solution (Solution): Current solution to identify assigned/unassigned vessels.
            - get_unassigned_vessels (callable): To get list of unassigned vessel IDs.
            - find_feasible_assignments (callable): To generate feasible assignments for selected vessel.
        algorithm_data (dict): Not used in this algorithm (can be empty).
    
    Returns:
        BaseOperator: CompleteVesselAssignmentOperator for assignment/reassignment, 
                      UnassignVesselOperator for destruction phase of reassignment,
                      or None if no action possible.
        dict: Updated algorithm data with {'perturbation_type': str ('assign'/'reassign'/'none'), 
                                           'target_vessel': int (vessel ID affected, -1 if none),
                                           'action_success': bool}.
    """
    # Extract hyper-parameters
    seed = kwargs.get('seed', None)
    max_feasible_results = kwargs.get('max_feasible_results', 5)
    force_mode = kwargs.get('force_mode', None)
    
    if seed is not None:
        random.seed(seed)
    
    # Get current solution and helper functions
    current_solution = problem_state['current_solution']
    vessel_num = problem_state['vessel_num']
    get_unassigned_vessels = problem_state['get_unassigned_vessels']
    find_feasible_assignments = problem_state['find_feasible_assignments']
    
    # Identify unassigned and assigned vessels
    unassigned_vessels = get_unassigned_vessels(current_solution)
    assigned_vessels = []
    for vessel_id in range(vessel_num):
        if vessel_id not in unassigned_vessels:
            assigned_vessels.append(vessel_id)
    
    # Determine perturbation strategy with 50/50 probability or force mode
    if force_mode == 'assign':
        strategy = 'assign'
    elif force_mode == 'reassign':
        strategy = 'reassign'
    else:
        # 50/50 random choice, but check availability
        available_strategies = []
        if unassigned_vessels:
            available_strategies.append('assign')
        if assigned_vessels:
            available_strategies.append('reassign')
        
        if not available_strategies:
            # No vessels to perturb (empty solution or all-assigned/no-feasible case)
            return None, {'perturbation_type': 'none', 'target_vessel': -1, 'action_success': False}
        
        if len(available_strategies) == 1:
            strategy = available_strategies[0]
        else:
            strategy = random.choice(['assign', 'reassign'])  # 50/50 choice
    
    # Execute strategy
    if strategy == 'assign':
        return _handle_assign_strategy(unassigned_vessels, find_feasible_assignments, max_feasible_results)
    elif strategy == 'reassign':
        return _handle_reassign_strategy(assigned_vessels, current_solution, find_feasible_assignments, max_feasible_results)
    else:
        return None, {'perturbation_type': 'none', 'target_vessel': -1, 'action_success': False}


def _handle_assign_strategy(unassigned_vessels: list, find_feasible_assignments: callable, max_feasible_results: int) -> tuple[BaseOperator, dict]:
    """Handle assignment of unassigned vessel."""
    if not unassigned_vessels:
        return None, {'perturbation_type': 'assign', 'target_vessel': -1, 'action_success': False}
    
    # Randomly select an unassigned vessel
    target_vessel = random.choice(unassigned_vessels)
    
    # Find feasible assignments for the vessel
    feasible_assignments = find_feasible_assignments(target_vessel, max_results=max_feasible_results)
    if not feasible_assignments:
        return None, {'perturbation_type': 'assign', 'target_vessel': target_vessel, 'action_success': False}
    
    # Randomly select from feasible assignments (not necessarily the cheapest)
    selected_assignment = random.choice(feasible_assignments)
    
    # Create assignment operator
    operator = CompleteVesselAssignmentOperator(
        vessel_id=target_vessel,
        berth_id=selected_assignment['berth_id'],
        start_time=selected_assignment['berth_start_time'],
        inbound_tugboats=selected_assignment['inbound_tugboats'],
        outbound_tugboats=selected_assignment['outbound_tugboats']
    )
    
    return operator, {'perturbation_type': 'assign', 'target_vessel': target_vessel, 'action_success': True}


def _handle_reassign_strategy(assigned_vessels: list, current_solution: Solution, find_feasible_assignments: callable, max_feasible_results: int) -> tuple[BaseOperator, dict]:
    """Handle reassignment of assigned vessel (destroy then repair)."""
    if not assigned_vessels:
        return None, {'perturbation_type': 'reassign', 'target_vessel': -1, 'action_success': False}
    
    # Randomly select an assigned vessel
    target_vessel = random.choice(assigned_vessels)
    
    # Create a temporary solution without this vessel's assignment to check feasibility
    temp_solution = Solution(
        vessel_assignments=current_solution.vessel_assignments.copy(),
        tugboat_inbound_assignments=current_solution.tugboat_inbound_assignments.copy(),
        tugboat_outbound_assignments=current_solution.tugboat_outbound_assignments.copy()
    )
    
    # Remove the vessel's current assignment from temporary solution
    temp_solution.vessel_assignments[target_vessel] = None
    temp_solution.tugboat_inbound_assignments[target_vessel] = []
    temp_solution.tugboat_outbound_assignments[target_vessel] = []
    
    # Find feasible assignments for the vessel in the context without its current assignment
    feasible_assignments = find_feasible_assignments(target_vessel, max_results=max_feasible_results, solution=temp_solution)
    
    if not feasible_assignments:
        # No feasible reassignment: return unassign operator (destruction only)
        operator = UnassignVesselOperator(target_vessel)
        return operator, {'perturbation_type': 'reassign', 'target_vessel': target_vessel, 'action_success': False}
    
    # Randomly select from feasible reassignments
    selected_assignment = random.choice(feasible_assignments)
    
    # Create reassignment operator (this will overwrite the current assignment)
    operator = CompleteVesselAssignmentOperator(
        vessel_id=target_vessel,
        berth_id=selected_assignment['berth_id'],
        start_time=selected_assignment['berth_start_time'],
        inbound_tugboats=selected_assignment['inbound_tugboats'],
        outbound_tugboats=selected_assignment['outbound_tugboats']
    )
    
    return operator, {'perturbation_type': 'reassign', 'target_vessel': target_vessel, 'action_success': True}