from src.problems.base.components import BaseSolution, BaseOperator
from typing import List, Tuple, Optional, Dict
import copy
import random

class Solution(BaseSolution):
    """The solution of Port Scheduling Problem.
    Contains vessel assignments, tugboat assignments, and timing information.
    
    Attributes:
        vessel_assignments (dict): Maps vessel_id to (berth_id, start_time) tuple. 
                                 None if vessel is unassigned.
        tugboat_inbound_assignments (dict): Maps vessel_id to list of (tugboat_id, start_time) tuples for inbound service.
        tugboat_outbound_assignments (dict): Maps vessel_id to list of (tugboat_id, start_time) tuples for outbound service.
    """
    def __init__(self, vessel_assignments: dict = None, 
                 tugboat_inbound_assignments: dict = None, 
                 tugboat_outbound_assignments: dict = None):
        self.vessel_assignments = vessel_assignments or {}
        self.tugboat_inbound_assignments = tugboat_inbound_assignments or {}
        self.tugboat_outbound_assignments = tugboat_outbound_assignments or {}

    def __str__(self) -> str:
        result = "Port Scheduling Solution:\n"
        result += "Vessel Assignments:\n"
        for vessel_id, assignment in self.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                result += f"  Vessel {vessel_id}: Berth {berth_id}, Start Time {start_time}\n"
            else:
                result += f"  Vessel {vessel_id}: Unassigned\n"
        
        result += "Inbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_inbound_assignments.items():
            if services:
                service_str = ", ".join([f"Tugboat {tug_id} at time {start_time}" 
                                       for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {service_str}\n"
        
        result += "Outbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_outbound_assignments.items():
            if services:
                service_str = ", ".join([f"Tugboat {tug_id} at time {start_time}" 
                                       for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {service_str}\n"
        
        return result

# ================== Core Assignment Operators ==================

class CompleteVesselAssignmentOperator(BaseOperator):
    """Assign a vessel to berth with complete tugboat services."""
    def __init__(self, vessel_id: int, berth_id: int, start_time: int, 
                 inbound_tugboats: List[Tuple[int, int]], 
                 outbound_tugboats: List[Tuple[int, int]]):
        self.vessel_id = vessel_id
        self.berth_id = berth_id
        self.start_time = start_time
        self.inbound_tugboats = inbound_tugboats
        self.outbound_tugboats = outbound_tugboats

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        new_vessel_assignments[self.vessel_id] = (self.berth_id, self.start_time)
        new_tugboat_inbound[self.vessel_id] = copy.deepcopy(self.inbound_tugboats)
        new_tugboat_outbound[self.vessel_id] = copy.deepcopy(self.outbound_tugboats)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

class UnassignVesselOperator(BaseOperator):
    """Remove assignment for a vessel."""
    def __init__(self, vessel_id: int):
        self.vessel_id = vessel_id

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        new_vessel_assignments[self.vessel_id] = None
        new_tugboat_inbound[self.vessel_id] = []
        new_tugboat_outbound[self.vessel_id] = []
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

# ================== Swap Operators ==================

class SwapVesselAssignmentsOperator(BaseOperator):
    """Swap complete assignments between two vessels."""
    def __init__(self, vessel_id1: int, vessel_id2: int):
        self.vessel_id1 = vessel_id1
        self.vessel_id2 = vessel_id2

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        # Swap vessel assignments
        assignment1 = new_vessel_assignments.get(self.vessel_id1)
        assignment2 = new_vessel_assignments.get(self.vessel_id2)
        new_vessel_assignments[self.vessel_id1] = assignment2
        new_vessel_assignments[self.vessel_id2] = assignment1
        
        # Swap tugboat assignments
        inbound1 = new_tugboat_inbound.get(self.vessel_id1, [])
        inbound2 = new_tugboat_inbound.get(self.vessel_id2, [])
        new_tugboat_inbound[self.vessel_id1] = copy.deepcopy(inbound2)
        new_tugboat_inbound[self.vessel_id2] = copy.deepcopy(inbound1)
        
        outbound1 = new_tugboat_outbound.get(self.vessel_id1, [])
        outbound2 = new_tugboat_outbound.get(self.vessel_id2, [])
        new_tugboat_outbound[self.vessel_id1] = copy.deepcopy(outbound2)
        new_tugboat_outbound[self.vessel_id2] = copy.deepcopy(outbound1)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

class SwapBerthsOperator(BaseOperator):
    """Swap berths between two vessels while keeping timing."""
    def __init__(self, vessel_id1: int, vessel_id2: int):
        self.vessel_id1 = vessel_id1
        self.vessel_id2 = vessel_id2

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        
        assignment1 = new_vessel_assignments.get(self.vessel_id1)
        assignment2 = new_vessel_assignments.get(self.vessel_id2)
        
        # Only swap if both vessels are assigned
        if assignment1 is not None and assignment2 is not None:
            berth_id1, start_time1 = assignment1
            berth_id2, start_time2 = assignment2
            
            # Swap berths, keep original start times
            new_vessel_assignments[self.vessel_id1] = (berth_id2, start_time1)
            new_vessel_assignments[self.vessel_id2] = (berth_id1, start_time2)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=copy.deepcopy(solution.tugboat_inbound_assignments),
            tugboat_outbound_assignments=copy.deepcopy(solution.tugboat_outbound_assignments)
        )

# ================== Time Adjustment Operators ==================

class TimeShiftVesselOperator(BaseOperator):
    """Shift a vessel's entire schedule by a time delta."""
    def __init__(self, vessel_id: int, time_delta: int):
        self.vessel_id = vessel_id
        self.time_delta = time_delta

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        # Shift vessel berth time
        if self.vessel_id in new_vessel_assignments and new_vessel_assignments[self.vessel_id] is not None:
            berth_id, start_time = new_vessel_assignments[self.vessel_id]
            new_start_time = max(1, start_time + self.time_delta)
            new_vessel_assignments[self.vessel_id] = (berth_id, new_start_time)
        
        # Shift tugboat services
        if self.vessel_id in new_tugboat_inbound:
            new_tugboat_inbound[self.vessel_id] = [
                (tug_id, max(1, start_time + self.time_delta)) 
                for tug_id, start_time in new_tugboat_inbound[self.vessel_id]
            ]
        
        if self.vessel_id in new_tugboat_outbound:
            new_tugboat_outbound[self.vessel_id] = [
                (tug_id, max(1, start_time + self.time_delta)) 
                for tug_id, start_time in new_tugboat_outbound[self.vessel_id]
            ]
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

# ================== Relocation Operators ==================

class RelocateVesselOperator(BaseOperator):
    """Move a vessel to a different berth and/or time."""
    def __init__(self, vessel_id: int, new_berth_id: int = None, new_start_time: int = None):
        self.vessel_id = vessel_id
        self.new_berth_id = new_berth_id
        self.new_start_time = new_start_time

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        
        current_assignment = new_vessel_assignments.get(self.vessel_id)
        
        # Handle both assigned and unassigned vessels
        if current_assignment is not None:
            current_berth_id, current_start_time = current_assignment
            berth_id = self.new_berth_id if self.new_berth_id is not None else current_berth_id
            start_time = self.new_start_time if self.new_start_time is not None else current_start_time
        else:
            # For unassigned vessels, both berth_id and start_time must be provided
            if self.new_berth_id is None or self.new_start_time is None:
                return copy.deepcopy(solution)  # Cannot relocate without full assignment
            berth_id = self.new_berth_id
            start_time = self.new_start_time
        
        new_vessel_assignments[self.vessel_id] = (berth_id, start_time)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=copy.deepcopy(solution.tugboat_inbound_assignments),
            tugboat_outbound_assignments=copy.deepcopy(solution.tugboat_outbound_assignments)
        )

class ReassignTugboatOperator(BaseOperator):
    """Reassign tugboat services for a vessel while keeping berth assignment."""
    def __init__(self, vessel_id: int, 
                 new_inbound_tugboats: List[Tuple[int, int]] = None, 
                 new_outbound_tugboats: List[Tuple[int, int]] = None):
        self.vessel_id = vessel_id
        self.new_inbound_tugboats = new_inbound_tugboats
        self.new_outbound_tugboats = new_outbound_tugboats

    def run(self, solution: Solution) -> Solution:
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        if self.new_inbound_tugboats is not None:
            new_tugboat_inbound[self.vessel_id] = copy.deepcopy(self.new_inbound_tugboats)
        if self.new_outbound_tugboats is not None:
            new_tugboat_outbound[self.vessel_id] = copy.deepcopy(self.new_outbound_tugboats)
        
        return Solution(
            vessel_assignments=copy.deepcopy(solution.vessel_assignments),
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

# ================== Destruction Operators ==================

class RandomDestructionOperator(BaseOperator):
    """Randomly remove a percentage of vessel assignments."""
    def __init__(self, destruction_rate: float = 0.3, seed: int = None):
        self.destruction_rate = destruction_rate
        self.seed = seed

    def run(self, solution: Solution) -> Solution:
        if self.seed is not None:
            random.seed(self.seed)
            
        assigned_vessels = [v_id for v_id, assignment in solution.vessel_assignments.items() 
                          if assignment is not None]
        
        if not assigned_vessels:
            return copy.deepcopy(solution)
        
        num_to_remove = max(1, int(len(assigned_vessels) * self.destruction_rate))
        vessels_to_remove = random.sample(assigned_vessels, num_to_remove)
        
        current_solution = copy.deepcopy(solution)
        for vessel_id in vessels_to_remove:
            current_solution = UnassignVesselOperator(vessel_id).run(current_solution)
        
        return current_solution