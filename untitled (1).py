import random
import os
import numpy as np
from datetime import datetime, timedelta

class RealisticPortDataGenerator:
    """现实化港口调度问题数据生成器 - 时间参数与T挂钩"""
    
    def __init__(self, seed=42):
        """初始化数据生成器"""
        random.seed(seed)
        np.random.seed(seed)
        
        # 现实港口参数配置 - 平衡复杂度与可解性
        self.ship_type_config = {
            'container': {'size_range': (3, 5), 'ratio': 0.35, 'priority_range': (2.2, 3.0)},
            'bulk': {'size_range': (3, 5), 'ratio': 0.3, 'priority_range': (1.8, 2.5)},
            'tanker': {'size_range': (2, 4), 'ratio': 0.25, 'priority_range': (2.0, 2.8)},
            'general': {'size_range': (1, 3), 'ratio': 0.1, 'priority_range': (1.2, 2.0)}
        }
    
    def _calculate_time_in_periods(self, hours, time_periods):
        """将小时转换为时间周期数"""
        if time_periods == 12:  # 每周期2小时
            return max(1, round(hours / 2))
        elif time_periods == 24:  # 每周期1小时  
            return max(1, round(hours))
        elif time_periods == 48:  # 每周期30分钟
            return max(1, round(hours * 2))
        else:
            # 默认按1小时/周期处理
            return max(1, round(hours))
    
    def _generate_realistic_vessels(self, vessel_num, time_periods):
        """生成现实化的船舶数据 - 时间参数与T挂钩"""
        vessels = []
        
        # 按船型比例分配
        for ship_type, config in self.ship_type_config.items():
            count = int(vessel_num * config['ratio'])
            if ship_type == 'general':  # 确保总数正确
                count = vessel_num - len(vessels)
            
            for _ in range(count):
                vessel = {
                    'type': ship_type,
                    'size': random.randint(*config['size_range']),
                    'priority': round(random.uniform(*config['priority_range']), 1)
                }
                vessels.append(vessel)
        
        # 打乱顺序
        random.shuffle(vessels)
        
        # 生成到达时间 - 现实中船舶到达相对分散
        vessel_etas = self._generate_realistic_etas(vessel_num, time_periods)
        
        # 泊位服务时间 1-3小时，转换为周期数
        vessel_durations = []
        for vessel in vessels:
            # 根据船舶类型和大小确定服务时间
            if vessel['type'] == 'container':
                base_hours = random.uniform(2.0, 3.0)  # 集装箱船作业时间较长
            elif vessel['type'] == 'tanker':
                base_hours = random.uniform(1.5, 2.5)  # 油轮适中
            elif vessel['type'] == 'bulk':
                base_hours = random.uniform(1.5, 2.5)  # 散货船适中
            else:  # general
                base_hours = random.uniform(1.0, 2.0)  # 杂货船较短
            
            # 大船服务时间更长
            if vessel['size'] >= 4:
                base_hours *= 1.2
            
            duration_periods = self._calculate_time_in_periods(base_hours, time_periods)
            vessel_durations.append(duration_periods)
        
        # 进港服务时间 1-2小时，转换为周期数
        vessel_inbound_times = []
        for vessel in vessels:
            base_hours = random.uniform(1.0, 2.0)
            # 大船需要更多拖船服务时间
            if vessel['size'] >= 4:
                base_hours *= 1.3
            elif vessel['size'] >= 3:
                base_hours *= 1.1
            
            inbound_periods = self._calculate_time_in_periods(base_hours, time_periods)
            vessel_inbound_times.append(inbound_periods)
        
        # 离港服务时间 1-2小时，转换为周期数
        vessel_outbound_times = []
        for vessel in vessels:
            base_hours = random.uniform(1.0, 2.0)
            # 大船需要更多拖船服务时间
            if vessel['size'] >= 4:
                base_hours *= 1.3
            elif vessel['size'] >= 3:
                base_hours *= 1.1
            
            outbound_periods = self._calculate_time_in_periods(base_hours, time_periods)
            vessel_outbound_times.append(outbound_periods)
        
        # 时间窗 - 早到限制 1-3小时
        vessel_early_limits = []
        for vessel in vessels:
            if vessel['type'] == 'container':  # 集装箱船时间要求较严格
                early_hours = random.uniform(1.0, 2.0)
            elif vessel['type'] == 'tanker':  # 油轮适中
                early_hours = random.uniform(1.5, 2.5)
            else:  # 散货船稍宽松
                early_hours = random.uniform(2.0, 3.0)
            
            early_periods = self._calculate_time_in_periods(early_hours, time_periods)
            vessel_early_limits.append(early_periods)
        
        # 时间窗 - 晚到限制 1-3小时
        vessel_late_limits = []
        for vessel in vessels:
            if vessel['type'] == 'container':  # 集装箱船时间要求较严格
                late_hours = random.uniform(1.0, 2.0)
            elif vessel['type'] == 'tanker':  # 油轮适中
                late_hours = random.uniform(1.5, 2.5)
            else:  # 散货船稍宽松
                late_hours = random.uniform(2.0, 3.0)
            
            late_periods = self._calculate_time_in_periods(late_hours, time_periods)
            vessel_late_limits.append(late_periods)
        
        vessel_sizes = [v['size'] for v in vessels]
        vessel_priorities = [v['priority'] for v in vessels]
        
        return (vessel_sizes, vessel_etas, vessel_durations, vessel_inbound_times,
                vessel_outbound_times, vessel_priorities, vessel_early_limits, vessel_late_limits)
    
    def _generate_realistic_etas(self, vessel_num, time_periods):
        """生成现实化的ETA分布 - 平衡集中度与可行性"""
        # 适度的到达集中，避免过度拥挤
        
        vessel_etas = []
        
        # 前1/3时间为主要到达期
        main_period = max(3, time_periods // 3)
        # 中1/3为次要到达期  
        mid_start = main_period + 1
        mid_end = min(time_periods - 4, time_periods * 2 // 3)
        
        # 分配到达：60%主要期，30%次要期，10%后期
        main_arrivals = int(vessel_num * 0.6)
        mid_arrivals = int(vessel_num * 0.3)
        late_arrivals = vessel_num - main_arrivals - mid_arrivals
        
        # 主要期到达
        for _ in range(main_arrivals):
            eta = random.randint(1, main_period)
            vessel_etas.append(eta)
        
        # 次要期到达
        for _ in range(mid_arrivals):
            eta = random.randint(mid_start, mid_end)
            vessel_etas.append(eta)
        
        # 后期到达
        for _ in range(late_arrivals):
            eta = random.randint(mid_end + 1, min(time_periods - 3, time_periods))
            vessel_etas.append(eta)
        
        random.shuffle(vessel_etas)
        return vessel_etas
    
    def _generate_realistic_berths(self, berth_num, max_vessel_size):
        """生成现实化的泊位配置 - 确保足够容量但保持挑战性"""
        # 平衡的泊位配置：既有竞争又能满足需求
        
        berth_capacities = []
        
        # 25%大型泊位（5级）- 足够但有限
        large_berths = max(1, berth_num // 4)
        berth_capacities.extend([5] * large_berths)
        
        # 40%中大型泊位（4级）- 主力泊位
        medium_large_berths = berth_num * 2 // 5
        berth_capacities.extend([4] * medium_large_berths)
        
        # 25%中型泊位（3级）
        medium_berths = berth_num // 4
        berth_capacities.extend([3] * medium_berths)
        
        # 10%中小型泊位（2级）
        small_berths = berth_num - len(berth_capacities)
        berth_capacities.extend([2] * small_berths)
        
        # 智能调整：确保大船有足够选择
        large_capacity_berths = sum(1 for cap in berth_capacities if cap >= 4)
        
        # 确保至少有足够的4+级泊位
        if large_capacity_berths < berth_num // 3:  # 至少1/3是大中型泊位
            # 升级一些3级泊位为4级
            for i in range(len(berth_capacities)):
                if berth_capacities[i] == 3 and large_capacity_berths < berth_num // 3:
                    berth_capacities[i] = 4
                    large_capacity_berths += 1
        
        return berth_capacities
    
    def _generate_realistic_tugboats(self, tugboat_num, vessel_sizes):
        """生成现实化的拖船配置 - 确保充足但有成本差异"""
        # 充足的拖船配置，但有明显的成本层次
        
        tugboat_capacities = []
        tugboat_costs = []
        
        # 30%高能力拖船（5级）- 昂贵但充足
        high_capacity_count = max(3, tugboat_num * 3 // 10)
        tugboat_capacities.extend([5] * high_capacity_count)
        tugboat_costs.extend([round(random.uniform(120, 160), 1) for _ in range(high_capacity_count)])
        
        # 35%中高能力拖船（4级）- 性价比好
        mid_high_count = tugboat_num * 35 // 100
        tugboat_capacities.extend([4] * mid_high_count)
        tugboat_costs.extend([round(random.uniform(80, 120), 1) for _ in range(mid_high_count)])
        
        # 25%中等能力拖船（3级）
        mid_count = tugboat_num // 4
        tugboat_capacities.extend([3] * mid_count)
        tugboat_costs.extend([round(random.uniform(50, 80), 1) for _ in range(mid_count)])
        
        # 10%较低能力拖船（2级）
        remaining_count = tugboat_num - len(tugboat_capacities)
        tugboat_capacities.extend([2] * remaining_count)
        tugboat_costs.extend([round(random.uniform(30, 50), 1) for _ in range(remaining_count)])
        
        # 智能调整：确保每种船舶类型都有充足服务
        vessel_size_counts = {}
        for size in vessel_sizes:
            vessel_size_counts[size] = vessel_size_counts.get(size, 0) + 1
        
        # 检查每种尺寸的拖船充足度
        for size, count in vessel_size_counts.items():
            if size >= 4:  # 大船需要特别关注
                capable_tugs = sum(1 for cap in tugboat_capacities if cap >= size)
                # 需要至少count * 1.5倍的拖船（考虑进港出港重叠）
                needed_tugs = max(2, int(count * 1.5))
                
                if capable_tugs < needed_tugs:
                    # 升级一些拖船的能力
                    upgrade_count = min(needed_tugs - capable_tugs, 
                                      len([cap for cap in tugboat_capacities if cap < size]))
                    upgraded = 0
                    for i in range(len(tugboat_capacities)):
                        if tugboat_capacities[i] < size and upgraded < upgrade_count:
                            tugboat_capacities[i] = size
                            # 调整成本
                            if size == 5:
                                tugboat_costs[i] = round(random.uniform(120, 160), 1)
                            else:
                                tugboat_costs[i] = round(random.uniform(80, 120), 1)
                            upgraded += 1
        
        return tugboat_capacities, tugboat_costs
    
    def _generate_realistic_costs(self, vessel_sizes):
        """生成现实化的成本参数"""
        vessel_waiting_costs = []
        vessel_jit_costs = []
        
        for size in vessel_sizes:
            # 根据船舶大小确定成本
            if size >= 4:  # 大船
                waiting_cost = round(random.uniform(40, 80), 1)
                jit_cost = round(random.uniform(15, 30), 1)
            elif size >= 3:  # 中船
                waiting_cost = round(random.uniform(25, 45), 1)
                jit_cost = round(random.uniform(10, 20), 1)
            else:  # 小船
                waiting_cost = round(random.uniform(15, 30), 1)
                jit_cost = round(random.uniform(5, 15), 1)
            
            vessel_waiting_costs.append(waiting_cost)
            vessel_jit_costs.append(jit_cost)
        
        return vessel_waiting_costs, vessel_jit_costs
    
    def _calculate_realistic_time_periods(self, vessel_num, berth_num, avg_duration):
        """计算现实的时间段需求 - 确保充足但紧凑"""
        # 基于实际需求，确保可解但有适度压力
        
        # 基础时间：考虑并行作业能力
        base_time = max(12, (avg_duration * vessel_num) // berth_num + 6)
        
        # 添加合理缓冲
        buffer_time = max(4, base_time // 4)
        total_time = base_time + buffer_time
        
        # 限制在标准范围内
        if total_time <= 16:
            return 12  
        elif total_time <= 32:
            return 24  
        else:
            return 48
    
    def _get_time_unit_info(self, time_periods):
        """获取时间单位信息"""
        if time_periods == 12:
            return "2 hours", 2.0
        elif time_periods == 24:
            return "1 hour", 1.0
        elif time_periods == 48:
            return "30 minutes", 0.5
        else:
            return "1 hour", 1.0
    
    def _print_realistic_analysis(self, vessel_sizes, berth_capacities, tugboat_capacities, 
                                vessel_num, berth_num, tugboat_num, time_periods, 
                                vessel_durations, vessel_inbound_times, vessel_outbound_times):
        """打印现实化分析"""
        time_unit, hours_per_period = self._get_time_unit_info(time_periods)
        
        print(f"\n=== 现实化港口调度案例分析 ===")
        print(f"时间配置: T={time_periods}, 每周期={time_unit}")
        print(f"船舶数量: {vessel_num}, 泊位数量: {berth_num}, 拖船数量: {tugboat_num}")
        
        # 船舶规模分析
        size_dist = {}
        for size in vessel_sizes:
            size_dist[size] = size_dist.get(size, 0) + 1
        print(f"船舶规模分布: {dict(sorted(size_dist.items()))}")
        
        # 泊位容量分析
        berth_dist = {}
        for cap in berth_capacities:
            berth_dist[cap] = berth_dist.get(cap, 0) + 1
        print(f"泊位容量分布: {dict(sorted(berth_dist.items()))}")
        
        # 拖船能力分析
        tug_dist = {}
        for cap in tugboat_capacities:
            tug_dist[cap] = tug_dist.get(cap, 0) + 1
        print(f"拖船能力分布: {dict(sorted(tug_dist.items()))}")
        
        # 时间分析
        avg_duration = sum(vessel_durations) / len(vessel_durations)
        avg_inbound = sum(vessel_inbound_times) / len(vessel_inbound_times)
        avg_outbound = sum(vessel_outbound_times) / len(vessel_outbound_times)
        
        print(f"平均泊位服务时间: {avg_duration:.1f}周期 ({avg_duration*hours_per_period:.1f}小时)")
        print(f"平均进港服务时间: {avg_inbound:.1f}周期 ({avg_inbound*hours_per_period:.1f}小时)")
        print(f"平均离港服务时间: {avg_outbound:.1f}周期 ({avg_outbound*hours_per_period:.1f}小时)")
        
        # 容量匹配分析
        large_ships = sum(1 for size in vessel_sizes if size >= 4)
        large_berths = sum(1 for cap in berth_capacities if cap >= 4)
        large_tugs = sum(1 for cap in tugboat_capacities if cap >= 4)
        
        print(f"大船(≥4级): {large_ships}艘")
        print(f"大泊位(≥4级): {large_berths}个")
        print(f"大拖船(≥4级): {large_tugs}艘")
        
        if large_ships > 0:
            berth_ratio = large_berths / large_ships
            tug_ratio = large_tugs / large_ships
            print(f"大船资源配比 - 泊位: {berth_ratio:.2f}, 拖船: {tug_ratio:.2f}")
    
    def generate_realistic_case(self, vessel_num, berth_num, tugboat_num, time_periods=None, 
                              case_name="realistic_case", output_dir="cases"):
        """生成现实化的港口调度用例"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果没有指定时间段，自动计算（限制在12/24/48范围内）
        if time_periods is None:
            # 基于船舶数量和泊位数量估算合理的时间段
            estimated_periods = max(12, min(48, vessel_num // berth_num * 8))
            if estimated_periods <= 16:
                time_periods = 12
            elif estimated_periods <= 32:
                time_periods = 24
            else:
                time_periods = 48
            print(f"  Auto-calculated time_periods: {time_periods}")
        
        # 确保时间段在合理范围内
        if time_periods not in [12, 24, 48]:
            if time_periods <= 15:
                time_periods = 12
            elif time_periods <= 36:
                time_periods = 24
            else:
                time_periods = 48
            print(f"  Adjusted time_periods to: {time_periods}")
        
        # 生成船舶数据
        (vessel_sizes, vessel_etas, vessel_durations, vessel_inbound_times,
         vessel_outbound_times, vessel_priorities, vessel_early_limits, 
         vessel_late_limits) = self._generate_realistic_vessels(vessel_num, time_periods)
        
        # 生成泊位配置
        berth_capacities = self._generate_realistic_berths(berth_num, max(vessel_sizes))
        
        # 生成拖船配置
        tugboat_capacities, tugboat_costs = self._generate_realistic_tugboats(tugboat_num, vessel_sizes)
        
        # 生成成本参数
        vessel_waiting_costs, vessel_jit_costs = self._generate_realistic_costs(vessel_sizes)
        
        # 系统参数 - 基于现实港口操作，转换为周期数
        inbound_preparation_time = self._calculate_time_in_periods(1.0, time_periods)  # 1小时
        outbound_preparation_time = self._calculate_time_in_periods(1.0, time_periods)  # 1小时
        time_constraint_tolerance = self._calculate_time_in_periods(1.5, time_periods)  # 1-2小时，取中位数1.5小时
        penalty_parameter = 5000.0  # 未服务惩罚较大
        
        # 目标权重 - 现实港口优先级
        # 1. 减少未服务船舶最重要
        # 2. 减少等待时间很重要  
        # 3. JIT准时性重要
        # 4. 拖船成本相对次要
        objective_weights = [0.5, 0.3, 0.15, 0.05]
        
        # 写入文件
        filename = os.path.join(output_dir, f"{case_name}.txt")
        time_unit, hours_per_period = self._get_time_unit_info(time_periods)
        
        with open(filename, 'w') as f:
            f.write("# Realistic Port Scheduling Problem Instance Data\n")
            f.write("# Generated with realistic maritime industry parameters\n")
            f.write("# All time parameters are scaled to match the time period configuration\n")
            f.write(f"# Time configuration: T={time_periods}, each period = {time_unit}\n")
            f.write("# Format: parameter_name = value or [array values]\n\n")
            
            # 基本维度
            f.write("# Basic dimensions\n")
            f.write(f"vessel_num = {vessel_num}\n")
            f.write(f"berth_num = {berth_num}\n")
            f.write(f"tugboat_num = {tugboat_num}\n")
            f.write(f"time_periods = {time_periods}\n\n")
            
            # 船舶数据
            f.write(f"# Vessel data ({vessel_num} vessels) - Time-scaled parameters\n")
            f.write(f"# Service durations: 1-3 hours scaled to periods\n")
            f.write(f"# Inbound/Outbound times: 1-2 hours scaled to periods\n")
            f.write(f"# Time windows: 1-3 hours scaled to periods\n")
            f.write(f"vessel_sizes = {vessel_sizes}\n")
            f.write(f"vessel_etas = {vessel_etas}\n")
            f.write(f"vessel_durations = {vessel_durations}\n")
            f.write(f"vessel_inbound_service_times = {vessel_inbound_times}\n")
            f.write(f"vessel_outbound_service_times = {vessel_outbound_times}\n")
            f.write(f"vessel_priority_weights = {vessel_priorities}\n")
            f.write(f"vessel_waiting_costs = {vessel_waiting_costs}\n")
            f.write(f"vessel_jit_costs = {vessel_jit_costs}\n")
            f.write(f"vessel_early_limits = {vessel_early_limits}\n")
            f.write(f"vessel_late_limits = {vessel_late_limits}\n\n")
            
            # 泊位数据
            f.write(f"# Berth data ({berth_num} berths) - Realistic size distribution\n")
            f.write(f"berth_capacities = {berth_capacities}\n\n")
            
            # 拖船数据
            f.write(f"# Tugboat data ({tugboat_num} tugboats) - Realistic fleet composition\n")
            f.write(f"tugboat_capacities = {tugboat_capacities}\n")
            f.write(f"tugboat_costs = {tugboat_costs}\n\n")
            
            # 系统参数
            f.write("# System parameters - Scaled to time periods\n")
            f.write(f"# Preparation times: 1 hour each = {inbound_preparation_time} periods\n")
            f.write(f"# Time tolerance: 1-2 hours = {time_constraint_tolerance} periods\n")
            f.write(f"inbound_preparation_time = {inbound_preparation_time}\n")
            f.write(f"outbound_preparation_time = {outbound_preparation_time}\n")
            f.write(f"time_constraint_tolerance = {time_constraint_tolerance}\n")
            f.write(f"penalty_parameter = {penalty_parameter}\n")
            f.write(f"objective_weights = {objective_weights}\n\n")
            
            # 数学模型参数
            f.write("# Mathematical model parameters\n")
            f.write(f"M = {penalty_parameter}\n")
            f.write(f"lambda_1 = {objective_weights[0]}\n")
            f.write(f"lambda_2 = {objective_weights[1]}\n")
            f.write(f"lambda_3 = {objective_weights[2]}\n")
            f.write(f"lambda_4 = {objective_weights[3]}\n")
            f.write(f"epsilon_time = {time_constraint_tolerance}.0\n")
            f.write(f"rho_in = {inbound_preparation_time}\n")
            f.write(f"rho_out = {outbound_preparation_time}\n")
            f.write(f"tau_in = {vessel_inbound_times}\n")
            f.write(f"tau_out = {vessel_outbound_times}\n")
            f.write(f"alpha = {vessel_priorities}\n")
            f.write(f"beta = {vessel_waiting_costs}\n")
            f.write(f"gamma = {vessel_jit_costs}\n")
            f.write(f"c_k = {tugboat_costs}\n")
            f.write(f"Delta_early = {vessel_early_limits}\n")
            f.write(f"Delta_late = {vessel_late_limits}\n")
        
        # 打印现实化分析
        self._print_realistic_analysis(vessel_sizes, berth_capacities, tugboat_capacities, 
                                     vessel_num, berth_num, tugboat_num, time_periods,
                                     vessel_durations, vessel_inbound_times, vessel_outbound_times)
        print(f"Generated realistic case: {filename}")
        return filename

# 使用示例
if __name__ == "__main__":
    generator = RealisticPortDataGenerator(seed=42)
    
    # 生成不同时间精度的测试用例
    test_cases = [
        (10, 4, 8, 12, "small_2h_periods"),     # 小规模，2小时/周期
        (15, 5, 10, 24, "medium_1h_periods"),   # 中规模，1小时/周期  
        (20, 6, 12, 48, "large_30min_periods"), # 大规模，30分钟/周期
    ]
    
    for vessel_num, berth_num, tugboat_num, time_periods, case_name in test_cases:
        print(f"\n生成测试用例: {case_name}")
        generator.generate_realistic_case(
            vessel_num=vessel_num,
            berth_num=berth_num, 
            tugboat_num=tugboat_num,
            time_periods=time_periods,
            case_name=case_name
        )