import os  # 导入os模块，用于处理文件路径等操作系统相关功能
from src.problems.base.components import BaseOperator  # 导入操作基类，作为所有具体操作的父类
from src.problems.base.env import BaseEnv  # 导入环境基类，定义了问题求解的环境接口
from src.util.util import load_function  # 导入工具函数，用于加载启发式算法函数


class SingleConstructiveSingleImproveHyperHeuristic:
    def __init__(
        self,
        constructive_heuristic_file: str,  # 构造性启发式算法的文件路径
        improve_heuristic_file: str,  # 改进性启发式算法的文件路径
        problem: str,  # 问题名称，用于定位对应问题的相关资源
        iterations_scale_factor: float = 2.0  # 迭代规模因子，控制改进阶段的最大步数，默认值为2.0
    ) -> None:
        # 加载构造性启发式函数并赋值给实例变量
        self.constructive_heuristic = load_function(constructive_heuristic_file, problem=problem)
        # 加载改进性启发式函数并赋值给实例变量
        self.improve_heuristic = load_function(improve_heuristic_file, problem=problem)
        # 保存迭代迭代规模因子赋值        self.iterations_scale_factor = iterations_scale_factor

    def run(self, env: BaseEnv) -> bool:  # 运行超启发式算法，接收环境对象并返回解的有效性结果
        # 计算改进阶段的最大步数：构造阶段步数 × 迭代规模因子（取整）
        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        # 初始化启发式操作结果为BaseOperator实例，用于启动构造阶段循环
        heuristic_work = BaseOperator()
        # 构造阶段循环：持续执行构造性启发式，直到无法生成有效操作（非BaseOperator实例）
        while isinstance(heuristic_work, BaseOperator):
            # 调用环境的run_heuristic方法执行构造性启发式，获取操作结果
            heuristic_work = env.run_heuristic(self.constructive_heuristic)
        # 改进阶段循环：在剩余步数内执行改进性启发式
        for _ in range(max_steps - env.construction_steps):
            # 调用环境的run_heuristic方法执行改进性启发式，获取操作结果
            heuristic_work = env.run_heuristic(self.improve_heuristic)
            # 若操作结果无效，则提前退出改进阶段
            if not heuristic_work:
                break
        # 返回解的完整性和有效性判断结果（两者均为True时返回True）
        return env.is_complete_solution and env.is_valid_solution