from typing import List, Dict, Any
from rich import print
import json
class EnvironmentContextManager:
    def __init__(self):
        self.environment_context: List[Dict[str, Any]] = []
    
    def get_environment_context(self) -> List[Dict[str, Any]]:
        return self.environment_context
    
    def add_context(self, response: Dict[str, Any]):
        self.environment_context.append(response)
    
    def whose_turn(self) -> str:
        """
        EnvironmentContextManager的核心功能。
        1. 根据response中的send_to字段，判断当前轮到哪个角色发言。
        2. 根据规则判断任务是否结束。
        """
        # 如果环境上下文为空，说明需要User发言，开始任务
        if not self.environment_context:
            return "User"
        # 如果环境上下文不为空，判断当前轮到哪个角色发言
        else:
            return self.environment_context[-1]["send_to"]
    
    def is_task_finished(self) -> bool:
        """
        根据规则判断任务是否结束。
        """
        if self.environment_context and self.environment_context[-1]["role"] == "Planner":
            if self.environment_context[-1]["stop"] == "Completed":
                return True
        return False
    
    def print_lastest_planner_message(self):
        """
        从列表末尾开始查找，找到最近一条Planner的消息，并打印。
        """
        if self.environment_context:
            for response in self.environment_context[::-1]:
                if response["role"] == "Planner":
                    print(f"Planner: {response['message']}")
                    break
        else:
            print("Planner: 你好，请您描述任务。")
    
    def save_environment_context(self, file_path: str):
        """
        保存环境上下文到文件。
        """
        with open(file_path, "w") as f:
            json.dump(self.environment_context, f, indent=4, ensure_ascii=False)

