import yaml
import json
import logging
from typing import Dict, Any, List
from src.base_agent import BaseAgent
from tenacity import retry, stop_after_attempt, wait_exponential
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_planner_prompt():
    try:
        with open("src/planner/planner_prompt.yaml", "r") as f:
            prompt_dict = yaml.safe_load(f)
            return (
                prompt_dict["instruction_template"],
                prompt_dict["response_json_schema"]
            )
    except Exception as e:
        logger.error(f"Failed to load planner prompt: {str(e)}")
        raise

class Planner(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"Planner initialized with config: {config}")
        super().__init__(config)
        self.instruction_template, self.response_json_schema = load_planner_prompt()
        logger.info(f"Planner initialized successfully")

    def reply(self, environment_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        planner将根据环境上下文environment_context和当前工作区内容workspace_context，选择以下几种回复方式：
        1. 接收到来自User的信息，开始制定计划，并把计划的第一个子任务发送给CodeInterpreter。
        2. 接收到来自CodeInterpreter的符合预期的回复，将把计划的下一个子任务发送给CodeInterpreter或根据回复内容向User完成最终回答。
        3. CodeInterpreter执行失败，将根据总结的历史经验重新制定计划，并把计划的第一个子任务发送给CodeInterpreter。
        Args:
            environment_context (Dict[str, Any]): 多智能体共同对话记录
            workspace_context (List[str]): 当前工作区文件列表

        Returns:
            Dict[str, Any]: json格式的回复
        """
        
        similar_cases: List[str, None] = []
        
        # 1. 接收到来自User的信息，开始制定计划，并把计划的第一个子任务发送给CodeInterpreter。
        if environment_context[-1]["role"] == "User":
            # 1.1 从环境上下文中提取出所有User和Planner的对话记录
            dialog = ""
            for message in environment_context:
                if message["role"] == "User" or message["role"] == "Planner":
                    dialog += f"{message['role']}: {message['message']}\n"
            # 1.2 根据对话记录，检索出相似的case
            similar_cases = self.case_retriever.search(query=dialog, top_k=1)
            logger.info(f"Similar cases: {similar_cases}")

        # 2. 接收到来自CodeInterpreter的符合预期的回复，将把计划的下一个子任务发送给CodeInterpreter或根据回复内容向User完成最终回答。
        # 3. CodeInterpreter执行失败，将根据总结的历史经验重新制定计划，并把计划的第一个子任务发送给CodeInterpreter。
        # 这两种情况不需要检索相似案例
        try:
            response_json = self._call_llm_for_json(
                messages=[
                    {"role": "user", "content": self.instruction_template.format(
                    environment_context=environment_context,
                    workspace_context=self.workspace_context,
                    similar_cases=similar_cases,
                    response_json_schema=self.response_json_schema
                )},
                {"role": "assistant", "content": "```json"}
            ],
                response_json_schema=self.response_json_schema
            )
        except Exception as e:
            logger.error(f"Failed to call LLM: {str(e)}")
            raise
        final_response = self._decorate_response(response_json["response"], role="Planner", send_to=None)
        return final_response
        

if __name__ == "__main__":
    from rich import print
    with open("config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    planner = Planner(config["planner"])
    
    environment_context = [
        {
            "role": "User",
            "message": "读取test.txt文件内容，并将其复制到test2.txt文件中",
            "send_to": "Planner"
        },
        {
            "role": "Planner",
            "plan_reasoning": '读取test.txt文件内容，并将其复制到test2.txt文件中',
            "init_plan": '用代码读取test.txt文件内容，并将其复制到test2.txt文件中',
            "plan": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中",
            "current_plan_step": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中",
            "stop": "InProcess",
            "message": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中",
            "send_to": "CodeInterpreter"
        },
        {
            'role': 'CodeInterpreter',
            'thought': "根据任务要求，我需要读取'test.txt'文件的内容并将其复制到'test2.txt'文件中。然而，当前工作区中并没有'test.txt'文件。为了完成任务，我将假设'test.txt'存在，并编写相应的代码来处理文件操作。",
            'reply_type': 'python',
            'reply_content': '# 假设 \'test.txt\' 文件存在于当前工作区\ntry:\n    with open(\'test.txt\', \'r\', encoding=\'utf-8\') as file:\n        content = file.read()\nexcept FileNotFoundError:\n    content = ""\n    print("Error: \'test.txt\' not found.")\n\n# 将内容写入 \'test2.txt\'\nwith open(\'test2.txt\', \'w\', encoding=\'utf-8\') as file:\n    file.write(content)\n',
            'status': True,
            'output': "Error: 'test.txt' not found.",
            'error': None,
            'send_to': 'Planner'
        },
    ]
    
    response_json = planner.reply(environment_context)
    print(response_json)