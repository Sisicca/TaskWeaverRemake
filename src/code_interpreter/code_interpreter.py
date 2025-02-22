from src.base_agent import BaseAgent
from typing import Dict, Any, List, Tuple
import yaml
import logging
import os
import re
from jupyter_client import KernelManager
from queue import Empty
import nbformat
import ast
import atexit
from rich import print

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_code_interpreter_prompt():
    try:
        with open("src/code_interpreter/code_interpreter_prompt.yaml", "r") as f:
            prompt_dict = yaml.safe_load(f)
            return (
                prompt_dict["instruction_template"],
                prompt_dict["response_json_schema"]
            )
    except Exception as e:
        logger.error(f"Failed to load code interpreter prompt: {str(e)}")
        raise

# 代码执行器，不属于BaseAgent，类似Retriever，属于智能体的组件，负责执行代码
class CodeExecutor:
    """代码执行器，负责安全地执行Python代码并管理执行环境"""
    
    def __init__(self, workspace_path: str, notebook_name: str = "execution_log.ipynb", timeout: int = 30):
        """
        初始化代码执行器
        
        Args:
            workspace_path: 工作目录路径
            timeout: 代码执行超时时间(秒)
        """
        if not workspace_path:
            raise ValueError("workspace_path cannot be empty")
            
        self.workspace_path = os.path.abspath(workspace_path)
        self.timeout = timeout
        self.notebook_path = os.path.join(self.workspace_path, notebook_name)
        
        # 确保工作目录存在
        os.makedirs(self.workspace_path, exist_ok=True)
        
        self.blocked_operations = []
        
        # 初始化notebook
        self.notebook = self._init_notebook()
        
        # 初始化kernel
        self.km = KernelManager()
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=self.timeout)

    def _init_notebook(self) -> nbformat.NotebookNode:
        """初始化notebook文件"""
        try:
            if os.path.exists(self.notebook_path):
                with open(self.notebook_path, 'r', encoding='utf-8') as f:
                    return nbformat.read(f, as_version=4)
            else:
                notebook = nbformat.v4.new_notebook()
                self._save_notebook(notebook)
                return notebook
        except Exception as e:
            logger.error(f"Failed to initialize notebook: {e}")
            raise

    def _save_notebook(self, notebook: nbformat.NotebookNode) -> None:
        """保存notebook到文件"""
        try:
            with open(self.notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)
        except Exception as e:
            logger.error(f"Failed to save notebook: {e}")
            raise

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """验证代码安全性"""
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        if name.name.split('.')[0] in self.blocked_operations:
                            return False, f"Forbidden module import: {name.name}"
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.blocked_operations:
                            return False, f"Forbidden function call: {node.func.id}"
                    elif isinstance(node.func, ast.Attribute):
                        if node.func.attr in self.blocked_operations:
                            return False, f"Forbidden method call: {node.func.attr}"
            
            return True, "Code validation passed"
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Code validation error: {str(e)}"

    def execute(self, code: str) -> Dict[str, Any]:
        """执行代码并返回结果"""
        if not code.strip():
            return {"status": False, "output": "", "error": "Empty code"}

        # 验证代码
        is_safe, message = self.validate_code(code)
        if not is_safe:
            return {"status": False, "output": "", "error": f"Code validation failed: {message}"}

        try:
            # 创建新的代码单元
            cell = nbformat.v4.new_code_cell(source=code)
            self.notebook.cells.append(cell)
            
            # 执行代码
            msg_id = self.kc.execute(code)
            outputs = []
            errors = []
            
            while True:
                try:
                    msg = self.kc.get_iopub_msg(timeout=self.timeout)
                    # print(f"msg: {msg}")
                    msg_type = msg['header']['msg_type']
                    
                    if msg_type == 'execute_result':
                        output = str(msg['content']['data'].get('text/plain', ''))
                        outputs.append(output)
                        cell.outputs.append(nbformat.v4.new_output('execute_result', 
                                                                 data={'text/plain': output}))
                    elif msg_type == 'stream':
                        text = msg['content']['text'].rstrip()  # 移除末尾空白字符
                        if text:  # 只添加非空输出
                            outputs.append(text)
                            cell.outputs.append(nbformat.v4.new_output('stream',
                                                                 name=msg['content']['name'],
                                                                 text=text))
                    elif msg_type == 'error':
                        error = '\n'.join(msg['content']['traceback'])
                        errors.append(error)
                        cell.outputs.append(nbformat.v4.new_output('error',
                                                                 ename=msg['content']['ename'],
                                                                 evalue=msg['content']['evalue'],
                                                                 traceback=msg['content']['traceback']))
                    elif msg_type == 'status' and msg['content']['execution_state'] == 'idle':
                        break
                        
                except Empty:
                    errors.append(f"Execution timeout ({self.timeout}s)")
                    break
            
            # 保存notebook
            self._save_notebook(self.notebook)
            
            return {
                "status": len(errors) == 0,
                "output": '\n'.join(outputs),
                "error": '\n'.join(errors) if errors else None
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return {"status": False, "output": "", "error": str(e)}

    def get_variable_state(self, var_name: str) -> Dict[str, Any]:
        """获取变量当前状态"""
        if not isinstance(var_name, str) or not var_name.strip():
            return {"name": var_name, "value": None, "error": "Invalid variable name"}
            
        code = f"print(repr({var_name}))"
        result = self.execute(code)
        
        return {
            "name": var_name,
            "value": result["output"] if result["status"] else None,
            "error": result["error"]
        }

    def get_environment_info(self) -> Dict[str, str]:
        """获取当前kernel环境信息"""
        env_info = {}
        
        # 获取Python解释器路径
        result = self.execute("import sys; print(sys.executable)")
        env_info['python_path'] = result['output'].strip()
        
        # 获取Python版本
        result = self.execute("import sys; print(sys.version)")
        env_info['python_version'] = result['output'].strip()
        
        # 获取当前工作目录
        result = self.execute("import os; print(os.getcwd())")
        env_info['working_directory'] = result['output'].strip()
        
        # 获取Python路径列表
        result = self.execute("import sys; print(sys.path)")
        env_info['python_path_list'] = result['output'].strip()
        
        return env_info

class CodeInterpreter(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        logger.info(f"CodeInterpreter initialized with config: {config}")
        super().__init__(config)
        self.instruction_template, self.response_json_schema = load_code_interpreter_prompt()
        self.executor = CodeExecutor(workspace_path=config["workspace_path"], notebook_name=config["notebook_name"])
        logger.info(f"CodeInterpreter initialized successfully")
        
    def reply(self, environment_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        CodeInterpreter将根据环境上下文environment_context和当前工作区内容workspace_context，选择以下几种回复方式：
        1. 接收到来自Planner的子任务，生成相应的Python代码并运行，根据运行结果选择发送对象（成功则发送给Planner，失败则发送回CodeInterpreter）。
        2. 接收到来自CodeInterpreter自身的错误信息，根据错误信息生成新的代码，根据运行结果选择发送对象（成功则发送给Planner，失败则发送回CodeInterpreter）。
        Args:
            environment_context (Dict[str, Any]): 多智能体共同对话记录
            workspace_context (List[str]): 当前工作区文件列表

        Returns:
            Dict[str, Any]: json格式的回复
        """
        
        # 更新工作区上下文
        self._update_workspace_context()
        
        similar_cases: List[str, None] = []
        
        # 1. 接收到来自Planner的子任务，生成相应的Python代码并运行，根据运行结果选择发送对象（成功则发送给Planner，失败则发送回CodeInterpreter）。
        if environment_context[-1]["role"] == "Planner":
            logger.info(f"Planner发送子任务给CodeInterpreter，开始构建代码")
            # 1.1 从环境上下文中提取出当前Planner的子任务current_plan_step和message
            current_plan_step = environment_context[-1]["current_plan_step"]
            message = environment_context[-1]["message"]
            # 1.2 根据当前子任务current_plan_step和message检索出相似的case
            similar_cases = self.case_retriever.search(query=f"current_plan_step: {current_plan_step}\nmessage: {message}", top_k=1)
            logger.info(f"Similar cases: {similar_cases}")
        
        # 2. 接收到来自CodeInterpreter自身的错误信息，根据错误信息生成新的代码，根据运行结果选择发送对象（成功则发送给Planner，失败则发送回CodeInterpreter）。
        try:
            response_json = self._call_llm_for_json(
                messages=[
                    {"role": "user", "content": self.instruction_template.format(
                    environment_context=environment_context,
                    workspace_context=self.workspace_context,
                    similar_cases=similar_cases,
                    response_json_schema=self.response_json_schema)}
            ],  
            response_json_schema=self.response_json_schema
        )
        except Exception as e:
            logger.error(f"Failed to call LLM: {str(e)}")
            raise
        
        # logger.info(f"CodeInterpreter初步回复: {response_json}")
        
        # 如果回复类型为python，则执行代码
        if response_json["response"]["reply_type"] == "python":
            try:
                result = self.executor.execute(response_json["response"]["reply_content"])
                # logger.info(f"CodeInterpreter执行代码结果: {result}")
                response_json["response"].update(result)
                
                if result["status"]: # 如果代码执行成功，则发送给Planner
                    final_response = self._decorate_response(response_json["response"], role="CodeInterpreter", send_to="Planner")
                else: # 如果代码执行失败，则发送回CodeInterpreter
                    final_response = self._decorate_response(response_json["response"], role="CodeInterpreter", send_to="CodeInterpreter")
                
                return final_response
            
            except Exception as e:
                logger.error(f"Failed to execute code: {str(e)}")
                raise
        else: # 如果回复类型为text，则直接返回回复内容  
            final_response = self._decorate_response(response_json["response"], role="CodeInterpreter", send_to="Planner")
            return final_response
    
if __name__ == "__main__":
    with open("config/test_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    test_code_interpreter = CodeInterpreter(config["code_interpreter"])
    
    environment_context = [
        {
            "role": "User",
            "send_to": "Planner",
            "message": "读取test.txt文件内容，并将其复制到test2.txt文件中"
        },
        {
            "role": "Planner",
            "plan_reasoning": '读取test.txt文件内容，并将其复制到test2.txt文件中',
            "init_plan": '用代码读取test.txt文件内容，并将其复制到test2.txt文件中',
            "plan": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中",
            "current_plan_step": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中",
            "stop": "InProcess",
            "send_to": "CodeInterpreter",
            "message": "用代码读取test.txt文件内容，并将其复制到test2.txt文件中"
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
        {
            'role': 'Planner',
            'send_to': 'CodeInterpreter',
            'current_plan_step': '重新读取test.txt文件内容，并将其复制到test2.txt文件中',
            'message': '用户补充了test.txt文件内容，请重新读取test.txt文件内容，并将其复制到test2.txt文件中'
        }
    ]
    
    response_json = test_code_interpreter.reply(environment_context)
    print(response_json)
    
    # workspace_dir = "src/workspaces/test"
    # notebook_name = "execution_log_1.ipynb"
    
    # executor = CodeExecutor(workspace_path=workspace_dir, notebook_name=notebook_name)
    # result = executor.execute("import time\ntime.sleep(5)")
    # print(result)