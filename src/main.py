from src.planner import Planner
from src.code_interpreter import CodeInterpreter
from src.environment_context_manager import EnvironmentContextManager
from typing import Dict, Any
import yaml
import argparse
from datetime import datetime
def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main(config: Dict[str, Any]):
    planner = Planner(config["planner"])
    code_interpreter = CodeInterpreter(config["code_interpreter"])
    environment_context_manager = EnvironmentContextManager()
    
    print(f"planner: {planner.workspace_context}")
    print(f"code_interpreter: {code_interpreter.workspace_context}")

    while not environment_context_manager.is_task_finished():
        if environment_context_manager.whose_turn() == "User":
            environment_context_manager.print_lastest_planner_message()
            user_input = input("User: ")
            environment_context_manager.add_context({
                "role": "User",
                "message": user_input,
                "send_to": "Planner"
            })
        elif environment_context_manager.whose_turn() == "Planner":
            reply = planner.reply(environment_context_manager.get_environment_context())
            environment_context_manager.add_context(reply)
        else:
            reply = code_interpreter.reply(environment_context_manager.get_environment_context())
            environment_context_manager.add_context(reply)
    environment_context_manager.print_lastest_planner_message()
    print(f"Task finished")
    environment_context_manager.save_environment_context(f"src/logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_environment_context.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/test_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config_path)
    main(config)


