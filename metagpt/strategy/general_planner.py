from metagpt.strategy.planner import Planner

from typing import List

from metagpt.strategy.task_type import TaskType
from metagpt.utils.common import remove_comments

GENERAL_TASK_PROMPT = """
### execution result
{task_results}

## Current Task
{current_task}

### execution result
{current_task_result}

## Task Guidance
{action_per_task_descr}. Specifically, {guidance}
"""

class GeneralPlanner(Planner):
    action_per_task_descr : str = ""

    def get_plan_status(self, include_details = False, exclude: List[str] = None) -> str:
        # prepare components of a plan status
        exclude = exclude or []
        exclude_prompt = "omit here"
        finished_tasks = self.plan.get_finished_tasks()

        task_results = [task.result for task in finished_tasks]
        task_results = "\n\n".join(task_results)
        task_type_name = self.current_task.task_type
        task_type = TaskType.get_type(task_type_name)
        guidance = task_type.guidance if task_type else ""

        # combine components in a prompt
        prompt = GENERAL_TASK_PROMPT.format(
            task_results=task_results if "task_result" not in exclude else exclude_prompt,
            current_task=self.current_task.instruction,
            current_task_result=self.current_task.result if "task_result" not in exclude else exclude_prompt,
            guidance=guidance,
            action_per_task_descr=self.action_per_task_descr
        )
        
        if include_details:
            return [task_results, task_type_name, task_type, guidance], prompt
        
        return prompt