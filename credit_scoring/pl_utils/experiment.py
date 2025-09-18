import os
from pathlib import Path

def create_new_run(chkp_path: str, run_alias: str):
    chkp_dir = Path(chkp_path).resolve()
    dir_list = [item for item in chkp_dir.iterdir() if item.is_dir()]
    nums = [str(dir_name).split('-')[-1] for dir_name in dir_list]
    nums = sorted([int(num) for num in nums if num.isdigit()])

    if not len(nums):
        last_num = 0
    else:
        for i in range(nums[-1] + 2):
            if i not in nums:
                last_num = i
            
    dir_name = f"{run_alias}-{str(last_num)}"
    os.system(f"mkdir {chkp_path}/{dir_name}")
    return dir_name