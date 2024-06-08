import os


def get_unique_proj_dir_name(base_dir, project_name):
    project_dir = os.path.join(base_dir, project_name)
    if not os.path.exists(project_dir):
        return project_dir
    counter = 1
    while os.path.exists(f"{project_dir}_{counter}"):
        counter += 1
    return f"{project_dir}_{counter}"