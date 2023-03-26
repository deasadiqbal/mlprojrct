from setuptools import find_packages, setup

def get_requirements(file_path):
    with open(file_path) as file_obj:
        requirements = [line.strip() for line in file_obj if not line.startswith('-')]
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='asad',
    author_email='asad95298@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
