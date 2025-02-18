from setuptools import setup, find_packages
from setuptools.command.install import install
import sys
# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

class CustomInstall(install):
    def run(self):
        install.run(self)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: setup.py [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]")
        print("   or: setup.py --help [cmd1 cmd2 ...]")
        print("   or: setup.py --help-commands")
        print("   or: setup.py cmd --help")
        print("\nerror: no commands supplied")
        sys.exit(1)

    setup(
        name="zerolm",
        version="0.1.0",
        packages=find_packages(),
        install_requires=required,
        cmdclass={
            'install': CustomInstall,
        },
    )
