from setuptools import setup

setup(name="gym_openpit",
      version="1.2",
      url="./gym-openpit",
      author="Da Huo",
      license="MIT",
      packages=["gym_openpit", "gym_openpit.envs"],
      package_data = {
          "gym_openpit.envs": ["openpit_samples/*.npy"]
      },
      install_requires = ["gym", "pygame", "numpy"]
)
