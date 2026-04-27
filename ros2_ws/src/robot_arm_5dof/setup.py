from setuptools import setup

package_name = 'robot_arm_5dof'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Robot Arm Developer',
    maintainer_email='user@example.com',
    description='5 DOF Robot Arm + YOLO + ESP32 Serial Eye-to-Hand Controller',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Node entry points will be added as nodes are implemented.
            # Placeholder stubs for Phase 1 build validation:
            'robot_arm_5dof_node = robot_arm_5dof.robot_arm_5dof_node:main',
        ],
    },
)