from setuptools import setup

desc = 'Utility for generating a Bingo-style social game.'

if __name__ == "__main__":
    setup(
        name='mixer_bingo',
        version='0.0.1',
        description=desc,
        author='Eric J. Humphrey',
        author_email='ejhumphrey@spotify.com',
        url='',
        download_url='',
        packages=[],
        package_data={
            'mixer_bingo': ['template.tex']
        },
        long_description=desc,
        classifiers=[
            "Programming Language :: Python",
            "Development Status :: 3",
            'Environment :: Console'
        ],
        keywords='',
        license='',
        install_requires=[
            'numpy',
            'pandas',
            'networkx'
        ],
        extras_require={}
    )
