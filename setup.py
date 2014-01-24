
from setuptools import setup, find_packages

PACKAGE_NAME = "eventsim"

setup(
	name = PACKAGE_NAME,
	description = "no description",
	author = "Kyle Treleaven",
	author_email = "ktreleav@gmail.com",
	version = "0.0.0",
	packages = find_packages(),
	namespace_packages = [ 'setiptah', 'setiptah.%s' % PACKAGE_NAME, ],
)

