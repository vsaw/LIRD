#!/usr/bin/pyhton

import argparse
import os.path
import sys


INCLUDE_TOKEN_BEGINN = '${include:{'
INCLUDE_TOKEN_END = '}}'


# First the command line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--wiki', action='store', default='../LIRD.wiki',
	help='Path to the Wiki-Folder')
args = parser.parse_args()
if not os.path.exists(args.wiki) and not os.path.isdir(args.wiki):
	raise ValueError
# So far the main_page is manually added to the args, but could be added
# automatically later on
args.main_page = open(os.path.join(args.wiki, 'Final-Report.md'), 'r')


def __write(s):
	""" Writes the data to the output file

	:param s:
		The string/data to write

	So far this will be dumped to standard output but it could be redirected to
	a file later on.
	"""
	sys.stdout.write(s)


def __clean_file_name(filename):
	""" Parse the File names and cleans the github stuff from them

	:param filename:
		The name to parse

	:return:
		The clean name
	"""
	# Trim the .md extension
	ret = filename.replace('.md', '')
	# Replace '-' with ' '
	ret = ret.replace('-', ' ')
	return ret


def __parse_file(f):
	""" Parse the file and replace any includes

	:param f:
		The file to start parsing

	Recursing parsing of nested includes is possible.
	"""
	# Read the raw file
	raw_file = f.read()

	while True:
		# Look for ${include:{SOME_FILE}}
		i = raw_file.find(INCLUDE_TOKEN_BEGINN)
		if i > -1:
			# Print all the data that comes before the first match
			__write(raw_file[:i])

			# Trim raw file to only contain the stuff that has not yet been
			# printed
			raw_file = raw_file[i + len(INCLUDE_TOKEN_BEGINN):]

			# Get the file name of the included file, try to open the file and
			# recurse into it
			end_index = raw_file.find('}}')
			include_filename = raw_file[:end_index]

			with open(os.path.join(args.wiki, include_filename), 'r') as child:
				# Write the heading
				heading = __clean_file_name(include_filename)
				__write('%s\n' % heading)
				__write('=' * len(heading))
				__write('\n')

				__parse_file(child)

			# Trim the filename and continue parsing
			raw_file = raw_file[end_index + len(INCLUDE_TOKEN_END):]
		else:
			# Print the rest of the file if there is nothing left to include
			__write(raw_file)
			break


# Start parsing at the main page
if __name__ == "__main__":
	__parse_file(args.main_page)
