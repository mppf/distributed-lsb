#!/usr/bin/env python3

import sys
import subprocess

def count_lines_ignoring_blank(file_path):

    # read the file into a string but delete anything between
    # BEGIN_IGNORE_FOR_LINE_COUNT and END_IGNORE_FOR_LINE_COUNT

    BEGIN_IGNORE="BEGIN_IGNORE_FOR_LINE_COUNT"
    END_IGNORE="END_IGNORE_FOR_LINE_COUNT"

    file_contents = ""
    ignoring = 0
    ignored_lines = 0
    with open(file_path, 'r') as file:
        for line in file:
            found_ignore = False
            if BEGIN_IGNORE in line:
                ignoring += 1
                found_ignore = True
            if END_IGNORE in line:
                ignoring -= 1
                found_ignore = True
            if ignoring == 0 and not found_ignore:
                file_contents += line
            else:
                ignored_lines += 1

    # find the language based on the file name
    lang = "unknown"
    if file_path.endswith(".cpp"):
        lang = "C++"
    elif file_path.endswith(".c"):
        lang = "C"
    elif file_path.endswith(".chpl"):
        lang = "Chapel"
    elif file_path.endswith(".rs"):
        lang = "Rust"

    # run cloc to count ignoring comments
    result = subprocess.run(["cloc", "--csv", "--hide-rate", "--quiet",
                             "-", "--lang-no-ext="+lang],
                             input=file_contents,
                             capture_output=True, text=True,
                             check=True);

    # use the SUM code line count from cloc
    code_lines = 0
    for line in result.stdout.splitlines():
        if "SUM" in line:
            files, language, blank, comment, code = line.split(",")
            code_lines = code


    return code_lines, ignored_lines

if __name__ == "__main__":
    paths = [
             "chpl/arkouda-radix-sort-terse.chpl",
             "chpl/arkouda-radix-sort.chpl",
             "chpl/arkouda-radix-sort-strided-counts.chpl",
             "chpl/arkouda-radix-sort-strided-counts-no-agg.chpl",
             "mpi/mpi_lsbsort-terse.cpp",
             "mpi/mpi_lsbsort.cpp",
             "shmem/shmem_lsbsort-terse.cpp",
             "shmem/shmem_lsbsort.cpp",
             "shmem/shmem_lsbsort_convey.cpp",
            ]

    for p in paths:
        count, ignored_lines = count_lines_ignoring_blank(p)
        print(f"{p:50} {count} ({ignored_lines} lines ignored)")
