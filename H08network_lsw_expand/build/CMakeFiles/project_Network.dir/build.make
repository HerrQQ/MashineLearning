# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build

# Include any dependencies generated for this target.
include CMakeFiles/project_Network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/project_Network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/project_Network.dir/flags.make

CMakeFiles/project_Network.dir/src/testMain.cpp.o: CMakeFiles/project_Network.dir/flags.make
CMakeFiles/project_Network.dir/src/testMain.cpp.o: ../src/testMain.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/project_Network.dir/src/testMain.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/project_Network.dir/src/testMain.cpp.o -c /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/src/testMain.cpp

CMakeFiles/project_Network.dir/src/testMain.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/project_Network.dir/src/testMain.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/src/testMain.cpp > CMakeFiles/project_Network.dir/src/testMain.cpp.i

CMakeFiles/project_Network.dir/src/testMain.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/project_Network.dir/src/testMain.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/src/testMain.cpp -o CMakeFiles/project_Network.dir/src/testMain.cpp.s

# Object files for target project_Network
project_Network_OBJECTS = \
"CMakeFiles/project_Network.dir/src/testMain.cpp.o"

# External object files for target project_Network
project_Network_EXTERNAL_OBJECTS =

project_Network: CMakeFiles/project_Network.dir/src/testMain.cpp.o
project_Network: CMakeFiles/project_Network.dir/build.make
project_Network: CMakeFiles/project_Network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable project_Network"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/project_Network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/project_Network.dir/build: project_Network

.PHONY : CMakeFiles/project_Network.dir/build

CMakeFiles/project_Network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/project_Network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/project_Network.dir/clean

CMakeFiles/project_Network.dir/depend:
	cd /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build /home/kaifengqu/Desktop/H01Coding/H08WorkingDev/H08network_lsw_expand/build/CMakeFiles/project_Network.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/project_Network.dir/depend

