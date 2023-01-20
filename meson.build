# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# --- Project ---
# Define project meta data
project(
  'themachinethatgoesping_echogrids',
  'cpp',
  license: 'MPL-2.0',

  version: '0.1.2',
  default_options: ['warning_level=2', 'buildtype=release', 'cpp_std=c++17'],
  meson_version: '>=1.0' #force new meson version to limit necessary testing
)

projectnamespace = 'themachinethatgoesping'
 
# themachinethatgoesping
# tools
tools_dep = dependency('themachinethatgoesping_tools', required: false)

if not tools_dep.found()
    tools_dep = subproject('tools').get_variable('tools_dep')
endif

# -- python modules --
if get_option('build_pythonmodule').enabled()
  pymod = import('python').find_installation(get_option('python_path'))

  python3_dep = pymod.dependency()
  pybind11_dep = dependency('pybind11')
  
  tools_pybind_dep = dependency('themachinethatgoesping_tools_pybind', required: false)

  if not tools_pybind_dep.found()
      tools_pybind_dep = subproject('tools').get_variable('tools_pybind_dep')
  endif

  # set path where to install sources (pythonpath/site-packages/projectnamespace)
  pythoninstallpath = pymod.get_install_dir() + projectnamespace + '/'
  message('python module install directory:', pythoninstallpath)

endif

#themachinethatgoesping meta info
meta_dep = dependency('themachinethatgoesping_meta', required: false)
if not meta_dep.found()
    meta_dep = subproject('meta').get_variable('meta_dep')
endif

# add python src
subdir('python')

