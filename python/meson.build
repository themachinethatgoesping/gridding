# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- python sources --
sources = [
    'themachinethatgoesping/echogrids/__init__.py',
    'themachinethatgoesping/echogrids/functions/__init__.py',
    'themachinethatgoesping/echogrids/functions/gridfunctions.py',
    'themachinethatgoesping/echogrids/functions/helperfunctions.py',
]

pymod = import('python').find_installation(get_option('python_path'), required : get_option('build_pythonmodule'))
if get_option('build_pythonmodule').enabled()
    
    if get_option('pydev_install').disabled() #don't install if pydev_install is activated
        foreach source : sources
            #create install path for file
            fields = [pymod.get_install_dir()]
            foreach field : source.split('/')
                if not field.endswith('.py')
                    fields += [field]
                endif
            endforeach
            pth = '/'.join(fields)

            install_data(source, install_dir: pth, install_tag: 'python-runtime')
        endforeach

    endif
endif
