# SPDX-FileCopyrightText: 2022 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- python sources --
sources = [
    #'themachinethatgoesping/gridding/__init__.py', this file will be configured below
    'themachinethatgoesping/gridding/echogrid.py',
    'themachinethatgoesping/gridding/forwardgridder.py',
    'themachinethatgoesping/gridding/functions/__init__.py',
    'themachinethatgoesping/gridding/functions/gridfunctions.py',
    'themachinethatgoesping/gridding/functions/helperfunctions.py',
]

           
if get_option('build_pythonmodule').enabled()

    # get the python module
    pymod = import('python').find_installation(get_option('python_path'), required : get_option('build_pythonmodule'))
        
    #install headers
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

    # configure version for __init__.py
    # TODO: this does not yet work in pydev_install.enabled mode
    conf_data = configuration_data()
    conf_data.set('PROJECT_VERSION', meson.project_version())
    configure_file(input : 'themachinethatgoesping/gridding/__init__.py',
                output : '__init__.py',
                configuration : conf_data,
                install_dir : pymod.get_install_dir() + 'themachinethatgoesping/gridding', 
                install_tag : 'python-runtime',
                install: get_option('pydev_install').disabled()
                )
endif
