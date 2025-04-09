# SPDX-FileCopyrightText: 2022 - 2023 Peter Urban, Ghent University
#
# SPDX-License-Identifier: CC0-1.0

# -- python sources --
sources = [
  'themachinethatgoesping/__init__.py',
  'themachinethatgoesping/gridding/__init__.py',
  'themachinethatgoesping/gridding/echogrid.py',
  'themachinethatgoesping/gridding/forwardgridderlegacy.py',
  'themachinethatgoesping/gridding/forwardgridderlegacynew.py',
  'themachinethatgoesping/gridding/functions/__init__.py',
  'themachinethatgoesping/gridding/functions/gridfunctions.py',
  'themachinethatgoesping/gridding/functions/helperfunctions.py',
]

           
if get_option('build_pythonmodule').enabled()
        
    #install headers
    if get_option('pydev_install').disabled() #don't install if pydev_install is activated
        foreach source : sources
            #create install path for file
            fields = []
            foreach field : source.split('/')
                if not field.endswith('.py')
                    fields += [field]
                endif
            endforeach
            pth = '/'.join(fields)

            pymod.install_sources(source, subdir: pth, install_tag: 'python-runtime')
        endforeach

    endif

    # configure version for __init__.py
    # TODO: this does not yet work in pydev_install.enabled mode
    conf_data = configuration_data()
    conf_data.set('PROJECT_VERSION', meson.project_version())
    configure_file(input : 'themachinethatgoesping/gridding/__init__.py',
                output : '__init__.py',
                configuration : conf_data,
                install_dir : pymod.get_install_dir(subdir: 'themachinethatgoesping/gridding'), 
                install_tag : 'python-runtime',
                install: get_option('pydev_install').disabled()
                )
endif
