import pkgutil

# this is the package we are inspecting -- for example 'email' from stdlib
import etoLib

package = etoLib
for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
    print ("Found submodule %s " % (modname))
