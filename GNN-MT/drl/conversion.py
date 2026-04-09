from pyecore.resources import ResourceSet, URI
from io import StringIO
import re
import numpy as np


class ConverterUtility(object):

    def __init__(self, mm_path):
        self.rset = ResourceSet()
        resource = self.rset.get_resource(URI(mm_path))
        mm_root = resource.contents[0]
        self.rset.metamodel_registry[mm_root.nsURI] = mm_root

    def resource_from_xmi_str(self, uri, xmi_str):
        model_string = remove_encoding_declaration(xmi_str)
        uri = StringURI(uri, text=model_string)
        resource = self.rset.create_resource(uri)
        resource.load()
        return resource


def remove_encoding_declaration(xml_data):
    # Define a regular expression pattern to match the encoding declaration
    encoding_pattern = r'<\?xml.*?encoding=["\'](.*?)["\'].*?\?>'

    # Find the encoding declaration in the XML string
    match = re.search(encoding_pattern, xml_data)

    # If the encoding declaration is found, remove it from the XML string
    if match:
        encoding_declaration = match.group(0)
        xml_data = xml_data.replace(encoding_declaration, "")

    return xml_data


def rm_illegal_acts(q_vals, legal_acts):
    m = np.ones_like(q_vals, dtype=bool)
    m[legal_acts] = False
    l_out = q_vals.copy()
    l_out[m] = -1e10
    return l_out


class StringURI(URI):
    def __init__(self, uri, text=None):
        super(StringURI, self).__init__(uri)
        if text is not None:
            self.__stream = StringIO(text)

    def getvalue(self):
        return self.__stream.getvalue()

    def create_instream(self):
        return self.__stream

    def create_outstream(self):
        self.__stream = StringIO()
        return self.__stream
