"""
Must install pubmed_parser.

    pip install git+git://github.com/titipata/pubmed_parser.git

Source: https://github.com/titipata/pubmed_parser
"""

import os
import pubmed_parser as pp

path = './sample/Am_J_Speech_Lang_Pathol/PMC6802915.nxml'

dict_out = pp.parse_pubmed_xml(path)

print(dict_out.keys())
