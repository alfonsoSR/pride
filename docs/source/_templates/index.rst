PRIDE Reference
===================

This reference manual details functions, modules, and objects included in PRIDE, describing what they are and what they do.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}
